import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, sampler

from args import *
SOFTMAX = torch.nn.Softmax(dim=1)


def get_batch_accuracy(output, target, loss):
    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    #print(f' loss: {loss.item():.4f}    batch Acc: {100 * correct / total:.2f} %')
    print(f' loss: {loss.item()}    batch Acc: {100 * correct / total:.2f} %')


def train_step(sample, device, model, loss_fn, optimizer, batch_idx, log_interval, wreg_fn, lambda_):
    data, target = sample[0], sample[1]
    data, target = data.to(device), target.to(device)
    # Forward pass
    output = model(data)
    loss = loss_fn(output, target)
    #print(loss)
    if wreg_fn is not None:
        loss += wreg_fn(model, lambda_)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (batch_idx+1) % log_interval == 0:
        #print(loss)
        get_batch_accuracy(output, target, loss)


def train(model, device, train_loader, loss_fn, optimizer, log_interval,
          wreg_fn=None, lambda_=.001, scheduler=None):
    model.train()  # This should always be here!
    for batch_idx, sample in enumerate(train_loader):
        train_step(sample, device, model, loss_fn, optimizer, batch_idx, log_interval, wreg_fn, lambda_)
        if scheduler.__class__.__name__ == "OneCycleLR":
            scheduler.step()


def val(model, device, val_loader):
    """Get overall validation accuracy"""
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, sample in enumerate(val_loader):
            images, labels = sample[0], sample[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        val_oa = 100 * correct / total
    return val_oa


def val_full(model, device, val_loader, num_classes):
    """Get overall accuracy, class accuracies, and average accuracy"""
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, sample in enumerate(val_loader):
            images, labels = sample[0], sample[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = SOFTMAX(outputs)
            #print(outputs.data)
            #_, preds = torch.max(outputs.data, 1)
            _, preds = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        val_oa = 100 * correct / total
        val_ca = (confusion_matrix.diag() / confusion_matrix.sum(1)).numpy() * 100
        val_aa = np.mean(val_ca)
        #print(confusion_matrix)
        # print(val_ca.shape)
        # for i, item in enumerate(label_values):
        #     print(item, val_pca[i])
        # print('val oa: {}%\tval aa: {}%'.format(val_oa, val_aa))
    return val_oa, val_aa, val_ca


def test(model, device, test_set, test_labels, num_patchs_per_img, num_classes):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    index = 0
    labels_index = 0
    #print(test_set.shape)
    #print(num_patchs_per_img)
    correct_avg = 0
    correct_med = 0
    total = len(test_labels)
    #print("printing the test labels and the total number of test labels")
    #print(test_labels)
    #print(total)
    confusion_matrix_avg = torch.zeros((num_classes, num_classes))
    confusion_matrix_med = torch.zeros((num_classes, num_classes))
    err_tbl = []
    err_tbl.append(["correct_label", "predicated_avg_label", "predicted_med_label", "prob_mean", "prob_median", "patch_prob"])
    #probs_tbl = None
    with torch.no_grad():
        for num_patches in num_patchs_per_img:
            #print(num_patches)
            label = test_labels[labels_index]
            probs_lst = None
            #print(label)
            for i in range(num_patches):
                #print(index)
                sample = torch.unsqueeze(test_set[index] , 0)
                #print(sample.shape)
                patch = sample.to(device)
                patch_output = model(patch)
                probs_p_out = SOFTMAX(patch_output)
                if probs_lst == None:
                    probs_lst = probs_p_out.data
                else:
                    probs_lst = torch.cat([probs_lst, probs_p_out.data], dim = 0)
                
                index+=1
            #print("The predication was incorrect")
            #print(f"the patch probabilties: {probs_lst}")
            #print(f"The prediction: {pred.item()} and the label:{label.item()}")
            med, _ = torch.median(probs_lst, 0)
            avg = torch.mean(probs_lst, 0)
            #print(f" the mean: {avg.data} and the median:{med.data}")
            #patch_err = []
            tmp_probs = probs_lst.cpu().numpy()
            #print("about to print the patch probabilities being reshaped")
            #print(tmp_probs)
            #l, w = tmp_probs.shape
            #for j in range(l):
                #patch_err.append(tmp_probs[j])
            #print(patch_err)

            #if probs_tbl == None:
                #probs_tbl = probs_img.data
            #else:
                #probs_tbl = torch.cat([probs_tbl, probs_img.data], dim = 0)
            _, pred_avg = torch.max(torch.unsqueeze(avg,0),1)
            _, pred_med = torch.max(torch.unsqueeze(med,0),1)
            #print("printing the prediction first, avg then med, then the correct label for the image")
            #print(pred_avg.item())
            #print(pred_med.item())
            #print(label.item())
            err_tbl.append([label.item(), pred_avg.item(), pred_med.item(), avg.cpu().numpy(), med.cpu().numpy(), tmp_probs])
            if (pred_avg.item() == label.item()):
                correct_avg+=1
            if (pred_med.item() == label.item()):
                correct_med+=1
            confusion_matrix_avg[label.item(), pred_avg.item()] += 1
            confusion_matrix_med[label.item(), pred_med.item()] += 1
            labels_index +=1
        test_avg_oa = 100 * correct_avg / total
        test_med_oa = 100 * correct_med / total
        #print(confusion_matrix_avg)
        #print(confusion_matrix_avg.sum(1))
        #print(confusion_matrix_med)
        #print(confusion_matrix_med.sum(1))
        df_e = pd.DataFrame(err_tbl)
        #df_e.to_csv('probs_error.csv')
        #print(f"printing the error table: {err_tbl}")
        #print(f"printing the pandas error table: {df_e}")
        #print(args.save_path)
        test_avg_ca = (confusion_matrix_avg.diag() / confusion_matrix_avg.sum(1)).numpy() * 100
        test_avg_aa = np.mean(test_avg_ca)
        test_med_ca = (confusion_matrix_med.diag() / confusion_matrix_med.sum(1)).numpy() * 100
        test_med_aa = np.mean(test_med_ca)
        #prob_avg = torch.mean(probs_tbl, 0)
        #prob_med = torch.median(probs_tbl, 0)
        #print(f"the probability results for each individual soma: {probs_tbl} and the mean of these probabilites followed by the median {prob_avg} {prob_med}")
        #print(test_med_oa)
        #print(test_med_ca)
        #print(test_med_aa)
        #print(f"test med oa: {test_med_oa:.2f}%, test med aa: {test_med_aa:.2f}%")
        return test_avg_oa, test_avg_aa, test_med_oa, test_med_aa

    
def get_data(data_set, bs):
    train_idx = data_set.idx_train
    #print(train_idx.shape)
    test_idx = data_set.idx_test
    train_labels = data_set.y_train
    # uniform sampling
    weights = torch.as_tensor(make_weights_for_balanced_classes(train_labels), dtype=torch.double)
    sampler_fn = sampler.WeightedRandomSampler(weights, len(weights))

    #print(test_idx.shape)
    print('train -  {}   |   test -  {}'.format(np.bincount(data_set.y_train), np.bincount(data_set.y_test)))
    train_ds = Subset(data_set, train_idx)
    #print(len(train_ds))
    valid_ds = Subset(data_set, test_idx)
    return (
        DataLoader(train_ds, batch_size=bs, sampler = sampler_fn),
        DataLoader(valid_ds, batch_size=bs * 2),
        train_ds,
        valid_ds
    )

def make_weights_for_balanced_classes(labels):
    if isinstance(labels, np.ndarray):
        #print(labels.shape)
        #print(labels)
        #labels = labels[np.nonzero(labels)]-shift
        classes, counts = np.unique(labels, return_counts=True)
    elif isinstance(labels, list):
        classes, counts = np.unique(np.array(labels), return_counts=True)
    nclasses = len(classes)
    #print(nclasses)
    #print("classes: ", classes)
    #print("counts: ", counts)
    weight_per_class = [0.] * nclasses

    #print(len(weight_per_class))
    N = np.float32(np.sum(counts))
    #print("N: ", N)
    for i in range(nclasses):
        weight_per_class[i] = N / np.float32(counts[i])
    weight = [0] * int(N)
    #print(weight_per_class)
    #print(weight)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight

