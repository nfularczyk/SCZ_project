import numpy as np 
#import pandas as pd 

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, sampler

from sklearn.model_selection import train_test_split

import gc

from torchvision.transforms.transforms import ToTensor

from args import *
from ViT import *
from NeuronDataset import *
from ViT_functions import *

def main():
    IMG_SIZE = 224
    #BATCH_SIZE = 16
    LR = 2e-05
    #GAMMA = 0.7
    #N_EPOCHS = 10

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(IMG_SIZE),
            #transforms.Normalize((0.485,0.456, 0.406),(0.229, 0.224, 0.225)),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            #transforms.Normalize((0.485,0.456, 0.406),(0.229, 0.224, 0.225)),
        ]
    )


    #train_losses =[]
    #valid_losses = []
    #train_accs = []
    valid_accs = []
    print("Pretrained argument")
    print(args.pretrained)
    #if args.pretrained:
        #print("We will use a pretrained model")
        #print(args.channels)
        #args.channels = args.channels*args.nbands
        #print(args.channels)
    for run_idx in range(1, args.num_runs+1):
        print(f"run: {run_idx}")
        best_valid_acc = 0
        df_flist = load_data(args.data_subset)
        data_set, labels = build_dataset(df_flist)
        #print(data_set)
        #print(labels)
        indices = np.arange(len(data_set))
        X_train, X_valid, y_train, y_valid, idx_train, idx_test = train_test_split(data_set, labels, indices, test_size = args.train_test_split, stratify = labels)
        print('train -  {}   |   test -  {}'.format(np.bincount(y_train), np.bincount(y_valid)))
        #model = ViTBase16(n_classes=2, n_channels = args.nbands, pretrained=True)
        print("This is the input for the pretrained argument")
        print(args.pretrained)
        model = ViTBase16(n_classes=2, n_channels = args.nbands, pretrained=args.pretrained)
        model.to(device)
        train_dataset = NeuronDataset(X_train, y_train, transforms = transforms_train)
        valid_dataset = NeuronDataset(X_valid, y_valid, transforms = transforms_valid)

        weights = torch.as_tensor(make_weights_for_balanced_classes(y_train), dtype=torch.double)
        sampler_fn = sampler.WeightedRandomSampler(weights, len(weights))
        train_dl = DataLoader(train_dataset, batch_size=args.batch_size, sampler = sampler_fn)
        val_dl = DataLoader(valid_dataset, batch_size=args.test_batch_size)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        for epoch_idx in range(1, args.num_epochs + 1):
            if epoch_idx == 1:
                print(model)
            print(f"epoch: {epoch_idx}")
            train_loss, train_acc = model.train_one_epoch(train_dl, loss_fn, optimizer, device)
            #train_losses.append(train_loss)
            #train_accs.append(train_acc)
            print(f' train loss: {train_loss}    train Acc: {train_acc} %')
            valid_loss, valid_acc = model.validate_one_epoch(val_dl, loss_fn, device)
            #valid_losses.append(valid_loss)
            if (valid_acc > best_valid_acc):
                print(f"updating best valid acc from {best_valid_acc} to {valid_acc}")
                best_valid_acc = valid_acc
            
            print(f' valid loss: {valid_loss}    valid Acc: {valid_acc} %')
            #gc.collect()
            #torch.cuda.empty_cache()
        valid_accs.append(best_valid_acc.cpu().numpy())
    print(valid_accs)
    avg_acc = np.mean(valid_accs)
    std = np.std(valid_accs)
    print(f"The mean accuracy is {avg_acc} with standard deviation of {std} over {len(valid_accs)} iterations")
if __name__ == '__main__':
    main()
