import torch
import torchvision
import torchvision.models as models
from torchvision.transforms import Compose, RandomOrder, RandomChoice, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
import torch.nn as nn
from torch.utils.data import DataLoader
#import torch.nn.functional as F

from extract_neuron_patches import Neuron_Patches_Dataset
import Network_classes as nc
from conv_op import weight_reg
from NeuronCNN_functions import *
from args import *
    
def main():
    #p = 20
    #s = 10
    #t_p = 0.10
    #lr = 0.5  # learning rate
    #epochs = 20  # how many epochs to train for
    #tp = 0.3 #percentage of dataset belonging to test set
    #bs = 64
    #rep_im_path = "/home/cpu/Downloads/NeuronImages/9-8-20 SCZ + HC Soma Quantification/Rep_Imgs/"
    use_cuda = torch.cuda.is_available()
    #kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    run_avg_ta = []
    run_med_ta = []
    transforms = None
    if args.data_augmentation:
        transforms = Compose([
            RandomOrder(
                [
                    RandomChoice([RandomRotation(degrees=90),
                                  RandomRotation(degrees=180),
                                  RandomRotation(degrees=270)]),
                    RandomChoice([RandomHorizontalFlip(), RandomVerticalFlip()])
                ]
        ),
                                ])
        

    loss_fn = nn.CrossEntropyLoss()
    conv_models = [args.conv_model]*args.num_layers
    #print(conv_models)
    num_layers = args.num_layers
    kernel_sizes = [(args.kernel_size, args.kernel_size)]*num_layers
    #print(kernel_sizes)
    padding = [(1, 1)]*num_layers
    #print(padding)
    stride = [(1, 1)]*num_layers
    #print(stride)
    #print(len(train_dl))
    drop_rates = [args.drop3x3]*args.num_layers
    rf_configs = [args.rf_config]*args.num_layers
    torch.backends.cudnn.benchmark = True

    print(torch.__version__)
    print(torchvision.__version__)
    for run_idx in range(1, args.num_runs+1):
        ds = Neuron_Patches_Dataset(
            root_dir = args.data_path, 
            data_class = args.data_subset, 
            patch_size = args.patch_size, 
            stride = args.patch_stride, 
            threshold_percentage = args.threshold_percentage, 
            test_percentage = args.train_test_split,
            transform = transforms) 

        train_dl, val_dl, train_ds, val_ds = get_data(ds, args.batch_size)
        test_ds = ds.torch_val_data
        patches_per_image = ds.patch_per_image
        test_labels = ds.val_labels
        args.steps_per_epoch = len(train_dl)
        
        # TODO: Rework model code
        args.run_idx = run_idx + args.start
        model = getattr(nc, args.arch_name)(
            conv_models=conv_models,
            in_channels=args.in_channels,
            input_shape=(1, args.nbands, args.patch_size, args.patch_size),
            num_classes=args.num_classes,
            dropout_prob=args.dropout_prob,
            num_kernels=args.conv_filters,
            kernel_sizes=kernel_sizes,
            padding=padding,
            stride=stride,
            groups = args.nbands,
            kernel_drop_rates=drop_rates,
            bias=True,
            bn=args.bn,
            nl=args.nonlinearity,
            eps=args.eps,
            rf_configs=rf_configs,
            fbank_type=args.fbank_type)
        if run_idx == 1:
            print(f"num train data: {len(train_ds)}")
            print(f"num val data: {len(val_ds)}")
            print(f"num test data: {len(test_ds)}")
            #print(f"num test data: {len(test_ds)}")
            print(model)

        print(f"RUN IDX: {run_idx}")
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
        if args.lr_scheduler_name == "OneCycleLR":
            print("OneCycleLR is being used")
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=args.steps_per_epoch, epochs=args.num_epochs)
        else:
            args.lr_scheduler_name = None
            scheduler = None

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)


        weight_reg_fn = weight_reg if "RF" not in args.conv_model else None
        for epoch_idx in range(1, args.num_epochs + 1):
            print(f"epoch: {epoch_idx}")
            train(model, device, train_dl, loss_fn, optimizer, args.log_interval,
                  wreg_fn=weight_reg_fn, lambda_=.001, scheduler=scheduler)
            #val_oa = val(model, device, val_dl)
            #print(f"val oa: {val_oa:.2f}%")

            # np.save("{}_val_oa.npy".format(cm), val_oa)
            #if val_oa >= args.validation_threshold:
                #break
        val_oa, val_aa, val_ca = val_full(model, device, val_dl, args.num_classes)
        print(f"val oa: {val_oa:.2f}%, val aa: {val_aa:.2f}%")
        print(val_ca)
        model_name = model_save_name(
            args.conv_model,
            args.fbank_type,
            args.rf_config,
            drop_rates[0],
            args.patch_size,
            kernel_sizes[0],
            args.conv_filters,
            args.nonlinearity,
            args.bn,
            epoch_idx,
            args.run_idx)
        print(model_name)

        test_avg_oa, test_avg_aa, test_med_oa, test_med_aa = test(model, device, test_ds, test_labels, patches_per_image, args.num_classes)
        #print(test_oa)
        #print(test_ca)
        #print(test_aa)
        #print(ds.class_count)
        print(f"test avg oa: {test_avg_oa:.2f}%, test avg aa: {test_avg_aa:.2f}%")
        print(f"test med oa: {test_med_oa:.2f}%, test med aa: {test_med_aa:.2f}%")
        run_avg_ta.append(test_avg_oa)
        run_med_ta.append(test_med_oa)

        
    test_mean_avg_oa = np.mean(run_avg_ta) 
    test_avg_oa_std = np.std(run_avg_ta)  
    test_mean_med_oa = np.mean(run_med_ta) 
    test_med_oa_std = np.std(run_med_ta)  
    print(f"The mean test average accuracy is {test_mean_avg_oa} and the standard deviation is {test_avg_oa_std} over {args.num_runs} iterations.") 
    print(f"The mean test median accuracy is {test_mean_med_oa} and the standard deviation is {test_med_oa_std} over {args.num_runs} iterations.") 



if __name__ == '__main__':
    main()
