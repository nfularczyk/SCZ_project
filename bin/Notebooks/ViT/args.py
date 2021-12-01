import argparse
import os
from sys import argv
from pathlib import Path, PosixPath
from torch.cuda import device_count
from datetime import datetime


NAMES2ABVS = \
    {
        "Conv2d": "c2d",
        "Conv2dRF": "c2drf",
        "Conv3d": "c3d",
        "Conv3dRF": "c3drf",
        "OneCycleLR": "oclr",
        "CyclicLR": "clr",
        "MultiStepLR": "mslr",
        "CosineAnnealingLR": "calr",
        "CosineAnnealingWarmRestarts": "cawr",
        None: "n",
        ###############################################
        # not used for naming just for your information
        0: "only_rf",
        1: "only_1st",
        2: "half_mix",
        3: "randomly_mix",
        4: "randomly_rf",
        "pframe": "pfr",
        "frame": "fr",
        "nn_bank": "nn"
        ###################################################
    }

ABVS2NAMES = dict(zip(NAMES2ABVS.values(), NAMES2ABVS.keys()))


def formatFloat(val, format):
    ret = f"{val:{format}}"
    if ret == "0.0":
        return "0"
    else:
        if ret.startswith("0."):
            return ret[1:]
        if ret.startswith("-0."):
            return "-" + ret[2:]
        return ret


def model_save_name(conv_model, fbank_type, rf_config, drop_rate, patch_size,
                   kernel_size, num_kernelss, nonlin, bn, epoch_idx, run_idx):
    fbank_type = f"{NAMES2ABVS[fbank_type]}" if "RF" in conv_model else "n"
    rf_config = f"{rf_config}" if "RF" in conv_model else "n"
    drop_rate = f"{formatFloat(drop_rate, '.1f')}" if "RF" in conv_model else "n"
    batchnorm = "bn" if bn is not None else "n"
    kernel_size = "x".join([str(size) for size in kernel_size])
    num_kernelss = "-".join([str(num_kernels) for num_kernels in num_kernelss])
    cm = NAMES2ABVS[conv_model]
    model_name = \
        f"{cm}_" \
        f"{fbank_type}_"\
        f"{rf_config}_"\
        f"{drop_rate}_" \
        f"{patch_size}_" \
        f"{kernel_size}_" \
        f"{num_kernelss}_" \
        f"{nonlin.lower()}_" \
        f"{batchnorm}_" \
        f"{args.validation_threshold}_" \
        f"{NAMES2ABVS[args.lr_scheduler_name]}_"\
        f"{epoch_idx}_"\
        f"{run_idx}"
    return model_name


def get_parser():
    parser = argparse.ArgumentParser(description="getting dataset arguments")
    #################################################################
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--param_count', type=bool, default=False)
    parser.add_argument('--full_val', type=bool, default=False)

    parser.add_argument('--num_classes', type=int, default=2, help="number of distinct data classes")
    parser.add_argument('--data_subset', type=str, default="CHIR", help="Perform classification in the following subset of the whole dataset")
    parser.add_argument('--pretrained', type=bool, default=False, help="To use a pretrained network or not")
    parser.add_argument('--in_channels', type=int, default=2)
    ########################################################################################
    # save options
    # parser.add_argument('--log_freq', type=int, default=1250,
    # help="how often to print training and validation accuracies/losses")
    parser.add_argument('--save_path', type=str, default = os.path.join(os.getcwd(),""))
    parser.add_argument('--load_path', type=str, default = os.path.join(os.getcwd(),"..","..","..","","r","radiomics",""), help ="filepath to directory of where to load radiomics features")
    parser.add_argument('--data_path', type=str, default = os.path.join(os.getcwd(),"..","..","..","","Rep_Imgs",""), help = "filepath from where to load neuron dataset")   
    #parser.add_argument('--load_path', type=str, default="/home/cpu/Downloads/NeuronImages/9-8-20 SCZ + HC Soma Quantification/r/radiomics/", help="filepath to directory of where to load radiomics features")
    #parser.add_argument('--data_path', type=str, default="/home/cpu/Downloads/NeuronImages/9-8-20 SCZ + HC Soma Quantification/Rep_Imgs/", help="filepath from where to load neuron dataset")
    #parser.add_argument('--model_path', type=str, default = os.path.join(os.getcwd(),"..","..","..","","Rep_Imgs",""), help = "filepath to a pretrained model such as ViT" )
    
    #####################################################################

    parser.add_argument('--model_dim', type=int, default=3)
    parser.add_argument('--nbands', type=int, default=2, help="number of channels in the input image")
    #parser.add_argument('--channels', type=list, default = ["TUBB","FGF"], help="determines channels of input image")
    parser.add_argument('--channels', nargs='+', default = ["TUBB","FGF"], help="determines channels of input image")
    parser.add_argument('--patch_size', type=int, default=20)
    parser.add_argument('--patch_stride', type=int, default=10)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout_prob', type=float, default=.5)
    parser.add_argument('--conv_filters', type=list, default=[16,32,64])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--nonlinearity', type=str, default="ReLU", help="Nonlinearity used in the network.")
    #parser.add_argument('--bn', type=str, default="BatchNorm2d", help="Batch normalization used.")
    parser.add_argument('--bn', type=str, default=None, help="Batch normalization used.")
    # transforms
    #parser.add_argument('--normalize', type=bool, default=True, help="whether to normalize the dataset as a preprocessing step")
    parser.add_argument('--normalize', type=bool, default=False, help="whether to normalize the dataset as a preprocessing step")

    ######################################################################
    # model training options
    batch_size = 16 if device_count() == 0 else 16 * device_count()
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=2*batch_size)
    #parser.add_argument('--steps_per_epoch', type=int)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--data_augmentation', type=bool, default=True)

    parser.add_argument('--run_idx', type=int, default=0)
    parser.add_argument('--date', type=str, default=datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))
    ###################################################################
    # optimizer options
    parser.add_argument('--optimizer_name', type=str, default="Adam", help="SGD or Adam")
    parser.add_argument('--base_lr', type=float, default=.001, help=".1 for SGD and .001 for Adam")
    parser.add_argument('--weight_decay', type=float, default=3e-6)

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)  
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--validation_threshold', type=int, default=100, help="option to stop training if accuracy on validation set reaches a certain percentage")
    parser.add_argument('--threshold_percentage', type=float, default=0.1)
    parser.add_argument('--train_test_split', type = float, default=0.3)
    parser.add_argument('--val_set', type = bool, default=False, help="whether to isolate a portion of the data for validation")
    parser.add_argument('--val_split', type = float, default=0.1)
    parser.add_argument('--lr_scheduler_name', type=str, default="OneCycleLR")
    #parser.add_argument('--lr_scheduler_name', type=str, default=None)
    parser.add_argument('--anneal_strategy', type=str, default="cos")
    parser.add_argument('--base_momentum', type=float, default=.8)
    parser.add_argument('--max_momentum', type=float, default=.85)
    parser.add_argument('--pct_start', type=float, default=.15)
    parser.add_argument('--max_lr', type=float, default=.01)
    parser.add_argument('--div_factor', type=float, default=25)
    parser.add_argument('--final_div_factor', type=float, default=1e4)
    ######################################################################
    # convolutional model and architecture options
    parser.add_argument('--arch_name', type=str, default="CNN")
    parser.add_argument('--conv_model', type=str, default="Conv2d",
                        help="use Conv1d, Conv2d, Conv3d, Conv2dRF, Conv3dRF")
    parser.add_argument('--custom_head', type=str, default=None, help="use a custom pooling layer at the end.")
    # rf setup Directional Receptive Field (DRF) Hyperparameters
    parser.add_argument('--drop7x7', type=float, default=0,
                        help="In each DRF convolutional layer, only (1-drop7x7)% of "
                             "the total available 7x7 filters will be used.")
    parser.add_argument('--drop5x5', type=float, default=0, )
    parser.add_argument('--drop3x3', type=float, default=0, )
    parser.add_argument('--drop3x3x3', type=float, default=0, )
    parser.add_argument('--fbank_type', type=str, default="frame", )

    parser.add_argument('--rf_config', type=int, default=0,
                        help=
                        "0 : only_rf ____________ All the conv layers of the network are DRF."
                        "1 : only_1st ___________ Only the first conv layer of the network is DRF."
                        "2 : half_mix ___________ In each conv layer, half of the filters are set to DRF and "
                        "the rest as conventional"
                        "3 : randomly_mix _______ In each conv layer, a random selection of filters are DRF,"
                        "the rest are conventional. "
                        "Must fix numpy random number generator seed in distributed parallel training."
                        "4 : randomly_rf ________ Each conv layer will be randomly set to either "
                        "conventional or DRF. Must fix numpy random number generator seed in DDP training.")
    #################################################################################
    # distributed training options
    # parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=100)  # 17
    parser.add_argument("--local_rank", type=int)

    return parser


args, _ = get_parser().parse_known_args()
assert 0 <= args.drop3x3 < 1, "args.drop3x3 has to be a float between 0 and 1!"
assert 0 <= args.drop5x5 < 1, "args.drop5x5 has to be a float between 0 and 1!"
assert 0 <= args.drop7x7 < 1, "args.drop7x7 has to be a float between 0 and 1!"
assert 0 <= args.drop3x3x3 < 1, "args.drop3x3x3 has to be a float between 0 and 1!"
#################################################################
