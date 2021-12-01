import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
#import torch
#from torch import nn
#from torch import optim
#from torch.utils.data import DataLoader, Subset, sampler

from args import *
#SOFTMAX = torch.nn.Softmax(dim=1)

def load_data(data_class):
    listdir = os.listdir(args.load_path)
    #print(listdir)
    #print("The channels input")
    #print(args.channels)
    df_list = []
    channel_order = [] #list to store the order the channels are added to dataset based on loading file, to replace args.channels and ensure the correct order in build_dataset
    for csv_file in listdir:
        ds_channel = csv_file.split('_')[0]
        if (ds_channel in args.channels):
            print(ds_channel)
            channel_order.append(ds_channel)
            csv_path = os.path.join(args.load_path, csv_file)
            #csv_path = args.load_path+csv_file
            #print(csv_path)
            df = pd.read_csv(csv_path)
            df = remove_set2_417(df)
            #print(df.LABEL.value_counts())
            df = get_class_data_points(df, data_class)
            #print(df.LABEL.value_counts())
            if args.pretrained:
                print("Will use a pretrained model")
                channel_order = channel_order * args.nbands
                for i in range(args.nbands):
                    df_list.append([df,ds_channel])
            else:
                df_list.append([df, ds_channel])
    #print(df_list)
    #print(args.channels)
    #if args.pretrained:
        #channel_order = channel_order * args.nbands
    #print(channel_order)
    args.channels = channel_order
    #print(args.channels)
    return df_list
        
def remove_set2_417(data_frame):
    df = data_frame
    df_ids= df.ID
    #print(df_ids)
    to_remove=[]
    for index, value in df_ids.items():
        #find which set the data point belongs to
        temp_s=value.split('-')[0]
        #find the datapoint number
        temp_n=value.split()[2]
        if (temp_s == "Set 2" and temp_n == "417"):
    	    #print(index)
    	    #print(temp_s)
    	    #print(temp_n)
    	    #print(value)
    	    to_remove.append(index)
    #print(to_remove)
    #print(len(to_remove))
    df = df.drop(to_remove)
    #print(df)
    df = df.reset_index(drop=True)
    return df
    
def get_class_data_points(data_frame, data_class):
    df = data_frame
    if (data_class == "HC"):
        df = df[(df.LABEL==0) | (df.LABEL==1)]
    elif (data_class == "SCZ"):
        df = df[(df.LABEL==5) | (df.LABEL==6)]
    elif (data_class == "DMSO"):
        df = df[(df.LABEL==0) | (df.LABEL==5)]
    elif (data_class == "CHIR"):
        df = df[(df.LABEL==1) | (df.LABEL==6)]
    else:
        raise Exception("Error please enter one of the following four classes for argument data_class: HC, SCZ, DMSO, or CHIR")
    df = df.reset_index(drop=True)
    return df

def build_dataset(df_list):
    data=[]
    labels=[]
    channels = args.channels
    #if args.pretrained:
        #print("This is a pretrained network")
        #channels = channels * args.nbands
    print(channels)
    print(args.nbands)
    assert(len(channels) == args.nbands)
    #temporary assert condition to only use one channel
    #assert(args.nbands == 1)
    for i in range(len(df_list[0][0])):
        #print("entered build data_set loop")
        #print(i)
        #print(df_list[0][0].ID[i])
        file_prefix = df_list[0][0].ID[i].split('C=')[0]
        #print(file_prefix)
        img_channel = []
        for j in range(args.nbands):
            file_to_open = df_list[j][0].ID[i]
            tmp_file_prefix = df_list[j][0].ID[i].split('C=')[0]
            #print(channels[j])
            #print(df_list[j][1])
            assert (file_prefix == tmp_file_prefix)
            assert (channels[j] == df_list[j][1])
            img_file_name = os.path.join(args.data_path, file_to_open + ".tif")
            #img_file_name = args.data_path + file_to_open +".tif"
            #print(img_file_name)
            img = io.imread(img_file_name)
            #img = Image.open(img_file_name)
            #print(img.shape)
            img_channel.append(img)
                
        img = np.stack(img_channel, axis=2)
        #print(img.shape)
        #img = Image.merge(img_channel)
        label = num_label(df_list[0][0].LABEL[i])
        data.append(img)
        #data.append(img.convert("RGB"))
        labels.append(label)
                
    return data, labels
        
def num_label(d_label):
        
    if (args.data_subset == "SCZ" and d_label == 5):
        nl=0
    elif (args.data_subset == "SCZ" and d_label == 6):
        nl=1
    elif (args.data_subset == "DMSO" and d_label == 5):
        nl=1
    elif (args.data_subset == "CHIR" and d_label == 1):
        nl=0
    elif (args.data_subset == "CHIR" and d_label == 6):
        nl=1
    else:
        nl=d_label
    assert( nl == 0 or nl == 1)
    return nl
    

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

