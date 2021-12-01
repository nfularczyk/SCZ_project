import numpy as np
import random
import pandas as pd

from skimage import io
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from args import *

import os

class Neuron_Patches_Dataset(TensorDataset):
    """Neuron Patches dataset."""

    def __init__(self, root_dir, data_class, patch_size, stride, threshold_percentage, test_percentage, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_class (str) : Indicates the data class to loaded from the following classes: HC, SCZ, DMSO, CHIR
            patch_size (int) : height and width to generate a square patch
            stride (int) : stride to be used for overlapping or nonoverlapping patches
            threshold_percentage (float): determines the percentage of zero pixels to either accept or discard a patch
            test_percentage (float): determines the percentage of the dataset belonging to the test set
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.data_class = data_class
        self.patch_size = patch_size
        self.stride = stride
        self.threshold_percentage = threshold_percentage
        self.transform = transform
        self.df_flist = self.__load_data__(data_class)
        #self.df_flist, self.val_flist = self.build_val_set()
        #self.class_count = self.df_flist[0][0].LABEL.value_counts()
        #print(f"The class counts for the whole dataset: {self.class_count}")
        
        #self.data_set = self.__build_dataset__(self.df_flist)
        if args.val_set:
            print("will check classifiction performance on a validation set")
            self.df_flist, self.val_flist = self.build_val_set()
            self.class_count = self.df_flist[0][0].LABEL.value_counts()
            print(f"The class counts for the validation dataset: {self.class_count}")
            self.data_set = self.__build_dataset__(self.df_flist)
            self.val_list = self.__build_dataset__(self.val_flist, val_bool=True)
        else:
            print("will check classifiction performance on the whole dataset")
            self.val_flist = self.df_flist
            self.class_count = self.df_flist[0][0].LABEL.value_counts()
            print(f"The class counts for the validation dataset: {self.class_count}")
            self.data_set = self.__build_dataset__(self.df_flist)
            self.val_list = self.__build_dataset__(self.val_flist, val_bool=True)
        self.np_data, self.labels = self.__data_to_numpy__()
        self.patch_per_image, self.val_data, self.val_labels = self.__val_to_numpy__()
        if args.normalize:
            MEAN, STD, MIN, MAX = np.mean(self.np_data), np.std(self.np_data), np.amin(self.np_data), np.amax(self.np_data)
            print(f"full ds: {self.np_data.shape} {MEAN} {STD} {MIN} {MAX}")
            self.np_data = np.float32((self.np_data-MEAN) / STD)
        self.indices = np.arange(self.__len__())
        self.X_train, self.X_test, self.y_train, self.y_test, self.idx_train, self.idx_test = train_test_split(self.np_data, self.labels, self.indices, test_size = test_percentage, stratify = self.labels)
        tmp_data = torch.from_numpy(self.np_data)
        self.labels = torch.from_numpy(self.labels)
        self.torch_data = tmp_data.view(-1, args.nbands, self.patch_size, self.patch_size)
        tmp_val_data = torch.from_numpy(self.val_data)
        self.torch_val_data = tmp_val_data.view(-1, args.nbands, self.patch_size, self.patch_size)
        self.val_labels = torch.from_numpy(self.val_labels)
        
    def build_val_set(self):
        assert(len(self.df_flist) == args.nbands)
        data = []
        val = []
        #print(self.df_list[0][0])
        #print(self.df_list[0][0].LABEL.value_counts())
        for i in range(args.nbands):
            #random state is important in line below to partition each channel in the same way
            X_train, X_test, y_train, y_test= train_test_split(self.df_flist[i][0], self.df_flist[i][0].LABEL, test_size = args.val_split, stratify = self.df_flist[i][0].LABEL, random_state=42)
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            #print(X_train)
            #print(X_train.LABEL.value_counts())
            #print(X_test)
            #print(X_test.LABEL.value_counts())
            data.append([X_train, self.df_flist[i][1]])
            val.append([X_test, self.df_flist[i][1]])
            
        return data, val
        
    
    
    
    def __load_data__(self, data_class):
        listdir = os.listdir(args.load_path)
        #print(listdir)
        df_list = []
        channel_order = [] #list to store the order the channels are added to dataset based on loading file, to replace args.channels and ensure the correct order in build_dataset
        for csv_file in listdir:
            ds_channel = csv_file.split('_')[0]
            if (ds_channel in args.channels):
                #print(ds_channel)
                channel_order.append(ds_channel)
                csv_path = args.load_path+csv_file
                #print(csv_path)
                df = pd.read_csv(csv_path)
                df = self.remove_set2_417(df)
                #print(df.LABEL.value_counts())
                df = self.get_class_data_points(df, data_class)
                #print(df.LABEL.value_counts())
                df_list.append([df, ds_channel])
        #print(df_list)
        #print(args.channels)
        args.channels = channel_order
        #print(args.channels)
        return df_list
        
    def remove_set2_417(self, data_frame):
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
        
    def get_class_data_points(self, data_frame, data_class):
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
        
    def __build_dataset__(self, df_list, val_bool = False):
        data=[]
        channels = args.channels
        assert(len(channels) == args.nbands)
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
                img_file_name = self.root_dir+file_to_open+".tif"
                #print(img_file_name)
                img = io.imread(img_file_name)
                #print(img.shape)
                img_channel.append(img)
                
            img = np.stack(img_channel)
            #print(img.shape)
            label = self.__num_label__(df_list[0][0].LABEL[i])
            img_patches = self.__get_img_patches__(img, label)
            if val_bool:
                data.append([img_patches, len(img_patches)])
            else:
                data.append(img_patches)
                
        return data
        
    def __num_label__(self, d_label):
        
        if (self.data_class == "SCZ" and d_label == 5):
            nl=0
        elif (self.data_class == "SCZ" and d_label == 6):
            nl=1
        elif (self.data_class == "DMSO" and d_label == 5):
            nl=1
        elif (self.data_class == "CHIR" and d_label == 1):
            nl=0
        elif (self.data_class == "CHIR" and d_label == 6):
            nl=1
        else:
            nl=d_label
        assert( nl == 0 or nl == 1)
        return nl

#Patches code derived and modified from https://github.com/orobix/retina-unet        
    def __get_img_patches__(self, img, img_label):
    

        #extend both images and masks so they can be divided exactly by the patches dimensions
        img = self.__paint_border_overlap__(img)
        #print("\n images shape:")
        #print(img.shape)
        #print("images range (min-max): " +str(np.min(img)) +' - '+str(np.max(img)))
        #extract the patches from the full images
        patches_img = self.__extract_ordered_overlap__(img, img_label)
        #print(len(patches_img))
        
        return patches_img
    
    def __paint_border_overlap__(self, image):
        n_channels = image.shape[0]
        img_h = image.shape[1]  #height of the full image
        img_w = image.shape[2]  #width of the full image
        leftover_h = (img_h - self.patch_size) % self.stride  #leftover on the h dim
        leftover_w = (img_w - self.patch_size) % self.stride  #leftover on the w dim
        if (leftover_h != 0):  #change dimension of img_h
            #print("\n the side H is not compatible with the selected stride of " +str(self.stride))
            #print("img_h " +str(img_h) + ", patch_h " +str(self.patch_size) + ", stride_h " +str(self.stride))
            #print("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
            #print("So the H dim will be padded with additional " +str(self.stride - leftover_h) + " pixels")
            tmp_img = np.zeros((n_channels, img_h+(self.stride - leftover_h) , img_w))
            tmp_img[:, 0:img_h,0:img_w] = image
            image = tmp_img
        if (leftover_w != 0):   #change dimension of img_w
            #print("the side W is not compatible with the selected stride of " +str(self.stride))
            #print("img_w " +str(img_w) + ", patch_w " +str(self.patch_size) + ", stride_w " +str(self.stride))
            #print("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
            #print("So the W dim will be padded with additional " +str(self.stride - leftover_w) + " pixels")
            tmp_img = np.zeros((n_channels, image.shape[1],img_w+(self.stride - leftover_w)))
            tmp_img[:, 0:image.shape[1],0:img_w] = image
            image = tmp_img
        #print("new full images shape: \n" +str(image.shape))
        return image
        
    def __extract_ordered_overlap__(self, image, image_label):
        patches_ret=[]
        n_channels = image.shape[0]
        img_h = image.shape[1]  #height of the full image
        img_w = image.shape[2] #width of the full image
        delta_h = img_h - self.patch_size
        delta_w = img_w - self.patch_size
        delta_h_div_s = delta_h // self.stride + 1 #// --> division between integers
        delta_w_div_s = delta_w // self.stride + 1 #// --> division between integers
        assert (delta_h % self.stride == 0 and delta_w % self.stride == 0)
        N_patches_img = (delta_h_div_s) * (delta_w_div_s)  

        #print("Number of patches on h : " +str(delta_h_div_s))
        #print("Number of patches on w : " +str(delta_w_div_s))
        #print("Number of patches per image: " + str(N_patches_img))
        
        for h in range(delta_h_div_s):
            for w in range(delta_w_div_s):
                patch = image[:, h * self.stride : (h * self.stride) + self.patch_size , w * self.stride : (w * self.stride) + self.patch_size]
                #print("The patches shape")
                #print(patch.shape)
                if (self.__patch_zero_test__(patch)):
                    #print("patch kept")
                    patches_ret.append([patch.copy(), image_label])
                    #io.imshow(patch)
                    #io.show()
                #else:
                    #print("rejected patch")
                    #io.imshow(patch)
                    #io.show()
                    
        return patches_ret  #array with the image divided into patches
    
    def __patch_zero_test__(self, patch):
        count = 0
        ret_value = True
        threshold = self.patch_size * self.patch_size * self.threshold_percentage
        #print(threshold)
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                if (patch[0][i][j] == 0):
                    count +=1
        #print(count)
        if (count >= threshold):
            ret_value = False
        return ret_value
        
    def __data_to_numpy__(self):
        data = np.zeros((self.__len__(), args.nbands, self.patch_size, self.patch_size), dtype = np.float32)
        labels = np.zeros((self.__len__()), dtype = np.int64)
        index=0
        for patches in self.data_set:
            for patch in patches:
                data[index] = patch[0]
                labels[index] = patch[1]
                index+=1
        return data, labels
        
    def __val_to_numpy__(self):
        patch_count = 0
        #print("converting Validation to numpy")
        num_val = len(self.val_list)
        patch_per_image = []
        #print(num_val)
        index = 0
        labels = np.zeros((num_val), dtype = int)
        for patches in self.val_list:
            #print("starting computing the patches per image")
            num_patches = patches[1]
            #print(num_patches)
            #print(patches[0][0][1])
            labels[index] = patches[0][0][1]
            patch_count = patch_count + num_patches
            patch_per_image.append(num_patches)
            index+=1
        #print(labels)
        data = np.zeros((patch_count, args.nbands, self.patch_size, self.patch_size), dtype = np.float32)

        index=0
        for patches in self.val_list:
            for patch in patches[0]:
                data[index] = patch[0]
                index+=1
        return patch_per_image, data, labels
         
    def __len__(self):
        count = 0
        for patches in self.data_set:
            for patch in patches:
                count+=1
        #print("The total number of patches")
        #print(count)
        return count

    def __getitem__(self, idx):
        sample = self.torch_data[idx]
        #sample = self.np_data[idx]
        #sample = torch.unsqueeze(sample , 0)
        #print(sample.shape)
        label = self.labels[idx]
        if (self.transform and (idx in self.idx_train)):
            #print("transforming the sample")
            sample = self.transform(sample)

        return sample, label

