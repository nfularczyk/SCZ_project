import torch

from args import *

class NeuronDataset(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, data_set, labels, transforms = None):
        super().__init__()
        self.data_set = data_set
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        sample = self.data_set[index]
        #sample = self.np_data[idx]
        #sample = torch.unsqueeze(sample , 0)
        #print(sample.shape)
        label = self.labels[index]
        if self.transforms is not None:
            #print(index)
            #print(self.transforms)
            #print("transforming the sample")
            sample = self.transforms(sample)
            #print(type(sample))

        return sample, label