#import numpy as np 

import torch
import torch.nn as nn
#import torchvision.transforms as transforms

import timm


class ViTBase16(nn.Module):
    def __init__(self, n_classes, n_channels, pretrained=False):
        super(ViTBase16, self).__init__()
        if pretrained:
            print("Using a pretrained ViT model")
            #self.model = timm.create_model("vit_base_patch16_224", pretrained = True)
            self.model = timm.create_model("vit_small_patch16_224", pretrained = True)
            #self.model = timm.create_model("vit_small_patch32_224", pretrained = True)
            self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        else:
            print("Using a non-pretrained ViT model")
            #self.model = timm.create_model("vit_base_patch16_224", pretrained = False)
            #self.model = timm.create_model("vit_small_patch32_224", pretrained = True)
            self.model = timm.create_model("vit_small_patch16_224", pretrained = False, in_chans = n_channels, num_classes = n_classes)

        

    def forward(self, x):
        x = self.model(x)
        return x

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        #keep track of training loss
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        ###################
        # train the model #
        ###################
        self.model.train()
        for i, (data, target) in enumerate(train_loader):
            #print(data)
            #print(target)
            #move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.to(device), target.to(device)
                
            #clear the gradients of all optimized variables
            optimizer.zero_grad()
            #forward pass: compute predicted outputs by passing inputs to the model
            output = self.forward(data)
            #calculate the batch loss
            loss = criterion(output, target)
            #backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            #Calculate Accuracy
            accuracy = (output.argmax(dim=1)==target).float().mean()
            #print(i)
            #print(accuracy)
            #update training loss and accuracy
            epoch_loss += loss
            epoch_accuracy += accuracy
            
            #perform a single optimization step(parameter update)
            optimizer.step()

        return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)
    
    def validate_one_epoch(self, valid_loader, criterion, device):
        #keep track of validation loss
        valid_loss = 0.0
        valid_accuracy = 0.0

        ######################
        # validate the model #
        ######################
        self.model.eval()
        for data, target in valid_loader:
            #move tensors to GPU if CUDA is available
            #print(device.type)
            if device.type == "cuda":
                #print(type(data))
                #print(type(target))
                data, target = data.to(device), target.to(device)
            with torch.no_grad():
                #forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                #calculate the batch loss
                loss = criterion(output, target)
                #Calculate Accuracy
                accuracy = (output.argmax(dim=1)==target).float().mean()
                #update average validation loss and accuracy
                valid_loss += loss
                valid_accuracy += accuracy

        return valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)
