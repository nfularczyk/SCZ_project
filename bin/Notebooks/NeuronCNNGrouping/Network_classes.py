import itertools
import numpy as np

import torch
import torch.nn as nn

from conv_op import BasicConv, SpectralBlock, SpatialBlock
from convrf import *

"""The three custom models used in the paper:
A Multiscale Deep Learning Approach for High-Resolution Hyperspectral Image Classification
https://ieeexplore.ieee.org/abstract/document/8970377
"""


class CNN(nn.Module):
    def __init__(self,
                 conv_models,
                 in_channels,
                 input_shape,
                 num_classes,
                 dropout_prob,
                 num_kernels,
                 kernel_sizes,
                 padding,
                 stride,
                 kernel_drop_rates,
                 bias=True,
                 groups = 1,
                 bn=None,
                 nl=None,
                 eps=1e-5,
                 rf_configs=None,
                 fbank_type='frame'):

        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        assert len(num_kernels) == len(kernel_sizes)
        self.num_layers = len(num_kernels)
        assert (self.num_layers >= 1)
        # padding = [(item-1)//2 for item in kernel_sizes]

        self.c1 = BasicConv(in_channels, num_kernels[0], kernel_sizes[0], bias=bias,
                            padding=padding[0],
                            stride=stride[0],
                            bn=bn, nl=nl, eps=eps,
                            conv_model=conv_models[0],
                            groups = groups,
                            kernel_drop_rate=kernel_drop_rates[0],
                            rf_config=rf_configs[0], fbank_type=fbank_type)
        if self.num_layers > 1:
            self.layers = nn.Sequential(
                *[BasicConv(num_kernels[j-1], num_kernels[j], kernel_sizes[j], bias=bias,
                            padding=padding[j],
                            stride=stride[j],
                            bn=bn, nl=nl, eps=eps,
                            conv_model=conv_models[j],
                            groups = groups,
                            kernel_drop_rate=kernel_drop_rates[j],
                            rf_config=rf_configs[j], fbank_type=fbank_type)
                  for j in range(1, self.num_layers)])

        self.num_features = self.get_num_features()
        self.dropout = nn.Dropout(p=dropout_prob, inplace=False)
        self.linear = nn.Linear(self.num_features, self.num_classes)

    def get_num_features(self):
        with torch.no_grad():
            x = torch.zeros(self.input_shape);  # print("x: ", x.size())
            x = self.c1(x);  # print("x: ", x.size())
            if self.num_layers > 1:
                x = self.layers(x);  # print("x: ", x.size())
            num_features = int(np.prod([*x.size()][1:]));  # print("feature size: ", num_features)
            return num_features

    def forward(self, x):
        x = self.c1(x)
        if self.num_layers > 1:
            x = self.layers(x)
        x = self.dropout(x)
        # print(f"x: {x.size()}")
        x = x.reshape(-1, self.num_features)
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.dp1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(1024, in_channels)
        self.dp2 = nn.Dropout(.5)
        self.fc3 = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x


class FDSSC_mod(nn.Module):
    def __init__(
            self,
            patch_size=3,
            in_channels=1,
            nbands=48,
            num_classes=19,
            kernel_size_2d=3,
            dp=.5,
            nl="PReLU",
            avg_pool=True,
            eps=1e-5,
            conv_model1="Conv3d",
            conv_model2="Conv2d",
            rf_config=0,
            fbank_type='frame',
            kernel_drop_rate=0
    ):
        super(FDSSC_mod, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.nbands = nbands
        self.num_classes = num_classes
        self.kernel_size_2d = kernel_size_2d
        self.dp = dp
        self.nl = nl
        self.avg_pool = avg_pool

        if self.nl is not None:
            self.nonlin1 = getattr(nn, self.nl)()
            self.nonlin2 = getattr(nn, self.nl)()

        # self.bn3d = nn.BatchNorm3d(60, eps=0.001)
        # self.bn2d = nn.BatchNorm2d(60, eps=0.001)

        self.spectral = SpectralBlock(self.in_channels,
                                      None, nl, eps,
                                      conv_model1,
                                      )

        self.pooling_size = (self.nbands - 1) // 2 + 1  # 24 for Uh2018, 72 for uh2013, 52 for paviaU
        self.tran = BasicConv(60, 200, (self.pooling_size, 1, 1), bias=True,
                              bn=None, nl=self.nl, eps=eps,
                              conv_model=conv_model1)
        # self.nbands*60
        self.spatial = SpatialBlock(200, self.kernel_size_2d,
                                    bn=None, nl=self.nl, eps=eps,
                                    conv_model=conv_model2,
                                    rf_config=rf_config,
                                    fbank_type=fbank_type,
                                    kernel_drop_rate=kernel_drop_rate)

        self.dropout = nn.Dropout(self.dp)
        _, pool_size_0, pool_size_1 = self.get_pool_size()
        self.avgpool3d = nn.AvgPool3d(kernel_size=(1, pool_size_0, pool_size_1))
        self.num_features = self.get_num_features()
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def same_padding(self, i, k, s, d):
        return int(((i - 1) * s - i + k + (k - 1) * (d - 1)) / 2)

    def get_features(self, x):
        x = self.spectral(x);  # print('x.size(): ', x.size())
        if self.nl is not None:
            x = self.nonlin1(x)
        x = self.tran(x);  # print('x.size(): ', x.size())
        x = torch.squeeze(x, 2);  # print('x.size(): ', x.size())
        x = self.spatial(x);  # print('x.size(): ', x.size())
        if self.nl is not None:
            x = self.nonlin2(x);  # print('x.size(): ', x.size())
        return x

    def get_pool_size(self):
        with torch.no_grad():
            batch_size = 1
            x = torch.zeros((batch_size, self.in_channels, self.nbands, self.patch_size, self.patch_size))
            x = self.get_features(x)
            pool_size_0, pool_size_1 = x.size()[2], x.size()[3]
            return x, pool_size_0, pool_size_1

    def get_num_features(self):
        with torch.no_grad():
            out, _, _ = self.get_pool_size()
            if self.avg_pool:
                out = self.avgpool3d(out)
            out = out.reshape(out.size(0), -1)
            out = self.dropout(out)
            _, h = out.size()
        return h

    def forward(self, x):
        x = self.get_features(x)
        # x = torch.squeeze(x) Not needed in the case of 2D convolutions
        if self.avg_pool:
            x = self.avgpool3d(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropx(x)
        x = self.fc(x)
        return x
