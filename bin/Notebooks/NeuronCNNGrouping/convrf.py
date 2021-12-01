import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from parseval import Parseval
from args import *
# import sys
# from pathlib import Path
# from time import time

# from random import sample
# from numpy.random import default_rng
# rng = default_rng()
# rng does not generalize to distributed training since the initial state
# of the different processes are not identical even when fixing the random seed using
# np.random.seed(seed) # We can use np.random.choice instead which does not have such issues.

"""
Written and copywrite by Kazem Safari. 
Class 'Parseval' is borrowed from Nikolaos Karantzas' Github with his permission.
To learn more about the filter bank and its properties, please refer to this paper and thesis:

On the design of multi-dimensional compactly supported Parseval framelets with directional characteristics
https://www.sciencedirect.com/science/article/abs/pii/S0024379519303155

Compactly Supported Frame Wavelets and Applications
https://uh-ir.tdl.org/handle/10657/4698

Many thanks to Mozahid Haque for his insightful comments.
"""


class _FilterBank(object):
    def __init__(self, dim=2):
        if dim == 2:
            kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        elif dim == 3:
            # kernel_sizes = [(3, 3, 3), (3, 3, 5), (3, 3, 7), (5, 5, 5), (7, 7, 7)]
            kernel_sizes = [(3, 3, 3), (3, 5, 5), (3, 7, 7)]

        self.frame = self.get_fb("frame", kernel_sizes)
        self.pframe = self.get_fb("pframe", kernel_sizes)
        self.nn_bank = self.get_fb("nn_bank", kernel_sizes)

    def get_fb(self, fbank_type, shapes):
        return {
            self.shape2str(item):
                np.float32(Parseval(
                    shape=item,
                    low_pass_kernel='gauss',
                    first_order=True,
                    second_order=True,
                    bank=fbank_type).fbank())
            for item in shapes
        }

    def shape2str(self, shape):
        return 'x'.join([f'{dim}' for dim in shape])


FilterBank2D = _FilterBank(dim=2)
# FilterBank3D = _FilterBank(dim=3)
# print(FilterBank.pframe['7x7'].shape)
# print(FilterBank.frame)
# print(sys.getrefcount(FilterBank))
# print(sys.getrefcount(FilterBank.nn_bank))


class _ConvNdRF(_ConvNd):
    """borrowed from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    The only new argument that is necessary to be added to _ConvNd's arguments is kernel_drop_rate.

    The kernel_span tensor defined in forward replaces the nn._ConvNd's weight tensor by
    a linear combination of predefined filters in each of its convolutional channels.

    Therefore there are two ingrdients to it: self.weight and self.kernels

    1) self.weight is a tensor that defines the coefficients used in such linear combinations.
    2) self.kernels is another tensor that defines the vectors/filters used in such linear combinations.

    Now there are two cases when writing such self.kernel_span:
    1) All the filters present in fbank are used in each linear combination per convolutional channel.
    2) A random subset of fbank are used.

    The 'kernels' tensor is a non-trainable parameter that should be saved and restored in the state_dict,
    therefore we register them as buffers. Buffers won’t be returned in model.parameters()

    # According to ptrblck's comments,
    .detach() prevents cpu memory leak from the "self.kernels" buffer in "forward".

    The following links were helpful and used in building this package:
    for memory leak issues:
    https://github.com/pytorch/pytorch/issues/20275
    https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/15

    difference between .data and .detach:
    https://github.com/pytorch/pytorch/issues/6990

    for masking the gradient:
    https://discuss.pytorch.org/t/update-only-sub-elements-of-weights/29101/2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, dim, kernel_drop_rate=0, fbank_type="frame"):

        super(_ConvNdRF, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode)
        self.kernel_drop_rate = kernel_drop_rate
        self.fbank_type = fbank_type
        self.dim = dim
        if self.dim not in [2, 3]:
            raise ValueError(f"dim value must be either 2 or 3. The value {self.dim} is not supported")
        # print(self.gain)
        if self.fbank_type not in ["nn_bank", "frame", "pframe"]:
            raise ValueError(f"fbank_type values must be one of the following: 'nn_bank', 'frame', 'pframe' "
                             f"but is input as {self.fbank_type}.")
        if self.kernel_drop_rate >= 1 or self.kernel_drop_rate < 0:
            raise ValueError(f"Can't drop all kernel. "
                             f"kernel_drop_rate must be a value strictly less than 1, "
                             f"But found {self.kernel_drop_rate}.")
        if 1 in self.kernel_size:
            raise ValueError("Cannot have any of kernel dimensions equal to 1.")

        if self.kernel_drop_rate == 0:
            self.get_all_kernels()
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.total_kernels))
        else:
            self.get_some_kernels()
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.num_kernels))

        self.reset_parameters()

    def get_all_kernels(self, ):
        """(num_filters, height, width)"""
        # this will load a custom pre-designed kernel_size filter-bank
        fbank = np.float32(self.get_filterbank())
        # print(fbank.shape)

        if self.dim == 2:
            assert fbank.ndim == 3, "dimensions has to be 3, but found {}".format(fbank.ndim)
        elif self.dim == 3:
            assert fbank.ndim == 4, "dimensions has to be 4, but found {}".format(fbank.ndim)

        self.total_kernels = fbank.shape[0]
        self.num_kernels = int((1-self.kernel_drop_rate) * self.total_kernels)
        # print(f"nk: {self.num_kernels}")
        # torch.tensor() always copies data. To avoid copying the numpy array, use torch.as_tensor() instead.
        fbank = torch.as_tensor(fbank)
        self.register_buffer("kernels", fbank)

    def get_some_kernels(self, ):
        """In order to impose the sparsity constraint, we randomly select a subset of the kernels in the
        fbank instead of using all of them."""
        fbank = np.float32(self.get_filterbank())

        if self.dim == 2:
            assert fbank.ndim == 3, "dimensions has to be 3, but found {}".format(fbank.ndim)
        elif self.dim == 3:
            assert fbank.ndim == 4, "dimensions has to be 4, but found {}".format(fbank.ndim)

        self.total_kernels = fbank.shape[0]
        self.num_kernels = int((1-self.kernel_drop_rate) * self.total_kernels)
        # print(f"nk: {self.num_kernels}")
        total = self.out_channels * (self.in_channels // self.groups)
        # select random indices "total" number of times
        indices = np.array(list(map(lambda x: np.random.choice(self.total_kernels, self.num_kernels, replace=False),
                                    np.zeros(total, dtype=np.uint8))))
        # Get the kernels from fbank that correspond to "indices"
        kernels = np.take(fbank, indices, axis=0)
        # reshape kernels to match the dimensions of a convolutional weights
        kernels = np.reshape(kernels,
                             (self.out_channels,
                              self.in_channels//self.groups,
                              self.num_kernels,
                              *self.kernel_size))

        kernels = torch.as_tensor(kernels)
        self.register_buffer("kernels", kernels)

    def get_filterbank(self, ):
        """ Load the filterbank using either the python script of the matlab files.
        The matlab 3D 3x3x3 Nikos filterbank performs better than the python 3x3x3 Nikos filterbank.
        Unfortunately, Nikos has lost the matlab code to reproduce them.
        """
        if self.dim == 2:
            return getattr(FilterBank2D, self.fbank_type)[FilterBank2D.shape2str(self.kernel_size)]
        elif self.dim == 3:
            # return getattr(FilterBank3D, self.fbank_type)[FilterBank3D.shape2str(self.kernel_size)]
            main_path = str(args.data_path)+"/uh2018_network_files"
            #main_path = "/media/kazem/ssd_1tb/PycharmProjects/nikos_filters/3d"
            # these are matlab filters so the last dim is z!!! Whereas, in pytorch the first dim is z!!!
            name = 'x'.join([f'{dim}' for dim in self.kernel_size])
            fbank = np.transpose(np.load(f"{main_path}/{name[::-1]}.npy"), (3, 0, 1, 2))
            return fbank


class Conv2dRF(_ConvNdRF):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', dim=2, kernel_drop_rate=0, fbank_type="frame"):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if 1 in kernel_size:
            raise ValueError("All kernel dimension values must be greater than 1.")

        super(Conv2dRF, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, dim, kernel_drop_rate, fbank_type)

    def _conv_forward(self, input, weight):

        kernel_span = self.get_kernel_span(weight, self.kernels)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            kernel_span, self.bias, self.stride,
                            _pair(0),
                            self.dilation, self.groups)
        return F.conv2d(input, kernel_span, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight)

    def get_kernel_span(self, weight, kernels):
        # because of pytorch's dynamic graph, meaning the computational graph gets erased after every iteration,
        # the linear combination operation (RF) has to be done in the forward method.
        # the .detach() ensures there is no memory leak caused by the kernels buffer.

        # the linear combination operation
        if self.kernel_drop_rate == 0:
            kernel_span = torch.einsum("ijk, klm -> ijlm", weight, kernels.detach())
        else:
            kernel_span = torch.einsum("ijk, ijklm -> ijlm", weight, kernels.detach())
        return kernel_span


class Conv3dRF(_ConvNdRF):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', dim=3, kernel_drop_rate=0, fbank_type="frame"):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        if 1 in kernel_size:
            raise ValueError("All kernel dimension values must be greater than 1.")

        super(Conv3dRF, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode, dim, kernel_drop_rate, fbank_type)

    def _conv_forward(self, input, weight):
        if self.kernel_drop_rate == 0:
            kernel_span = torch.einsum("ijk, klmn -> ijlmn", weight, self.kernels.detach())
        else:
            kernel_span = torch.einsum("ijk, ijklmn -> ijlmn", weight, self.kernels.detach())

        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            kernel_span, self.bias, self.stride,
                            _triple(0),
                            self.dilation, self.groups)
        return F.conv3d(input, kernel_span, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight)
