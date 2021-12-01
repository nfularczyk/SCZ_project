import torch
import torch.nn as nn
import convrf as convrf

nonlineartities = \
    [
        "Tanh",
        "Tanhshrink",
        "Hardtanh",
        "Sigmoid",
        "Hardsigmoid",
        "LogSigmoid",
        "Hardshrink",
        "Softshrink",
        "Hardswish",
        "Softsign",
        "Softplus",
        "ReLU",
        "PReLU",
        "LeakyReLU",
        "ReLU6",
        "GELU",
        "ELU",
        "SELU",
        "CELU",
        # "MultiheadAttention",
    ]


def weight_reg(model, lambda_, p=2, arch='CNN'):
    reg_term = 0
    arch = arch.lower()
    assert arch == 'cnn' or arch == 'mlp', \
        f"{arch} is either not implemented or does not need weight regularization. " \
        f"It must be one of the following: 'cnn' or 'mlp'"

    if arch == 'cnn':
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                reg_term += lambda_ * torch.mean(m.weight ** p)
    elif arch == 'mlp':
        for m in model.modules():
            assert isinstance(m, nn.Linear)
            reg_term += lambda_ * torch.mean(m.weight ** p)

    return reg_term


class BasicConv(nn.Module):
    """The idea is to have one basic convolutional layer/block for different variations of convolution, nonlinearity,
    and batch normalization layers in a multi-layered CNN architecture.

    convolution layer choices:
    Conv1d, Conv2d, Conv2dRF, Conv3d, and Conv3dRF

    nonlinearity choices include:
    all 'nonlineartity' entries in the python list defined above

    batch normalization layer choices include:
    None, BatchNorm1d, BatchNorm2d, BatchNorm3d, InstanceNorm1d, InstanceNorm2d, and InstanceNorm3d

    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, **kwargs):
        super(BasicConv, self).__init__()

        self.conv_model, self.bn, eps, self.nl = kwargs['conv_model'], kwargs['bn'], kwargs['eps'], kwargs['nl']
        self.rf_config = kwargs['rf_config']
        del kwargs['conv_model'], kwargs['rf_config'], kwargs['bn'], kwargs['eps'], kwargs['nl']

        assert self.conv_model in {"Conv1d", "Conv2d", "Conv3d", "Conv2dRF", "Conv3dRF"},  \
            print(f"{self.conv_model} is neither an attribute of torch.nn nor an attribute of convrf.convrf."
                  f"It has to be one of the following: Conv1d, Conv2dRF, Conv2d, Conv2dRF, Conv3d, Conv3dRF")

        if hasattr(nn, self.conv_model):  # conventional CNN
            for ext_arg in ['rf_config', 'kernel_drop_rate', 'fbank_type']:
                if ext_arg in kwargs:
                    del kwargs[ext_arg]
            self.conv = getattr(nn, self.conv_model)(in_channels, out_channels, kernel_size, bias=bias, **kwargs)

        elif hasattr(convrf, self.conv_model):  # receptive field (RF) CNN

            if self.rf_config == 0:
                self.conv = getattr(convrf, self.conv_model)(in_channels,
                                                             out_channels,
                                                             kernel_size,
                                                             bias=bias,
                                                             **kwargs)
            elif self.rf_config == 2:  # half and half
                assert out_channels % 2 == 0
                self.conv1 = getattr(convrf, self.conv_model)(
                    in_channels, out_channels // 2, kernel_size, bias=bias, **kwargs)
                del kwargs['kernel_drop_rate'], kwargs['fbank_type']
                cm = f"Conv{filter(str.isdigit, self.conv_model)}d"
                self.conv2 = getattr(nn, cm)(in_channels,
                                             out_channels // 2,
                                             kernel_size,
                                             bias=bias,
                                             **kwargs)

        if self.bn is not None:
            self.bn = getattr(nn, self.bn)(out_channels, eps=eps)
        if self.nl is not None:
            self.nl = getattr(nn, self.nl)()

    def forward(self, x):
        if "RF" not in self.conv_model or self.rf_config == 0:
            x = self.conv(x)
        elif "RF" in self.conv_model and self.rf_config == 2:
            x = torch.cat((self.conv1(x), self.conv2(x)), dim=1)
        if self.bn is not None:
            x = self.bn(x)
        if self.nl is not None:
            x = self.nl(x)
        return x


class SpectralBlock(nn.Module):
    growth_rate = 12

    def __init__(self, in_channels=1,
                 bn=None, nl="PReLU", eps=1e-5,
                 conv_model="Conv3d",):
        super(SpectralBlock, self).__init__()
        kernel_size, padding = (7, 1, 1), (3, 0, 0)
        kwargs = {
            'kernel_size': kernel_size,
            'padding': padding,
            'conv_model': conv_model,
            'bn': bn,
            'nl': nl,
            'eps': eps,
            'rf_config': None,
            'fbank_type': None,
            'kernel_drop_rate': None
        }
        self.c1 = BasicConv(in_channels, 24, stride=(2, 1, 1), **kwargs)
        self.c2 = BasicConv(24, 12, **kwargs)
        self.c3 = BasicConv(36, 12, **kwargs)
        self.c4 = BasicConv(48, 12, **kwargs)

    def forward(self, x):
        x1_0 = self.c1(x)
        x1_1 = self.c2(x1_0)
        x1_1_ = torch.cat((x1_0, x1_1), 1)
        x1_2 = self.c3(x1_1_)
        x1_2_ = torch.cat((x1_0, x1_1, x1_2), 1)
        x1_3 = self.c4(x1_2_)
        out = torch.cat((x1_0, x1_1, x1_2, x1_3), 1)
        return out


class SpatialBlock(nn.Module):
    growth_rate = 12

    def __init__(self,
                 in_channels=1,
                 kernel_size=3,
                 bn=None,
                 nl=None,
                 eps=1e-5,
                 conv_model="Conv2d",
                 rf_config=0,
                 fbank_type="frame",
                 kernel_drop_rate=0,
                 ):
        super(SpatialBlock, self).__init__()
        padding = (kernel_size-1) // 2
        kwargs = {
            'kernel_size': kernel_size,
            'padding': padding,
            'bn': bn,
            'nl': nl,
            'eps': eps,
            'conv_model': conv_model,
            'rf_config': rf_config,
            'fbank_type': fbank_type,
            'kernel_drop_rate': kernel_drop_rate
        }
        self.c1 = BasicConv(in_channels, 24, **kwargs)
        self.c2 = BasicConv(24, 12, **kwargs)
        self.c3 = BasicConv(36, 12, **kwargs)
        self.c4 = BasicConv(48, 12, **kwargs)

    def forward(self, x):
        x1_0 = self.c1(x)
        x1_1 = self.c2(x1_0)
        x1_1_ = torch.cat((x1_0, x1_1), 1)
        x1_2 = self.c3(x1_1_)
        x1_2_ = torch.cat((x1_0, x1_1, x1_2), 1)
        x1_3 = self.c4(x1_2_)
        out = torch.cat((x1_0, x1_1, x1_2, x1_3), 1)
        return out
