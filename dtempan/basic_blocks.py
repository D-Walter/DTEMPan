import torch.nn as nn
from .netutils import ModuleWithInit


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class CNABlock(ModuleWithInit):
    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, dilation=1, bias=True, activate=nn.ReLU):
        super(CNABlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = activate()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ResBlock(ModuleWithInit):
    def __init__(self, in_channels, out_channels, mid_channels=None, activate=nn.Identity):
        super(ResBlock, self).__init__()
        if mid_channels == None:
            mid_channels = out_channels
        self.conv1 = CNABlock(in_channels, mid_channels)
        self.conv2 = CNABlock(mid_channels, out_channels, activate=activate)
        if in_channels != out_channels:
            self.ch_align = CNABlock(in_channels, out_channels, kernal_size=1, padding=0)
        else:
            self.ch_align = nn.Identity()

    def forward(self, x):
        x1 = self.ch_align(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x1
        return out


class LightResBlock(ModuleWithInit):
    def __init__(self, in_channels, out_channels, activate=nn.Identity):
        super(LightResBlock, self).__init__()
        self.conv1 = CNABlock(in_channels, out_channels, kernal_size=1, padding=0, activate=activate)
        if in_channels != out_channels:
            self.ch_align = CNABlock(in_channels, out_channels, kernal_size=1, padding=0)
        else:
            self.ch_align = nn.Identity()

    def forward(self, x):
        x1 = self.ch_align(x)
        out = self.conv1(x)
        out = out + x1
        return out
