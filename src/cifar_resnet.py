"""CIFAR ResNet-20 (6n+2, n=3), option-A shortcuts, 16/32/64 channels.

Architecture follows He et al. (2016), section 4.2. No ImageNet stem or
pretrained weights; batch normalization is included in every residual branch.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.pad = (out_channels - in_channels) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x if self.stride == 1 else F.pad(
            x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + residual)


class CifarResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU())
        blocks = []
        channels = 16
        for width in (16, 32, 64):
            for i in range(3):
                stride = 2 if i == 0 and width != 16 else 1
                blocks.append(Block(channels, width, stride))
                channels = width
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.blocks(self.stem(x)).mean(dim=(2, 3))
        return self.fc(x)
