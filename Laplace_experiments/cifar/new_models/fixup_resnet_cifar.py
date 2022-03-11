import torch
import torch.nn as nn
import numpy as np
import asdfghjkl
from asdfghjkl.operations import Bias, Scale


__all__ = ['FixupResNet', 'fixup_resnet8', 'fixup_resnet14', 'fixup_resnet20', 'fixup_resnet26', 'fixup_resnet32', 'fixup_resnet44', 'fixup_resnet56', 'fixup_resnet110', 'fixup_resnet1202']


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(self.bias1a(x))
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(self.bias1a(x))
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, width=32):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = width
        self.conv1 = conv3x3(3, width)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, width*4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = Bias()
        self.fc = nn.Linear(width*4, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.bias2(x))

        return x
    
def fixup_resnet8(width, **kwargs):
    """Constructs a Fixup-ResNet-8 model.

    """
    model = FixupResNet(FixupBasicBlock, [1, 1, 1], width=width, **kwargs)
    return model

def fixup_resnet14(width, **kwargs):
    """Constructs a Fixup-ResNet-14 model.

    """
    model = FixupResNet(FixupBasicBlock, [2, 2, 2], width=width, **kwargs)
    return model

def fixup_resnet20(width, **kwargs):
    """Constructs a Fixup-ResNet-20 model.

    """
    model = FixupResNet(FixupBasicBlock, [3, 3, 3], width=width, **kwargs)
    return model

def fixup_resnet26(width, **kwargs):
    """Constructs a Fixup-ResNet-26 model.

    """
    model = FixupResNet(FixupBasicBlock, [4, 4, 4], width=width, **kwargs)
    return model


def fixup_resnet32(width, **kwargs):
    """Constructs a Fixup-ResNet-32 model.

    """
    model = FixupResNet(FixupBasicBlock, [5, 5, 5], width=width, **kwargs)
    return model


def fixup_resnet44(width, **kwargs):
    """Constructs a Fixup-ResNet-44 model.

    """
    model = FixupResNet(FixupBasicBlock, [7, 7, 7], width=width, **kwargs)
    return model


def fixup_resnet56(width, **kwargs):
    """Constructs a Fixup-ResNet-56 model.

    """
    model = FixupResNet(FixupBasicBlock, [9, 9, 9], width=width, **kwargs)
    return model


def fixup_resnet110(width,**kwargs):
    """Constructs a Fixup-ResNet-110 model.

    """
    model = FixupResNet(FixupBasicBlock, [18, 18, 18], width=width, **kwargs)
    return model


def fixup_resnet1202(width, **kwargs):
    """Constructs a Fixup-ResNet-1202 model.

    """
    model = FixupResNet(FixupBasicBlock, [200, 200, 200], width=width, **kwargs)
    return model


def rename_old_statedict(statedict):
    def fix_name(s):
        return f"{s}.weight" if "bias1" in s or "bias2" in s or "scale" in s else s

    return {fix_name(s): v for s, v in statedict.items()}