import torch
import torch.nn as nn
import numpy as np


__all__ = ['ResNet', 'resnet8', 'resnet14', 'resnet20', 'resnet26', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, width=32):
        super(ResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = width
        self.conv1 = conv3x3(3, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, width*4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width*4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         for m in self.modules():
#             if isinstance(m, BasicBlock):
#                 nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
def resnet8(width, **kwargs):
    """Constructs a ResNet-8 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1], width=width, **kwargs)
    return model

def resnet14(width, **kwargs):
    """Constructs a ResNet-14 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2], width=width, **kwargs)
    return model

def resnet20(width, **kwargs):
    """Constructs a ResNet-20 model.
    """
    model = ResNet(BasicBlock, [3, 3, 3], width=width, **kwargs)
    return model

def resnet26(width, **kwargs):
    """Constructs a ResNet-26 model.
    """
    model = ResNet(BasicBlock, [4, 4, 4], width=width, **kwargs)
    return model


def resnet32(width, **kwargs):
    """Constructs a ResNet-32 model.
    """
    model = ResNet(BasicBlock, [5, 5, 5], width=width, **kwargs)
    return model


def resnet44(width, **kwargs):
    """Constructs a ResNet-44 model.
    """
    model = ResNet(BasicBlock, [7, 7, 7], width=width, **kwargs)
    return model


def resnet56(width, **kwargs):
    """Constructs a ResNet-56 model.
    """
    model = ResNet(BasicBlock, [9, 9, 9], width=width, **kwargs)
    return model


def resnet110(width,**kwargs):
    """Constructs a ResNet-110 model.
    """
    model = ResNet(BasicBlock, [18, 18, 18], width=width, **kwargs)
    return model


def resnet1202(width, **kwargs):
    """Constructs a ResNet-1202 model.
    """
    model = ResNet(BasicBlock, [200, 200, 200], width=width, **kwargs)
    return model