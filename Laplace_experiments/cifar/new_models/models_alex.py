import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from asdfghjkl.operations import Scale, Bias


class AdaCNN(nn.Sequential):

    def __init__(self, in_pixels=28, in_channels=1, n_out=10, width=64, depth=3):
        super().__init__()
        assert 1 <= depth <= 5
        self.output_size = n_out
        # layer 1
        self.add_module('Conv1', nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('bnscale', Scale())
        self.add_module('bnbias', Bias())
        self.add_module('ReLU1', nn.ReLU())

        cur_width = width
        pixels = in_pixels
        for i in range(depth - 1):  # first layer in already
            self.add_module(f'Conv{i+2}', nn.Conv2d(cur_width, cur_width*2, kernel_size=3, stride=1, padding=1, bias=True))
            self.add_module(f'bnscale{i+2}', Scale())
            self.add_module(f'bnbias{i+2}', Bias())
            self.add_module(f'ReLU{i+2}', nn.ReLU())
            self.add_module(f'Pool{i+2}', nn.MaxPool2d(2))
            cur_width = cur_width * 2
            pixels = int(math.floor(pixels/2))

        desired_width = width * 16
        ds_widtha = ds_widthb = math.ceil(math.sqrt(desired_width / cur_width))
        if depth == 4:  # special case sqrt(2) = 1.41 not easy to resolve
            ds_widtha = 2
            ds_widthb = 1

        self.add_module('Resample', nn.AdaptiveAvgPool2d((ds_widtha, ds_widthb)))
        self.add_module('Flatten', nn.Flatten())
        self.add_module('Classifier', nn.Linear(cur_width * ds_widtha * ds_widthb, n_out, bias=True))
        
        
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = Bias() # nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = Bias()  #nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()  #nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = Scale() # nn.Parameter(torch.ones(1))
        self.bias2b = Bias()  #nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, in_planes=16, in_channels=3):
        super(FixupResNet, self).__init__()
        self.output_size = num_classes
        self.num_layers = sum(layers)
        self.inplanes = in_planes
        self.conv1 = conv3x3(in_channels, in_planes)
        self.bias1 = Bias()  # nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, layers[0])
        self.layer2 = self._make_layer(block, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = Bias()  # nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(in_planes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, 
                                mean=0, 
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
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

    
def fixup_resnet8(**kwargs):
    """Constructs a Fixup-ResNet-20 model.
    """
    model = FixupResNet(FixupBasicBlock, [1, 1, 1], **kwargs)
    return model

    
def fixup_resnet14(**kwargs):
    """Constructs a Fixup-ResNet-20 model.
    """
    model = FixupResNet(FixupBasicBlock, [2, 2, 2], **kwargs)
    return model


def fixup_resnet20(**kwargs):
    """Constructs a Fixup-ResNet-20 model.
    """
    model = FixupResNet(FixupBasicBlock, [3, 3, 3], **kwargs)
    return model


def fixup_resnet26(**kwargs):
    """Constructs a Fixup-ResNet-32 model.
    """
    model = FixupResNet(FixupBasicBlock, [4, 4, 4], **kwargs)
    return model


def fixup_resnet32(**kwargs):
    """Constructs a Fixup-ResNet-32 model.
    """
    model = FixupResNet(FixupBasicBlock, [5, 5, 5], **kwargs)
    return model


def fixup_resnet44(**kwargs):
    """Constructs a Fixup-ResNet-44 model.
    """
    model = FixupResNet(FixupBasicBlock, [7, 7, 7], **kwargs)
    return model


def fixup_resnet56(**kwargs):
    """Constructs a Fixup-ResNet-56 model.
    """
    model = FixupResNet(FixupBasicBlock, [9, 9, 9], **kwargs)
    return model


def fixup_resnet110(**kwargs):
    """Constructs a Fixup-ResNet-110 model.
    """
    model = FixupResNet(FixupBasicBlock, [18, 18, 18], **kwargs)
    return model


if __name__ == '__main__':
    cnn_model = AdaCNN(in_pixels=32, in_channels=3, n_out=10, width=4, depth=3)
    fixup_resnet = fixup_resnet20()
