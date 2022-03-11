import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FullyConnectedBlock(nn.Module):
    def __init__(self, width, BN=False):
        super(FullyConnectedBlock, self).__init__()
        self.BN = BN
        self.linear = nn.Linear(width, width, bias = not self.BN)
        #self.linear = nn.Linear(width, width)
        if self.BN: self.bn = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.linear(x)
        if self.BN: out=self.bn(x)
        return self.relu(out)

class ConvBlock(nn.Module):
    def __init__(self, input_filters, output_filters, BN=False):
        super(ConvBlock, self).__init__()
        self.BN = BN
        self.conv = nn.Conv2d(input_filters, output_filters, kernel_size=3, stride=1, padding=1, bias=not self.BN)
        if self.BN: self.bn = nn.BatchNorm2d(output_filters)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        if self.BN: out = self.bn(out)
        return self.relu(out)

class Fully_Connected_Net(nn.Module):
    def __init__(self, block=FullyConnectedBlock, channels=3, image_dim = 32, num_classes=10, width=1000, depth=2, BN=False):
        super(Fully_Connected_Net, self).__init__()
        self.block = block
        self.BN = BN
        self.linear_first = nn.Linear(image_dim*image_dim*channels, width, bias=not self.BN)
        if BN: self.bn_first = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.layers = self._make_layer(block, width, depth-2, self.BN)
        self.linear_last = nn.Linear(width, num_classes)
    def _make_layer(self, block, width, depth, BN):
        layers = []
        for i in range(depth):
            layers.append(block(width, BN=BN))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear_first(out)
        if self.BN: out = self.bn_first(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.linear_last(out)
        return out

class CNN_Net(nn.Module):
    def __init__(self, block=ConvBlock, channels=3, image_dim = 32, num_classes=10, num_filters=[64,128], BN=False):
        super(CNN_Net, self).__init__()
        self.block = block
        self.BN = BN
        self.conv_first = nn.Conv2d(channels, num_filters[0], kernel_size=3, stride=1, padding=1, bias=not self.BN)
        if self.BN: self.bn_first = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU()
        self.layers = self._make_layer(block, num_filters, self.BN)
        self.linear = nn.Linear(num_filters[-1]*image_dim*image_dim, num_classes)
    def _make_layer(self, block, num_filters, BN):
        layers = []
        for i in range(len(num_filters)-1):
            layers.append(block(num_filters[i], num_filters[i+1], BN=BN))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv_first(x)
        if self.BN: out = self.bn_first(out)
        out = self.relu(out)
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def convnet(width=32, in_channels=3, num_classes=10, depth=0, **kwargs):
    """Define a simple ConvNet. This architecture only really works for CIFAR10."""

    module_list = []
    module_list.append(('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)))
    module_list.append(('relu0', torch.nn.ReLU()))
    module_list.append(('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)))
    module_list.append(('relu1', torch.nn.ReLU()))
    module_list.append(('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)))
    module_list.append(('relu2', torch.nn.ReLU()))
    for i in range(depth):
        module_list.append((f'conv_ext{i}', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)))
        module_list.append((f'relu_ext{i}', torch.nn.ReLU()))
    module_list.append(('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)))
    module_list.append(('relu3', torch.nn.ReLU()))
    module_list.append(('pool3', torch.nn.MaxPool2d(3)))
    module_list.append(('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)))
    module_list.append(('relu4', torch.nn.ReLU()))
    module_list.append(('pool4', torch.nn.MaxPool2d(3)))
    module_list.append(('flatten', torch.nn.Flatten()))
    module_list.append(('linear', torch.nn.Linear(36 * width, num_classes)))


    model = torch.nn.Sequential(OrderedDict(module_list))
    '''
    model = torch.nn.Sequential(OrderedDict([
        ('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)),
        ('relu0', torch.nn.ReLU()),
        ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu1', torch.nn.ReLU()),
        ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu2', torch.nn.ReLU()),
        ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu3', torch.nn.ReLU()),
        ('pool3', torch.nn.MaxPool2d(3)),
        ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu4', torch.nn.ReLU()),
        ('pool4', torch.nn.MaxPool2d(3)),
        ('flatten', torch.nn.Flatten()),
        ('linear', torch.nn.Linear(36 * width, num_classes))
    ]))
    '''
    return model

def MLP(channels = 3, image_dim = 32, num_classes=10, width=1000, depth=2, BN=False):
    return Fully_Connected_Net(channels=channels, image_dim=image_dim, num_classes=num_classes, width=width, depth=depth, BN=BN)

def CNN(channels=3, image_dim = 32, num_classes=10, num_filters=[64, 128], BN=False):
    return CNN_Net(channels=channels, image_dim = image_dim, num_classes=num_classes, num_filters=num_filters, BN=BN)
