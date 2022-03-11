'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_multiplier=1.0, num_make_layers=4):
        super(ResNet, self).__init__()
        self.num_make_layers = num_make_layers
        self.in_planes = 64
        self.width_multiplier = width_multiplier
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        if num_make_layers == 1:
            self.layer1 = self._make_layer(block, int(64 * self.width_multiplier), num_blocks[0], stride=1)
            self.linear1 = nn.Linear(int(4096 * self.width_multiplier)*block.expansion, num_classes)

        if num_make_layers == 2:
            self.layer1 = self._make_layer(block, int(64 * self.width_multiplier), num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, int(128 * self.width_multiplier), num_blocks[1], stride=2)
            self.linear2 = nn.Linear(int(2048 * self.width_multiplier)*block.expansion, num_classes)

        if num_make_layers == 3:
            self.layer1 = self._make_layer(block, int(64 * self.width_multiplier), num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, int(128 * self.width_multiplier), num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, int(256 * self.width_multiplier), num_blocks[2], stride=2)
            self.linear3 = nn.Linear(int(1024 * self.width_multiplier)*block.expansion, num_classes)        

        if num_make_layers == 4:
            self.layer1 = self._make_layer(block, int(64 * self.width_multiplier), num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, int(128 * self.width_multiplier), num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, int(256 * self.width_multiplier), num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, int(512 * self.width_multiplier), num_blocks[3], stride=2)
            self.linear4 = nn.Linear(int(512 * self.width_multiplier)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.num_make_layers > 0:
            out = self.layer1(out)

        if self.num_make_layers > 1:
            out = self.layer2(out)

        if self.num_make_layers > 2:
            out = self.layer3(out)

        if self.num_make_layers > 3:
            out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if self.num_make_layers == 1:
            out = self.linear1(out)
        if self.num_make_layers == 2:
            out = self.linear2(out)
        if self.num_make_layers == 3:
            out = self.linear3(out)
        if self.num_make_layers == 4:
            out = self.linear4(out)
        return out

def ResNet6(num_classes,width_multiplier):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_multiplier=width_multiplier, num_make_layers=1)

def ResNet8(num_classes,width_multiplier):
    return ResNet(BasicBlock, [3, 2, 2, 2], num_classes=num_classes, width_multiplier=width_multiplier, num_make_layers=1)

def ResNet10(num_classes,width_multiplier):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_multiplier=width_multiplier, num_make_layers=2)

def ResNet12(num_classes,width_multiplier):
    return ResNet(BasicBlock, [2, 3, 2, 2], num_classes=num_classes, width_multiplier=width_multiplier, num_make_layers=2)

def ResNet14(num_classes,width_multiplier):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_multiplier=width_multiplier, num_make_layers=3)

def ResNet16(num_classes,width_multiplier):
    return ResNet(BasicBlock, [2, 2, 3, 2], num_classes=num_classes, width_multiplier=width_multiplier, num_make_layers=3)

def ResNet18(num_classes, width_multiplier):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_multiplier=width_multiplier)

#def ResNet18(num_classes):
#    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

#def ResNet34(num_classes):
#    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

#def ResNet50(num_classes):
#    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


#def ResNet101(num_classes):
#    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

#def ResNet152(num_classes):
#    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def ResNetFlex(num_classes, width_multiplier, depth):
    return ResNet(BasicBlock, [depth, depth, depth, depth], num_classes=num_classes, width_multiplier=width_multiplier)

def ResNetFlex34(num_classes, width_multiplier):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, width_multiplier=width_multiplier)

def ResNetFlex50(num_classes, width_multiplier):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, width_multiplier=width_multiplier)

def ResNetFlex101(num_classes, width_multiplier):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, width_multiplier=width_multiplier)

def ResNetFlex152(num_classes, width_multiplier):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, width_multiplier=width_multiplier)

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
