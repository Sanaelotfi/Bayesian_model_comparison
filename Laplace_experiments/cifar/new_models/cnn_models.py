import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from asdfghjkl.operations import Scale, Bias

__all__ = ['FixupCNN']


class FixupCNN(nn.Sequential):

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