''' model combining SENet18 and DPN92 '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .senet import SENet18 
from .dpn import DPN92 

class SENet18_DPN92(nn.Module):
    def __init__(self, num_classes=10):
        super(SENet18_DPN92, self).__init__()
        self.sub1 = SENet18(num_classes)
        self.sub2 = DPN92(num_classes)
        sub1Feat = self.sub1.linear.in_features
        sub2Feat = self.sub2.linear.in_features
        self.sub1.linear = nn.Identity()
        self.sub2.linear = nn.Identity()
        
        self.linear = nn.Linear(sub1Feat+sub2Feat,num_classes)
        
    def forward(self, x):
        out1 = self.sub1(x)
        out2 = self.sub2(x)
        out = torch.cat((out1, out2), dim=1)
        return out

