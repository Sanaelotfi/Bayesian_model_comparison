'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import random
import csv
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def save_output_from_dict(state, save_dir='/checkpoint/', save_name = 'table.csv'):
    out_path = os.path.join(save_dir,save_name)
    print(out_path)

    # Read input information
    args = []
    values = []
    for arg, value in state.items():
        args.append(arg)
        values.append(value)

    # Check for file
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    fieldnames = [arg for arg in args]

    # Read or write header
    try:
        with open(out_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except:
        with open(out_path, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()

    # Add row for this experiment
    with open(out_path, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow({arg: value for (arg, value) in zip(args, values)})
    print('\nResults saved to '+out_path)

def get_indices(trainset, num_per_class, seed=None):
    if num_per_class == -1:
        return [*range(len(trainset))]
    else:
        idx_dict = {} #keys are classes, values are lists of trainset indices
        indices = []
        for idx, (inputs, targets) in enumerate(trainset):
            if targets in idx_dict:
                idx_dict[targets].append(idx)
            else:
                idx_dict[targets] = [idx]
        num_classes = len(idx_dict)
        if seed is not None:
            random.seed(seed)
        for key in idx_dict:
            class_size = min(len(idx_dict[key]), num_per_class)
            indices.extend(random.sample(idx_dict[key], class_size))
        return indices

def conv_helper(width, depth):
    return [width if i == 0 else 2*width for i in range(depth)]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_json(stats, out_dir, log_name="test_stats.json"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    if os.path.isfile(fname):
        with open(fname, 'r') as fp:
            data_from_json = json.load(fp)
            num_entries = data_from_json['num entries']
        data_from_json[num_entries] = stats
        data_from_json["num entries"] += 1
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)
    else:
        data_from_json = {0: stats, "num entries": 1}
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)

def get_dataset(data_dir, datasrc, num_classes=None):
    '''num_classes only used in subCIFAR100 '''

    if datasrc == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)

    elif datasrc == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test)

    elif datasrc == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        trainset = torchvision.datasets.SVHN(
            root=data_dir, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=transform_test)

    elif datasrc == 'augCIFAR100':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                    brightness=1.0,
                    contrast=1.0,
                    saturation=1.0,
                    hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),
        ])
        class augDataset(Dataset):
            def __init__(self, origData,transform=None):
                self.transform = transform
                self.origData = origData
                self.n_samples = len(origData)
                self.xProto = {}
                for idx, (inputs, targets) in enumerate(self.origData):
                    if targets not in self.xProto: 
                        self.xProto[targets] = inputs
                                
            def __getitem__(self, index):
                y = self.origData[index][1]  
                x=self.xProto[y]
                if self.transform:
                    x = self.transform(x)
                return x,y
                
            def __len__(self):
                return self.n_samples

        orig_cifar100 = torchvision.datasets.CIFAR100(root=data_dir,train=True, download=True, transform=None)
        trainset = augDataset(origData=orig_cifar100,transform=transform)
        testset = augDataset(origData=orig_cifar100,transform=transform)

    elif datasrc == 'subCIFAR100':
        trainset, testset = get_dataset(data_dir, 'CIFAR100')

        class subDataset(Dataset):
            def __init__(self, origData, numClasses=10, transform=None):
                self.origData = origData
                origNumClasses = len(origData.classes)    # assumes origData.classes attrib
                self.x = []
                self.y = []
                random.seed(1)
                numClasses= min(numClasses, origNumClasses)
                subList = random.sample([*range(origNumClasses)],numClasses)
                for idx, (inputs, targets) in enumerate(self.origData):
                    if targets in subList: 
                        self.x.append(inputs)
                        self.y.append(subList.index(targets))    #reindex labels from 0
                self.n_samples = len(self.x)
                self.transform = transform
            def __getitem__(self, index):
                sample = self.x[index],self.y[index]
                if self.transform:
                    sample = self.transform(sample)  
                return sample
            def __len__(self):
                return self.n_samples
    
        trainset = subDataset(origData=trainset,numClasses=num_classes)
        testset = subDataset(origData=testset,numClasses=num_classes)

    return trainset, testset


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
    
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

