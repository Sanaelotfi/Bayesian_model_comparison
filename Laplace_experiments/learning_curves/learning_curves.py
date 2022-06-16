'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import SubsetRandomSampler,Dataset
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import parameters_to_vector, vector_to_parameters


import torchvision
import torchvision.transforms as transforms
import scipy.special


import argparse
import csv
from laplace import Laplace
from laplace.curvature import AsdlEF, BackPackGGN, AsdlGGN
import sys
import tqdm

from models import *
from utils import get_dataset, progress_bar, get_indices


import time

import numpy as np
import random
import math

la_method =  AsdlGGN


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--width_multiplier', type=float, help='width multiplier for neural network')
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--sample_batch_size', default=16, type=int, help='batch size')
parser.add_argument('--seed', default=10, type=int, help='seed')
parser.add_argument('--prior_prec', default=5e-4, type=float, help='weight decay')
parser.add_argument('--data_ratio', default=0.8, type=float, help='ratio of the data to use for CMLL')
parser.add_argument('--use_bma', default=True, type=bool, help='whether or not to use bma to get CMLL')
parser.add_argument('--bma_nsamples', default=20, type=int, help='whether or not to use bma to get CMLL')
parser.add_argument('--num_runs', default=5, type=int, help='whether or not to get the CMLL')
parser.add_argument('--hessian_structure', default='diag', type=str, help='structure of the hessian')
parser.add_argument('--chk_path', type=str, help='path for the checkpoint for this model')
parser.add_argument('--result_folder', default='./results/', type=str, help='path of the results')

args = parser.parse_args()


seed = args.seed

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()


data_directory = "/datasets"
dataset = "CIFAR10"
num_classes = 10

batchsize = args.batch_size

# Data
print('==> Preparing data..')

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


trainset1 = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=transform_train)
trainset2 = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=transform_train)
trainset3 = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=transform_train)

# getting subsets of the data 
subsets_data = np.load('./data/cifar10_subsets.npz')
trainset1.data = subsets_data['x1']
trainset1.targets = subsets_data['y1'].tolist()
trainset2.data = subsets_data['x2']
trainset2.targets = subsets_data['y2'].tolist()
trainset3.data = subsets_data['x3']
trainset3.targets = subsets_data['y3'].tolist()

trainloader_train = torch.utils.data.DataLoader(trainset1, batch_size=batchsize, shuffle=True, num_workers=2)
trainloader_valid = torch.utils.data.DataLoader(trainset2, batch_size=args.sample_batch_size, shuffle=False, num_workers=2)
trainloader_test = torch.utils.data.DataLoader(trainset3, batch_size=args.sample_batch_size, shuffle=False, num_workers=2)

print("this is the size of the new train loader: {}, valid train {}, test train {}".format(len(trainloader_train.dataset),
                                                                                           len(trainloader_valid.dataset),
                                                                                           len(trainloader_test.dataset)))

testset = torchvision.datasets.CIFAR10(root=data_directory, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)


def get_acc(net, loader):
    device = parameters_to_vector(net.parameters()).device
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    net.train()
    return 100.*correct/total



def get_bma_acc(net, la, loader, n_samples, hessian_structure, temp=1.0):
    device = parameters_to_vector(net.parameters()).device
    samples = torch.randn(n_samples, la.n_params, device=device)
    if hessian_structure == "kron":
        samples = la.posterior_precision.bmm(samples, exponent=-0.5)
        params = la.mean.reshape(1, la.n_params) + samples.reshape(n_samples, la.n_params) * temp
    elif hessian_structure == "diag":
        samples = samples * la.posterior_scale.reshape(1, la.n_params) * temp
        params = la.mean.reshape(1, la.n_params) + samples
    else:
        raise
    all_probs = []
    for sample_params in params:
        sample_probs = []
        all_ys = []
        with torch.no_grad():
            vector_to_parameters(sample_params, net.parameters())
            net.eval()
            for x, y in loader:
                logits = net(x.cuda()).detach().cpu()
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sample_probs.append(probs.detach().cpu().numpy())
                all_ys.append(y.detach().cpu().numpy())
            sample_probs = np.concatenate(sample_probs, axis=0)
            all_ys = np.concatenate(all_ys, axis=0)
            all_probs.append(sample_probs)

    all_probs = np.stack(all_probs)
    bma_probs = np.mean(all_probs, 0)
    bma_accuracy = (np.argmax(bma_probs, axis=-1) == all_ys).mean() * 100

    return bma_accuracy, bma_probs, all_ys


def get_cmll_new(net, la, loader, n_samples,
    hessian_structure, temp=1.0, eps=1e-4):
    device = parameters_to_vector(net.parameters()).device
    samples = torch.randn(n_samples, la.n_params, device=device)
    if hessian_structure == "kron":
        samples = la.posterior_precision.bmm(samples, exponent=-0.5)
        params = la.mean.reshape(1, la.n_params) + samples.reshape(n_samples, la.n_params) * temp
    elif hessian_structure == "diag":
        samples = samples * la.posterior_scale.reshape(1, la.n_params) * temp
        params = la.mean.reshape(1, la.n_params) + samples
    else:
        raise
    all_probs = []
    for sample_params in params:
        sample_probs = []
        all_ys = []
        with torch.no_grad():
            vector_to_parameters(sample_params, net.parameters())
            net.eval()
            for x, y in loader:
                logits = net(x.cuda()).detach().cpu()
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sample_probs.append(probs.detach().cpu().numpy())
                all_ys.append(y.detach().cpu().numpy())
            sample_probs = np.concatenate(sample_probs, axis=0)
            all_ys = np.concatenate(all_ys, axis=0)
            all_probs.append(sample_probs)
    all_probs = np.stack(all_probs)
    bma_probs = np.mean(all_probs, 0)
    bma_accuracy = (np.argmax(bma_probs, axis=-1) == all_ys).mean() * 100
    probs_arrays = []  
    for i, label in enumerate(all_ys):
        probs_i = all_probs[:,i,:]
        lik_array = probs_i[:,label]
        probs_arrays.append(lik_array)
    probs_arrays = np.stack(probs_arrays)
    log_probs_arrays = np.sum(np.log(probs_arrays), axis=0)
    cmll = scipy.special.logsumexp(log_probs_arrays) - np.log(len(log_probs_arrays))
    
    return cmll, bma_accuracy



def get_mll_acc(model,
                width_multiplier,
                chk_path,
                prior_prec,
                data_ratio,
                hessian_structure, 
                batch_size,
                sample_batch_size,
                num_runs,
                use_bma,
                bma_nsamples,
                result_folder):
    
    path = "cifar10_" + model + "_" + str(width_multiplier) + "_priorprec_" + str(prior_prec) + "_hessianstr_" + hessian_structure

    path += "_dataratio_" + str(data_ratio) + '_cmllbma'
        
    print("this is the current path ... ", path)

    
    # log for keeping accuracy 
    logname = result_folder + path + '.csv'
    
        
    # Model
    print("=> creating model '{}_{}'".format(model, width_multiplier))
    if model == 'VGG19':
        net = VGG('VGG19', num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(num_classes=num_classes, width_multiplier=width_multiplier)
    elif model == "GoogLeNet":
        net = GoogLeNet(num_classes=num_classes)
    else:
        raise
    num_params = sum(p.numel() for p in net.parameters())
    print("num of params: ", num_params)
    
    net.load_state_dict(torch.load(chk_path))
    
    if use_cuda:
        net.cuda()
        print('Using CUDA..')
        
        
    # making results directory        
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['num_params', 'real test acc', 'real bma acc', 'bma test (s3) acc', 'decay', 'best temperature', 'cmll', 'test LL'])

    test_acc = get_acc(net, testloader)
    

    print("fitting LA for the trainset 1")
    la = Laplace(net, 'classification', subset_of_weights='all',
             hessian_structure=hessian_structure, prior_precision=prior_prec,
             backend=la_method)
    la.fit(trainloader_train)
    print("done fitting LA for the small data")


    epoch = 0
    temp = 1e-4
    best_accuracy = 0
    best_temp = temp
    buffer = 0
    max_buffer = 3
    max_epochs = 50
    while epoch < max_epochs: 
        cmll, bma_accuracy = get_cmll_new(net, la, trainloader_valid, bma_nsamples, hessian_structure, temp=temp)
        print("current temperate: {}, bma accuracy: {}, cmll: {}".format(temp, bma_accuracy, cmll))
        if bma_accuracy > best_accuracy:
            best_accuracy = bma_accuracy
            best_temp = temp
            temp /= 10.
            buffer = 0
        elif buffer < max_buffer:
            buffer += 1
            temp /= 5
        else:
            break
        epoch += 1

    print("best temperate: {}, bma accuracy: {}".format(best_temp, best_accuracy))

    cmll, bma_accuracy = get_cmll_new(net, la, trainloader_test, bma_nsamples, hessian_structure, temp=best_temp)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([num_params, test_acc, None, bma_accuracy, prior_prec, best_temp, cmll, None])   


models  = ["GoogLeNet", "VGG19", "ResNet18", "ResNet18"]
width_multipliers = [1.0, 1.0, 1.0, 2.0]

checkpoints = ["GoogLeNet_1.0_0.8.pt",
               "VGG19_1.0_0.8.pt",
               "ResNet18_1.0_0.8.pt",
               "ResNet18_2.0_0.8.pt"
]

for i, model in enumerate(models):
    if i > 2:
        width_multiplier = width_multipliers[i]
        chk_path = "./checkpoints/" + checkpoints[i]
        get_mll_acc(model = model,
                    width_multiplier =width_multiplier,
                    chk_path =chk_path,
                    prior_prec = args.prior_prec,
                    data_ratio = args.data_ratio,
                    hessian_structure = args.hessian_structure, 
                    batch_size = args.batch_size,
                    num_runs = args.num_runs,
                    use_bma = args.use_bma,
                    bma_nsamples = args.bma_nsamples,
                    sample_batch_size = args.sample_batch_size,
                    result_folder = args.result_folder)

    
    