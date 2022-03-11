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

import argparse
import csv
from laplace import Laplace
from laplace.curvature import AsdlEF
import sys
import tqdm

import new_models
from utils import *
# from utils import get_dataset, progress_bar, get_indices


import time

import numpy as np
import pandas as pd
import random
import math


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--sample_batch_size', default=16, type=int, help='batch size used when computing the bma accuracy')
parser.add_argument('--seed', default=10, type=int, help='seed')
parser.add_argument('--decay', type=float, help='weight decay')
parser.add_argument('--base_lr', default=0.01, type=float, help='base learning rate')
parser.add_argument('--data_ratio', default=0.8, type=float, help='ratio of the data to use for CMLL')
parser.add_argument('--bma_nsamples', default=20, type=int, help='whether or not to use bma to get CMLL')
parser.add_argument('--max_iters', default=50, type=int, help='max epochs for tuning the temperature')
parser.add_argument('--hessian_structure', default='kron', type=str, help='structure of the hessian')
parser.add_argument('--prior_structure', default='scalar', type=str, help='structure of the prior: scalar or layerwise')
parser.add_argument('--partialtrain_chk_path', default="checkpoint/cifar10/subset/resnets" , type=str, help='path for the checkpoint for the model trained on a fraction of the data')
parser.add_argument('--fulltrain_chk_path', default="checkpoint/cifar10/optimized_la" , type=str, help='path for the checkpoint for the model trained on the full data')
parser.add_argument('--result_folder', default='./results/cifar10/subset/resnets/cmll/', type=str, help='path of the results')

args = parser.parse_args()

seed = args.seed

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()


data_directory = "/datasets"
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

trainset = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=transform_train)
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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
trainloader_train = torch.utils.data.DataLoader(trainset1, batch_size=batchsize, shuffle=True, num_workers=2)
trainloader_valid = torch.utils.data.DataLoader(trainset2, batch_size=args.sample_batch_size, shuffle=False, num_workers=2)
trainloader_test = torch.utils.data.DataLoader(trainset3, batch_size=args.sample_batch_size, shuffle=False, num_workers=2)

print("this is the size of the full train loader {}, new train loader: {}, valid train {}, test train {}".format(len(trainloader.dataset), len(trainloader_train.dataset), len(trainloader_valid.dataset), len(trainloader_test.dataset)))

testset = torchvision.datasets.CIFAR10(root=data_directory, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

def get_acc(net, loader):
    device = parameters_to_vector(net.parameters()).device
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_ys = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            logits = outputs.detach().cpu()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            all_probs.append(probs)
            all_ys.append(targets.detach().cpu())
    all_ys = np.concatenate(all_ys, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    net.train()
    return 100.*correct/total, all_probs, all_ys


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

def get_cmll(bma_probs, all_ys, eps=1e-4):
    log_lik = 0      
    eps = 1e-4
    for i, label in enumerate(all_ys):
        probs_i = bma_probs[i]
        probs_i += eps
        probs_i[np.argmax(probs_i)] -= eps * len(probs_i)
        log_lik += np.log(probs_i[label]).item()
    cmll = log_lik/len(all_ys)
    
    return cmll


def get_mll_acc(arch,
                width,
                partialtrain_chk_path,
                fulltrain_chk_path,
                prior_prec_init,
                data_ratio,
                hessian_structure, 
                prior_structure,
                base_lr,
                bma_nsamples,
                max_iters,
                result_folder):
        
    path = "cifar10_" + str(prior_prec_init) + '_' + arch + '_' + str(width) + '_' + prior_structure 
        
    print("this is the current path ... ", path)
    
    fullpath = "cifar10_" + str(prior_prec_init) + '_' + arch + '_' + str(width) + '_' + prior_structure 
    cknew_path1 = fullpath + '_final'
    cknew_path1 = './' + fulltrain_chk_path + '/' + cknew_path1 + '.ckpt'
    fullpath2 = fullpath + '_baselr_' + str(base_lr)
    cknew_path2 = fullpath2 + '_final'
    cknew_path2 = './' + fulltrain_chk_path + '/' + cknew_path2 + '.ckpt'
    
    if not os.path.exists(cknew_path1) and not os.path.exists(cknew_path2):
        print("the corresponding fully trained model does not exist ... ") 
        return
    
    # log for keeping accuracy 
    logname = result_folder + path + '_cmllbma.csv'
        
    # making results directory
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    if os.path.exists(logname):
        if len(pd.read_csv(logname)) > 0:
            print("++++++++++++++ this file already exists +++++++++++++")
            return 
    
    # Model
    print("=> creating model 'fixup_cnn{}_{}'".format(arch, width))
    net = new_models.__dict__[arch](width, num_classes=10)
    num_params = sum(p.numel() for p in net.parameters())
    print("num of params: ", num_params)
    
    
    cknew_path = path + '_final'
    cknew_path = './' + partialtrain_chk_path + '/' + cknew_path + '.ckpt'
    
    if not os.path.exists(cknew_path):
        print(" ++++++++++++++ smaller model still not fully trained yet .... ++++++++++++++")
        return
    else:
        print("loading the state dictionary ...") 
        net.load_state_dict(torch.load(cknew_path)["net_state_dict"])
    
    
    if use_cuda:
        net.cuda()
        print('Using CUDA..')
        
    # log for keeping accuracy 
    logname = result_folder + path + '_cmllbma.csv'
    
    if os.path.exists(logname):
        if len(pd.read_csv(logname)) > 0:
            print("++++++++++++++ this file already exists +++++++++++++")
            return 
        
    # making results directory
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['num_params', 'decay', 'best temperature', 'test acc', 'bma test acc', 'bma s3 acc', 'test LL', 'bma test LL', 'cmll'])
    
     
    print("fitting LA for the trainset 1")
    la = Laplace(net, 'classification', subset_of_weights='all',
             hessian_structure=hessian_structure, prior_precision=prior_prec_init,
             backend=AsdlEF)
    la.fit(trainloader_train)
    print("done fitting LA for the small data")

    
    epoch = 0
    temp = 1e-3
    best_accuracy = 0
    best_temp = temp
    buffer = 0
    max_buffer = 2
    while epoch < max_iters: 
        bma_accuracy, bma_probs, all_ys = get_bma_acc(net, la, trainloader_valid, bma_nsamples, hessian_structure, temp=temp)
        cmll = get_cmll(bma_probs, all_ys, eps=1e-4)
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
    
    bma_accuracy, bma_probs, all_ys = get_bma_acc(net, la, trainloader_test, bma_nsamples, hessian_structure, temp=best_temp)
    cmll = get_cmll(bma_probs, all_ys, eps=1e-4)
    
    # define the original net 
    print("=> creating model 'fixup_cnn{}_{}'".format(arch, width))
    net = net = new_models.__dict__[arch](width, num_classes=10)
    num_params = sum(p.numel() for p in net.parameters())
    
    
    if os.path.exists(cknew_path1):
        print("loading the state dictionary ...") 
        net.load_state_dict(torch.load(cknew_path1)["net_state_dict"])
    elif os.path.exists(cknew_path2):
        print("loading the state dictionary ...") 
        net.load_state_dict(torch.load(cknew_path2)["net_state_dict"])

        
    if use_cuda:
        net.cuda()
        print('Using CUDA..')
        
    print("fitting LA for the full trainset")
    la = Laplace(net, 'classification', subset_of_weights='all',
             hessian_structure=hessian_structure, prior_precision=prior_prec_init,
             backend=AsdlEF)
    la.fit(trainloader)
    print("done fitting LA for the full data")
        
        
    bma_test_accuracy, bma_test_probs, all_test_ys = get_bma_acc(net, la, testloader, bma_nsamples, hessian_structure, temp=best_temp)
    bma_test_ll = get_cmll(bma_test_probs, all_test_ys, eps=1e-4)
    
    test_acc, all_probs, all_ys = get_acc(net, testloader)
    test_nll = get_cmll(all_probs, all_ys, eps=1e-4)
    
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([num_params, prior_prec_init, best_temp,
                            test_acc, bma_test_accuracy, bma_accuracy,
                            test_nll, bma_test_ll, cmll])
              
              

wds = [100.0, 0.1, 0.01, 0.0001]
widths_cifar10 = [16, 24, 32, 40, 48] 
arch_names = ["fixup_resnet8", "fixup_resnet14", "fixup_resnet20", "fixup_resnet26", "fixup_resnet32"]

for decay in wds:              
    for arch in arch_names:
        for width in widths_cifar10:
            print("using the new split ....") 
            get_mll_acc(arch,
                        width,
                        prior_prec_init=decay,
                        partialtrain_chk_path=args.partialtrain_chk_path,
                        fulltrain_chk_path = args.fulltrain_chk_path,
                        data_ratio=args.data_ratio,
                        hessian_structure=args.hessian_structure, 
                        prior_structure = args.prior_structure,
                        base_lr = args.base_lr,
                        bma_nsamples=args.bma_nsamples,
                        max_iters = args.max_iters, 
                        result_folder=args.result_folder)
    