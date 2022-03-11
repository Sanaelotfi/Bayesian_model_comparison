'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
from laplace import Laplace
from laplace.curvature import AsdlEF

import new_models

import time

import numpy as np
import random
import math

from utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--decay', type=float, help='weight decay')
parser.add_argument('--hypers_lr', default=1.0, type=float, help='learning rate for hyperparameters')
parser.add_argument('--batchnorm', default=False, type=bool, help='whether or not to use batchnorm instead of fixup')
parser.add_argument('--use_sgdr', default=False, type=bool, help='whether or not to use cosine lr scheduler')
parser.add_argument('--base_lr', default=0.01, type=float, help='base learning rate')
parser.add_argument('--optimizehypers', default=False, type=bool, help='whether or not to optimize hyperparameters with LA')
parser.add_argument('--prior_structure', default='scalar', type=str, help='structure of the prior: scalar or layerwise')
parser.add_argument('--hessian_structure', default='kron', type=str, help='structure of the hessian: full, kron, diag')
parser.add_argument('--chk_path', default="checkpoint/cifar100/new_hypers/cnns", type=str, help='path to save the checkpoints')
parser.add_argument('--result_folder', default='./results/cifar100/new_hypers/cnns/', type=str, help='path to save the results')


args = parser.parse_args()


data_directory = "/datasets"

seed = 10
batchsize = 128

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

trainset = torchvision.datasets.CIFAR100(root=data_directory, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root=data_directory, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

def train_model(depth,
                width,
                prior_prec_init,
                prior_structure,
                hessian_structure,
                n_epoch=250,
                batchsize=128,
                seed=10,
                optimizehypers=False,
                hypers_lr = 1.0,
                freq=5,
                burnin=0,
                base_lr = 0.1,
                use_sgdr = True,
                hyperupdates = 100,
                batchnorm = False,
                chk_path = "checkpoint/cifar100/optimized_la",
                result_folder = './results/cifar100/optimized_la/',
                debug=False
               ):
    """
    la_kwargs = {hyperupdates = 100, freq=5, burnin=0} 
    """
    best_acc = 0  # best test accuracy
    batch_size = batchsize
    base_learning_rate = base_lr * batchsize / 128.
    print("optimizing with new hypers from Alex Immer")
    print("optimizehypers ", optimizehypers)
    
    if optimizehypers:
        path = "cifar100_" + str(prior_prec_init) + '_fixup_cnn' + str(depth) + '_' + str(width) + '_' + prior_structure + '_' + hessian_structure + '_hlr' + str(hypers_lr) + '_baselr_' + str(base_lr)
    else:
        path = "cifar100_" + str(prior_prec_init) + '_fixup_cnn' + str(depth) + '_' + str(width) + '_' + prior_structure + '_baselr_' + str(base_lr)
    if batchnorm:
        path = path + '_batchnorm'
        chk_path = chk_path + '/batchnorm'
        result_folder = result_folder + 'batchnorm/'
        print("this is the path for checkpoints: ", chk_path)
        print("this is the path for results: ", result_folder)
    print("This is the current path: ", path)
    
    
    if use_cuda:
        # data parallel
        n_gpu = torch.cuda.device_count()
        batch_size *= n_gpu
        base_learning_rate *= n_gpu
    
    
    # Model
    print("=> creating model 'fixup_cnn{}_{}'".format(depth, width))
    net = new_models.FixupCNN(in_pixels=32, in_channels=3, n_out=100, width=width, depth=depth)
    print("num of params: ", sum(p.numel() for p in net.parameters()))
    
    
    if optimizehypers:
        print("optimizing hyperparameters with Laplace approximation!")

    if use_cuda:
        net.cuda()
#         net = torch.nn.DataParallel(net)
#         print('Using', torch.cuda.device_count(), 'GPUs.')
#         cudnn.benchmark = True
        print('Using CUDA..')
    device = parameters_to_vector(net.parameters()).device

    # define optimization process for net weights
    cel = nn.CrossEntropyLoss()
    criterion = lambda pred, target, lam: (-F.log_softmax(pred, dim=1) * torch.zeros(pred.size()).cuda().scatter_(1, target.data.view(-1, 1), lam.view(-1, 1))).sum(dim=1).mean()
    parameters_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    # setting the weight decay ourselves
    optimizer = optim.SGD(
            [{'params': parameters_bias, 'lr': base_lr/10.}, 
            {'params': parameters_scale, 'lr': base_lr/10.}, 
            {'params': parameters_others}], 
            lr=base_learning_rate, 
            momentum=0.9,
            weight_decay=0.0)
        
    # making results directory
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # log for keeping accuracy 
    logname = result_folder + path + '.csv'
#     if not debug:
    if not os.path.exists(logname) and not debug:
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc', 'decay'])
            
    # log for keeping decay and marglik
    logname_la = result_folder + path + "_la" + '.csv'
    if not os.path.exists(logname_la) and not debug:
#     if not debug:
        with open(logname_la, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'step', 'decay', 'marglik'])

    
    # differentiable hyperparameters
    N = len(trainloader.dataset)
    H = len(list(net.parameters()))
    P = len(parameters_to_vector(net.parameters()))
    hyperparameters = list()
    # prior precision
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    hyperparameters.append(log_prior_prec)
    
    # set up hyperparameter optimizer
    hyper_optimizer = torch.optim.Adam(hyperparameters, lr=hypers_lr)
    
    sgdr = CosineAnnealingLR(optimizer, n_epoch, eta_min=0, last_epoch=-1)
 
    
    for epoch in range(n_epoch):
        lr = 0.
        if use_sgdr:
            sgdr.step()
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
        else:
            lr = adjust_learning_rate(optimizer, epoch, base_learning_rate)
        prior_prec = torch.exp(log_prior_prec).detach()
        train_loss, train_acc = train(epoch, net, trainloader, optimizer, criterion, prior_prec, batchnorm)
        test_loss, test_acc, best_acc = test(epoch, net, testloader, best_acc, path, chk_path, debug)
        
        
        if not debug: 
            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, lr, train_loss, train_acc, test_loss, test_acc, prior_prec])
        
        # optimize the prior precision
        if (epoch+1)%freq == 0 and epoch >= burnin and optimizehypers:
            prior_prec = torch.exp(log_prior_prec)
            la = Laplace(net, 'classification', subset_of_weights='all', hessian_structure=hessian_structure,
                         prior_precision=prior_prec, backend=AsdlEF)
            start = time.time()
            print("fitting la ...")
            la.fit(trainloader)
            print("done la ...")
            end = time.time()
            print("time to estimate LA", end - start)
            for step in range(hyperupdates):
                hyper_optimizer.zero_grad()
                prior_prec = torch.exp(log_prior_prec)
                neg_marglik = - la.log_marginal_likelihood(prior_precision=prior_prec)/N
                neg_marglik.backward()
                hyper_optimizer.step()
                marglik = - neg_marglik.item()
                if not debug:
                    with open(logname_la, 'a') as logfile:
                        logwriter = csv.writer(logfile, delimiter=',')
                        if prior_structure=='scalar':
                            logwriter.writerow([epoch, step, torch.exp(log_prior_prec).item(), marglik])
                        else:
                            logwriter.writerow([epoch, step, torch.exp(log_prior_prec).detach(), marglik])
            if prior_structure=='scalar':
                print('prior precision: ', torch.exp(log_prior_prec).item(), 'marglik: ', marglik)
            else:
                print('prior precision: ', torch.exp(log_prior_prec).detach(), 'marglik: ', marglik)
            
    if not debug:    
        new_path = path + '_final'
        checkpoint(test_acc, epoch, new_path, net, chk_path)
        
    
    print("getting LA")
    prior_prec = torch.exp(log_prior_prec)

    la = Laplace(net, 'classification', subset_of_weights='all',
                 hessian_structure="diag", prior_precision=prior_prec,
                 backend=AsdlEF)
    print("fitting LA")
    la.fit(trainloader)
    print("computing ML")
    marginal_likelihood = la.log_marginal_likelihood(prior_precision=prior_prec).item()/N
    print("la approx: ", marginal_likelihood)
    if not debug:
        with open(logname_la, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if prior_structure=='scalar':
                logwriter.writerow([epoch, "final", prior_prec.item(), marginal_likelihood])
            else:
                logwriter.writerow([epoch, "final", prior_prec.detach(), marginal_likelihood])
            
    
widths_cifar100_cnn = [4, 8, 16, 32, 64]
depths_cifar100_cnn = [1, 2, 3, 4, 5]


for depth in depths_cifar100_cnn:
    for width in widths_cifar100_cnn:
        train_model(depth,
                    width,
                    args.decay,
                    args.prior_structure,
                    args.hessian_structure,
                    base_lr = args.base_lr,
                    use_sgdr = args.use_sgdr,
                    optimizehypers = args.optimizehypers,
                    hypers_lr = args.hypers_lr,
                    batchnorm = args.batchnorm,
                    chk_path = args.chk_path,
                    result_folder = args.result_folder)