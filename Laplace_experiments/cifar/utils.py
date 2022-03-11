'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

import numpy as np
import torch

use_cuda = torch.cuda.is_available()

def mixup_data(x, y, alpha=1.0, use_cuda=True, per_sample=False):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    if alpha > 0. and not per_sample:
        lam = torch.zeros(y.size()).fill_(np.random.beta(alpha, alpha)).cuda()
        mixed_x = lam.view(-1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1)) * x[index,:]
    elif alpha > 0.:
        lam = torch.Tensor(np.random.beta(alpha, alpha, size=y.size())).cuda()
        mixed_x = lam.view(-1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1)) * x[index,:]
    else:
        lam = torch.ones(y.size()).cuda()
        mixed_x = x

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_lam_idx(batch_size, alpha, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    return lam, index    

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: criterion(pred, y_a, lam) + criterion(pred, y_b, 1 - lam)

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


def expand_prior_precision(prior_prec, net):
    theta = parameters_to_vector(net.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, net.parameters())])


def train(epoch, net, trainloader, optimizer, criterion, prior_prec, batchnorm):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # no mixup
    alpha = 0.0
    N = len(trainloader.dataset)
    cel = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        theta = parameters_to_vector(net.parameters())
        wdecay = expand_prior_precision(prior_prec, net)
            
        if batchnorm: 
            outputs = net(inputs)
            loss = cel(outputs, targets) + (0.5 * (wdecay * theta) @ theta) / N
            
        else:
            # generate mixed inputs, two one-hot label vectors and mixing coefficient
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, use_cuda)
            optimizer.zero_grad()
            outputs = net(inputs)
        

    #         print("this is the weight decay in train: ", wdecay)
            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs) + (0.5 * (wdecay * theta) @ theta) / N
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        if batchnorm:
            correct += predicted.eq(targets.data.float()).cpu().sum()
        else:
            correct += (lam * predicted.eq(targets_a.data).float()).cpu().sum() + ((1 - lam) * predicted.eq(targets_b.data).float()).cpu().sum()
        acc = 100.*float(correct)/float(total)
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), acc, correct, total))

    return (train_loss/batch_idx, acc)


def test(epoch, net, testloader, best_acc, path, chk_path, debug):
    net.eval()
    cel = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = cel(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))
        
        # Save checkpoint.
        acc = 100.*float(correct)/float(total)
        if acc > best_acc:
            best_acc = acc
            if not debug:
                checkpoint(acc, epoch, path, net, chk_path)

    return (test_loss/batch_idx, acc, best_acc)

def checkpoint(acc, epoch, path, net, chk_path):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net_state_dict': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    # load with net.load_state_dict(torch.load(path))
    if not os.path.isdir(chk_path):
        os.mkdir(chk_path)
    new_path = './' + chk_path + '/' + path + '.ckpt'
    torch.save(state, new_path)

def adjust_learning_rate(optimizer, epoch, base_learning_rate):
    """decrease the learning rate at 150 and 225 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibutatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 150:
        lr /= 10
    if epoch >= 225:
        lr /= 10
    for param_group in optimizer.param_groups:
        if param_group['initial_lr'] == base_learning_rate:
            param_group['lr'] = lr
        else:
            if epoch <= 9:
                param_group['lr'] = param_group['initial_lr'] * lr / base_learning_rate
            elif epoch < 150:
                param_group['lr'] = param_group['initial_lr']
            elif epoch < 225:
                param_group['lr'] = param_group['initial_lr'] / 10.
            else:
                param_group['lr'] = param_group['initial_lr'] / 100.
    return lr