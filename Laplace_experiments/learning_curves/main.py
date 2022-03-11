'''Train CIFAR10 with PyTorch.'''
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import SubsetRandomSampler,Dataset
import torch.utils.data 
import torchvision
import torchvision.transforms as transforms
import argparse
from models import *
from utils import get_indices, save_output_from_dict, conv_helper, count_parameters, to_json, get_dataset
from time import time, strftime, localtime
from statistics import pstdev
import random
import numpy as np

from laplace import Laplace
from laplace.curvature import AsdlEF

from torch.nn.utils import parameters_to_vector, vector_to_parameters
import tqdm

def train(loader):
    #print('\nEpoch: %d' % epoch)
    net.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
#            if args.not_CML:
#                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if args.debug:
            break
    train_loss = running_loss/(batch_idx+1)
    print(epoch, 'loss: %.4f' %train_loss)
    return train_loss

def test(loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    time0 = time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.not_CML:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if args.debug:
                break
    net.train()
    return 100.*correct/total, ((time() - time0)/60) 

def train_acc(loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.debug:
                break
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='LeNet', type=str, help='name of architecture')
    parser.add_argument('--data_dir', default='~/data', type=str, help='data directory')
    parser.add_argument('--save_path', default='./tables/', type=str, help='path for saving tables')
    parser.add_argument('--runs_save_name', default='runs.csv', type=str, help='name of runs file')
    parser.add_argument('--save_name', default='table.csv', type=str, help='name of experiment (>= 1 run) file')
    # this was substituted with "trainsize", the actual size of the training set that the model should train on 
#     parser.add_argument('--num_per_class', default=-1, type=int, help='number of training samples per class')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--not_CML', action='store_true', help='debug mode')
    parser.add_argument('--width_multiplier', default=1, type=float, help='width multiplier of ResNets or ResNetFlexes')
    parser.add_argument('--width', default=200, type=int, help='width of MLP')
    parser.add_argument('--depth', default=3, type=int, help='depth of MLP or depth of ResNetFlex')
    parser.add_argument('--conv_width', default=32, type=int, help='width parameter for convnet')
    parser.add_argument('--conv_depth', default=0, type=int, help='depth parameter for convnet')
    parser.add_argument('--num_filters', nargs='+', type=int, help='number of filters per layer for CNN')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in classification')
    parser.add_argument('--seed', default=None, type=int, help='seed for subset')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=600, type=int, help='num epochs to train')
    parser.add_argument('--test_freq', default=5, type=int, help='frequency which to test')
    parser.add_argument('--num_runs', default=1, type=int, help='num runs to avg over')
    parser.add_argument('--save_model', action='store_true', help='save model state_dict()')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument("--save_json", action="store_true", help="save json?")
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='name of dataset')
    parser.add_argument('--run_label', default='runlbl', type=str, help='label of run')
    parser.add_argument('--decay', default=5e-4, type=float, help='weight decay for the optimizer')
    parser.add_argument('--sample_batch_size', default=16, type=int, help='batch size for the validation set -- involves sampling from LA')
    parser.add_argument('--trainsize', default=250, type=int, help='weight decay for the optimizer')
    parser.add_argument('--hessian_structure', default='diag', type=str, help='structure of the hessian')
    parser.add_argument('--bma_nsamples', default=20, type=int, help='whether or not to use bma to get CMLL')
    parser.add_argument('--init_temp', default=1e-3, type=float, help='intial temperature for rescaling la covariance')
    parser.add_argument('--run_id', default=int, type=int, help='id of the run')
    args = parser.parse_args()
    print(args)

    if args.not_CML:
        from progress_bar import progress_bar

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Preparing data..')
    trainset, testset = get_dataset(args.data_dir, args.dataset, args.num_classes)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, num_workers=2, shuffle=False)
    num_good_runs = 0
    global_train_accuracy = 0
    test_acc_list = []
    global_test_acc = 0
    bma_test_acc_list = []
    bma_test_ll_list = []
    cmll_list = []
    mll_list = []
    
    data_directory = "./data"
    # making data directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    path = data_directory + "/cifar10_learningcurves_subsets.npz"
    
    if not os.path.exists(path): 
        n = len(trainset.data)
        idx = np.arange(n)
        np.random.shuffle(idx)
        
        n1, n2 = int(n * 0.9), int(n * 0.1)
        subset_1 = idx[:n1]
        subset_2 = idx[n1:n1+n2]
        
        x, y = trainset.data, np.array(trainset.targets)
        # training data
        x1, y1 = x[subset_1], y[subset_1]
        # data used to tune the laplace covariance matrix scale 
        x2, y2 = x[subset_2], y[subset_2]
        
        # saving the data 
        np.savez(path, 
         x1=x1,
         y1=y1,
         x2=x2,
         y2=y2)
   
    subsets_data = np.load(path)
        
    # getting the train and validation loaders 
    trainset.data = subsets_data["x1"]
    trainset.targets = subsets_data["y1"].tolist()

    validset, _ = get_dataset(args.data_dir, args.dataset, args.num_classes)

    validset.data = subsets_data["x2"]
    validset.targets = subsets_data["y2"].tolist()
    
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.sample_batch_size, shuffle=False, num_workers=2)

    
    for run in range(args.num_runs):
    
        # make the full trainset then subset that to two smaller sets 

        # making a trainset with the size we want 
        
        path = data_directory + "/cifar10_subset_train_run_size_{}_{}.npz".format(args.trainsize, args.run_id)
        
        if not os.path.exists(path):
        
            n = len(trainset.data)
            idx = np.arange(n)
            np.random.shuffle(idx)

            subset_1 = idx[:args.trainsize]

            x, y = trainset.data, np.array(trainset.targets)
            # training data
            x1, y1 = x[subset_1], y[subset_1]
            
            np.savez(path, 
                    x1=x1,
                    y1=y1)
            
        subsets_data = np.load(path)
        
        # creating the new train and new train-test 
        trainset.data = subsets_data["x1"]
        trainset.targets = subsets_data["y1"].tolist()
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, num_workers=2, shuffle=True)
        
        # now subset that to two loaders 
        # making a subset for the training and test for CMLL 
        
        path = data_directory + "/cifar10_subsets_icmll_run_size_{}_{}.npz".format(args.trainsize, args.run_id)
        
        if not os.path.exists(path):
        
            n = len(trainset.data)
            idx = np.arange(n)
            np.random.shuffle(idx)

            n1, n2 = int(n * 0.8), int(n * 0.2)
            subset_1 = idx[:n1]
            subset_2 = idx[n1:n1+n2]

            x, y = trainset.data, np.array(trainset.targets)
            # training data
            x1, y1 = x[subset_1], y[subset_1]
            # data used to tune the laplace covariance matrix scale 
            x2, y2 = x[subset_2], y[subset_2]
            
            np.savez(path, 
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2)
            
        subsets_data = np.load(path)
        
        # creating the new train and new train-test 
        train_cmll, _ = get_dataset(args.data_dir, args.dataset, args.num_classes)
        train_cmll.data = subsets_data["x1"]
        train_cmll.targets = subsets_data["y1"].tolist()

        test_cmll, _ = get_dataset(args.data_dir, args.dataset, args.num_classes)

        test_cmll.data = subsets_data["x2"]
        test_cmll.targets = subsets_data["y2"].tolist()
    
        # dataset to compute the CMLL on 
        test_cmllloader = torch.utils.data.DataLoader(test_cmll, batch_size=args.sample_batch_size, shuffle=False, num_workers=2)    
        
        # dataset to train the model on
        train_cmllloader = torch.utils.data.DataLoader(
            train_cmll, batch_size=args.batch_size, num_workers=2, shuffle=True)
        
        
        print("this is the size of the full train loader {}, \
        CMLL train loader: {}, temp valid train {}, CMLL test train {}".format(len(trainloader.dataset),
                                                                    len(train_cmllloader.dataset),
                                                                    len(validloader.dataset),
                                                                    len(test_cmllloader.dataset)))

        # Model
        print('==> Building model for CMLL training..')
        if args.model == 'VGG19':
            net = VGG('VGG19', num_classes=args.num_classes)
        elif args.model == 'VGG11':
            net = VGG('VGG11', num_classes=args.num_classes)
        elif args.model == 'VGG13':
            net = VGG('VGG13', num_classes=args.num_classes)
        elif args.model == 'VGG16':
            net = VGG('VGG16', num_classes=args.num_classes)
        elif args.model == 'ResNet6':
            net = ResNet6(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet8':
            net = ResNet8(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet10':
            net = ResNet10(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet12':
            net = ResNet12(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet14':
            net = ResNet14(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet16':
            net = ResNet16(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet18':
            net = ResNet18(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex':
            net = ResNetFlex(num_classes=args.num_classes, width_multiplier=args.width_multiplier, depth=args.depth)
        elif args.model == 'ResNetFlex34':
            net = ResNetFlex34(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex50':
            net = ResNetFlex50(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex101':
            net = ResNetFlex101(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex152':
            net = ResNetFlex152(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == PreActResNet18():
            net = PreActResNet18()
        elif args.model == 'GoogLeNet':
            net = GoogLeNet(num_classes=args.num_classes)
        elif args.model == 'LeNet':
            net = LeNet(num_classes=args.num_classes)
        elif args.model == 'DenseNet121':
            net = DenseNet121(num_classes=args.num_classes)
        elif args.model == 'DenseNet169':
            net = DenseNet169(num_classes=args.num_classes)
        elif args.model == 'DenseNet201':
            net = DenseNet201(num_classes=args.num_classes)
        elif args.model == 'DenseNet161':
            net = DenseNet161(num_classes=args.num_classes)
        elif args.model == 'ResNeXt29_2x64d':
            net = ResNeXt29_2x64d(num_classes=args.num_classes)
        elif args.model == 'MobileNet':
            net = MobileNet(num_classes=args.num_classes)
        elif args.model == 'MobileNetV2':
            net = MobileNetV2(num_classes=args.num_classes)
        elif args.model == 'DPN26':
            net = DPN26(num_classes=args.num_classes)
        elif args.model == 'DPN92':
            net = DPN92(num_classes=args.num_classes)
        elif args.model == 'ShuffleNetG2':
            net = ShuffleNetG2(num_classes=args.num_classes)
        elif args.model == 'ShuffleNetV2':
            net = ShuffleNetV2(net_size=1, num_classes=args.num_classes)
        elif args.model == 'SENet18':
            net = SENet18(num_classes=args.num_classes)
        elif args.model == 'EfficientNetB0':
            net = EfficientNetB0(num_classes=args.num_classes)
        elif args.model == 'RegNetX_200MF':
            net = RegNetX_200MF(num_classes=args.num_classes)
        elif args.model == 'RegNetX_400MF':
            net = RegNetX_400MF(num_classes=args.num_classes)
        elif args.model == 'RegNetY_400MF':
            net = RegNetY_400MF(num_classes=args.num_classes)
        elif args.model == 'PNASNetA':
            net = PNASNetA()
        elif args.model == 'AlexNet':
            net = AlexNet(num_classes=args.num_classes)
        elif args.model == 'MLP':
            net = MLP(width=args.width, depth=args.depth)
        elif args.model == 'CNN':
            # below commented out by lf for ease of producing heat maps
            net = CNN(num_filters=args.num_filters)
            #net = CNN(num_filters=conv_helper(args.width, args.depth))
        elif args.model == 'convnet':
            net = convnet(width=args.conv_width, depth=args.conv_depth)
        elif args.model == 'SENet18_DPN92':
            net = SENet18_DPN92(num_classes=args.num_classes)

        print('model ', args.model)
        print('width_multiplier ', args.width_multiplier)
        ctParams = count_parameters(net)
        print('ctParams ', ctParams)

        net = net.to(device)
        print("torch.cuda.device_count() ", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
                          
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epochs/2), int(args.epochs * 3/4)])
#        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones =[150, 225], gamma =0.1)
        test_acc = 0
        time0 = time()
        for epoch in range(args.epochs):
            train_loss = train(train_cmllloader)
            '''
            if epoch % args.test_freq == 0:
                test_acc = test(epoch)
            '''
            if epoch > 99 and epoch % 10 == 0:
                train_accuracy = train_acc(train_cmllloader)
                #test_acc, test_time = test()
                print('\t\ttrain_acc: %.4f' %train_accuracy)
                #print('\t\ttest_acc: %.4f' %test_acc)
                
                if train_accuracy >= 99.9:
                    test_acc, test_time = test(testloader)
                    break
                elif epoch > 199 and train_accuracy >= 99.5:
                    test_acc, test_time = test(testloader)
                    break
                elif epoch > 299 and train_accuracy >= 99.0:
                    test_acc, test_time = test(testloader)
                    break

            if epoch == (args.epochs - 1):
                test_acc, test_time = test(testloader)
                train_accuracy = train_acc(train_cmllloader)

            scheduler.step()
            if args.debug:
                break

        print('train_acc for CMLL', train_accuracy)
        print('test_acc for CMLL', test_acc)
        train_time = (time() - time0)/60
        print('training time in mins ', train_time)
        
        # saving this model:         
        
        print("fitting LA for the trainset")
        la = Laplace(net, 'classification', subset_of_weights='all',
                 hessian_structure=args.hessian_structure, prior_precision=args.decay,
                 backend=AsdlEF)
        la.fit(train_cmllloader)
        print("done fitting LA for the small data")
        
        
        epoch = 0
        temp = args.init_temp
        best_accuracy = 0
        best_temp = temp
        buffer = 0
        max_buffer = 3
        max_epochs = 50
        while epoch < max_epochs: 
            bma_accuracy, bma_probs, all_ys = get_bma_acc(net, la, validloader, args.bma_nsamples, args.hessian_structure, temp=temp)
            cmll = get_cmll(bma_probs, all_ys, eps=1e-4)
            print("current temperate: {}, bma accuracy: {}, cmll: {}".format(temp, bma_accuracy, cmll))
            if bma_accuracy > best_accuracy:
                best_accuracy = bma_accuracy
                best_temp = temp
                temp /= 10.
                buffer = 0
            elif buffer < max_buffer:
                buffer += 1
                temp /= 5.
            else:
                break
            epoch += 1

        print("best temperate: {}, bma accuracy: {}".format(best_temp, best_accuracy))
        
        # using the best temperature to get 
        bma_accuracy, bma_probs, all_ys = get_bma_acc(net,
                                                      la,
                                                      test_cmllloader,
                                                      args.bma_nsamples, args.hessian_structure, temp=best_temp)
        cmll = get_cmll(bma_probs, all_ys, eps=1e-4)
        
        print("cmll value (if nan, increase the initial temperature): ", cmll)
        
        
        # train the full model 
        
        # Model
        print('==> Building model for full training..')
        if args.model == 'VGG19':
            net = VGG('VGG19', num_classes=args.num_classes)
        elif args.model == 'VGG11':
            net = VGG('VGG11', num_classes=args.num_classes)
        elif args.model == 'VGG13':
            net = VGG('VGG13', num_classes=args.num_classes)
        elif args.model == 'VGG16':
            net = VGG('VGG16', num_classes=args.num_classes)
        elif args.model == 'ResNet6':
            net = ResNet6(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet8':
            net = ResNet8(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet10':
            net = ResNet10(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet12':
            net = ResNet12(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet14':
            net = ResNet14(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet16':
            net = ResNet16(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNet18':
            net = ResNet18(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex':
            net = ResNetFlex(num_classes=args.num_classes, width_multiplier=args.width_multiplier, depth=args.depth)
        elif args.model == 'ResNetFlex34':
            net = ResNetFlex34(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex50':
            net = ResNetFlex50(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex101':
            net = ResNetFlex101(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == 'ResNetFlex152':
            net = ResNetFlex152(num_classes=args.num_classes, width_multiplier=args.width_multiplier)
        elif args.model == PreActResNet18():
            net = PreActResNet18()
        elif args.model == 'GoogLeNet':
            net = GoogLeNet(num_classes=args.num_classes)
        elif args.model == 'LeNet':
            net = LeNet(num_classes=args.num_classes)
        elif args.model == 'DenseNet121':
            net = DenseNet121(num_classes=args.num_classes)
        elif args.model == 'DenseNet169':
            net = DenseNet169(num_classes=args.num_classes)
        elif args.model == 'DenseNet201':
            net = DenseNet201(num_classes=args.num_classes)
        elif args.model == 'DenseNet161':
            net = DenseNet161(num_classes=args.num_classes)
        elif args.model == 'ResNeXt29_2x64d':
            net = ResNeXt29_2x64d(num_classes=args.num_classes)
        elif args.model == 'MobileNet':
            net = MobileNet(num_classes=args.num_classes)
        elif args.model == 'MobileNetV2':
            net = MobileNetV2(num_classes=args.num_classes)
        elif args.model == 'DPN26':
            net = DPN26(num_classes=args.num_classes)
        elif args.model == 'DPN92':
            net = DPN92(num_classes=args.num_classes)
        elif args.model == 'ShuffleNetG2':
            net = ShuffleNetG2(num_classes=args.num_classes)
        elif args.model == 'ShuffleNetV2':
            net = ShuffleNetV2(net_size=1, num_classes=args.num_classes)
        elif args.model == 'SENet18':
            net = SENet18(num_classes=args.num_classes)
        elif args.model == 'EfficientNetB0':
            net = EfficientNetB0(num_classes=args.num_classes)
        elif args.model == 'RegNetX_200MF':
            net = RegNetX_200MF(num_classes=args.num_classes)
        elif args.model == 'RegNetX_400MF':
            net = RegNetX_400MF(num_classes=args.num_classes)
        elif args.model == 'RegNetY_400MF':
            net = RegNetY_400MF(num_classes=args.num_classes)
        elif args.model == 'PNASNetA':
            net = PNASNetA()
        elif args.model == 'AlexNet':
            net = AlexNet(num_classes=args.num_classes)
        elif args.model == 'MLP':
            net = MLP(width=args.width, depth=args.depth)
        elif args.model == 'CNN':
            # below commented out by lf for ease of producing heat maps
            net = CNN(num_filters=args.num_filters)
            #net = CNN(num_filters=conv_helper(args.width, args.depth))
        elif args.model == 'convnet':
            net = convnet(width=args.conv_width, depth=args.conv_depth)
        elif args.model == 'SENet18_DPN92':
            net = SENet18_DPN92(num_classes=args.num_classes)

        print('model ', args.model)
        print('width_multiplier ', args.width_multiplier)
        ctParams = count_parameters(net)
        print('ctParams ', ctParams)

        net = net.to(device)
        print("torch.cuda.device_count() ", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
                          
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epochs/2), int(args.epochs * 3/4)])
#        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones =[150, 225], gamma =0.1)
        test_acc = 0
        time0 = time()
        for epoch in range(args.epochs):
            train_loss = train(trainloader)
            '''
            if epoch % args.test_freq == 0:
                test_acc = test(epoch)
            '''
            if epoch > 99 and epoch % 10 == 0:
                train_accuracy = train_acc(trainloader)
                #test_acc, test_time = test()
                print('\t\ttrain_acc: %.4f' %train_accuracy)
                #print('\t\ttest_acc: %.4f' %test_acc)
                
                if train_accuracy >= 99.9:
                    test_acc, test_time = test(testloader)
                    break
                elif epoch > 199 and train_accuracy >= 99.5:
                    test_acc, test_time = test(testloader)
                    break
                elif epoch > 299 and train_accuracy >= 99.0:
                    test_acc, test_time = test(testloader)
                    break

            if epoch == (args.epochs - 1):
                test_acc, test_time = test(testloader)
                train_accuracy = train_acc(trainloader)

            scheduler.step()
            if args.debug:
                break

        print('train_acc for CMLL', train_accuracy)
        print('test_acc for CMLL', test_acc)
        train_time = (time() - time0)/60
        print('training time in mins ', train_time)
        
        
        print("fitting LA for the trainset")
        la = Laplace(net, 'classification', subset_of_weights='all',
                 hessian_structure=args.hessian_structure, prior_precision=args.decay,
                 backend=AsdlEF)
        la.fit(trainloader)
        print("done fitting LA for the small data")        
        
        # Getting the MLL on the fully trained model
        mll = la.log_marginal_likelihood(prior_precision=args.decay).item()/len(trainset.data)
        
        bma_test_accuracy, bma_test_probs, all_test_ys = get_bma_acc(net, la, testloader,
                                                                     args.bma_nsamples,
                                                                     args.hessian_structure,
                                                                     temp=best_temp)
        bma_test_ll = get_cmll(bma_test_probs, all_test_ys, eps=1e-4)
        
        bma_test_acc_list.append(bma_test_accuracy)
        bma_test_ll_list.append(bma_test_ll)
        cmll_list.append(cmll)
        mll_list.append(mll)
        print("bma_test_ll: {}, bma_test_accuracy: {}, mll {} ".format(bma_test_ll, bma_test_accuracy, mll))
        

        save_dict = {'model': args.model, 'wid_multi': args.width_multiplier, 'trainsize': args.trainsize, 'test_acc': test_acc, 'train_acc': train_accuracy, 'train_loss': train_loss, 'ctParams': ctParams, 'GPUs': torch.cuda.device_count(), 'epochs' : epoch,   'train_time': train_time, 'test_time': test_time, 'time': strftime("%m/%d %H:%M", localtime()), 'data': args.dataset,  'run_lbl': args.run_label, 'lr' : args.lr, 'maxEpochs' : args.epochs, 'bma_test_acc': bma_test_accuracy, 'bma_test_ll': bma_test_ll, 'cmll': cmll, 'mll': mll}
        fileName = args.dataset + args.runs_save_name
        save_output_from_dict(save_dict, save_dir=args.save_path, save_name=fileName)

        if train_accuracy >= 99.0:
            num_good_runs = num_good_runs + 1
            test_acc_list.append(test_acc)
            global_test_acc += test_acc
            global_train_accuracy += train_accuracy
        else:
            print("train_accuracy < 98.0\n")

        if args.debug:
            break

    if num_good_runs < 1:
        global_test_stdev = -1
    else:   
        global_test_stdev = pstdev(test_acc_list)
        global_test_acc /= num_good_runs
        global_train_accuracy /= num_good_runs
    avg_bma_test_acc = np.mean(bma_test_acc_list)
    avg_bma_test_ll = np.mean(bma_test_ll)
    avg_cmll = np.mean(cmll_list)
    avg_mll = np.mean(mll_list)    

    save_dict = {'model': args.model, 'wid_multi': args.width_multiplier, 'trainsize': args.trainsize, 'glb_test_acc': global_test_acc, 'glb_train_acc': global_train_accuracy, 'glb_test_stdev': global_test_stdev, 'ctParams': ctParams, 'GPUs': torch.cuda.device_count(), 'time': strftime("%m/%d %H:%M", localtime()), 'data': args.dataset,  'run_lbl': args.run_label,'good_runs': num_good_runs, 'lr' : args.lr, "avg_bma_test_acc": avg_bma_test_acc, "avg_bma_test_ll": avg_bma_test_ll, "avg_cmll": avg_cmll, "avg_mll": avg_mll}
#move num_good_runs right after stdev in next set of runs
    save_output_from_dict(save_dict, save_dir=args.save_path, save_name=args.save_name)
    if args.save_model:
        ckptFile = args.save_path+args.model+'_'+str(args.width_multiplier)+'_'+str(int(time()))+'.pt'
        print(ckptFile)
        torch.save(net.state_dict(), ckptFile)

    if args.save_json:
        stats = OrderedDict([("model", args.model),
                             ("test_acc", global_test_acc),
                             ("running_loss", running_loss),
                             ("width_multiplier", args.width_multiplier),
                             ("width", args.width),
                             ("depth", args.depth),
                             ("conv_width", args.conv_width),
                             ("conv_depth", args.conv_depth),
                             ("seed", args.seed),
                             ("epochs", args.epochs),
                             ("batch_size", args.batch_size)
                             ])
        to_json(stats, args.save_path)
