"""
Notes:
data: MNIST, cifar10, cifar100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import numpy as np
import os
import argparse

# model 9-CNN
from model.myModel import Net

# my function
from helper import noisyHelper
from loss import new_coteaching

# args
args = argparse.ArgumentParser()
args.add_argument('--data', help='data set', type=str, default='cifar10')
args.add_argument('--lr', help='Learning Rate', type=float, default=0.01)
args.add_argument('--NT', help='Noise type, symmetry or pair', type=str, default='symmetry')
args.add_argument('--NR', help='Noise Rate', type=float, default=0.4)
args.add_argument('--FR', help='Forget Rate', type=float, default=None)
args.add_argument('--batchSize', type=int, default=64)
args.add_argument('--epochs', type=int, default=200)
args.add_argument('--epochK', type=int, default=20)
args.add_argument('--SEED', type=int, default=924)
args = args.parse_args()

print('[INFO] Set noisy type: {}, noisy rate: {}'.format(args.NT, args.NR))

# some info
SEED = args.SEED
epochs = int(args.epochs)
epochK = int(args.epochK)
learningRate = float(args.lr)
momentum = 0.6
batchSize = int(args.batchSize)
noisyRate = float(args.NR)
N_type = args.NT
if args.FR == None:
    print('[INFO] Set forget rate: {} equals to noisy rate'.format(args.NR))
    forgetRate = float(args.NR)
else:
    forgetRate = float(args.FR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('[NOTICE] Using CPU')
else:
    print("[NOTICE] Using CUDA")

args.data = args.data.lower()
inchannel = 3
if args.data == 'MNIST':
    W_H = 8
    inchannel = 1
    outchannel = 10
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(W_H, padding=4), 
        # torchvision.transforms.RandomHorizontalFlip(), 
        torchvision.transforms.ToTensor(), 
    ])
    # load data
    print("[INFO] loading data..")
    filePath = "./"
    train_data = torchvision.datasets.MNIST(root=filePath, train=True, \
        transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root=filePath, train=False, \
        transform=transform, download=True)

elif args.data == 'cifar10':
    W_H = 32
    outchannel = 10
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(W_H, padding=4), 
        torchvision.transforms.RandomHorizontalFlip(), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # load data
    print("[INFO] loading data..")
    filePath = "./"
    train_data = torchvision.datasets.CIFAR10(root=filePath, train=True, \
        transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR10(root=filePath, train=False, \
        transform=transform, download=True)

elif args.data == 'cifar100':
    W_H = 32
    outchannel = 100
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(W_H, padding=4), 
        torchvision.transforms.RandomHorizontalFlip(), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # load data
    print("[INFO] loading data..")
    filePath = "./"
    train_data = torchvision.datasets.CIFAR100(root=filePath, train=True, \
        transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR100(root=filePath, train=False, \
        transform=transform, download=True)
else:
    AssertionError("Do not support {}".format(args.data))
    exit()

# noisy
train_data, noisy_idx = noisyHelper(train_data, noisyRate, SEED=SEED, N_type=N_type)
# init_epoch = 0
if noisyRate <= 0.5:
    init_epoch = 5
else:
    init_epoch = 7

# dataloader
train_loader = Data.DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=batchSize, shuffle=True)


# model_1, model_2: exchange with model_3; model_3 disagree with model_1, model_2
model_1 = Net(inchannel=inchannel, outchannel=outchannel)
model_2 = Net(inchannel=inchannel, outchannel=outchannel)
if device == 'cuda':
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
opt_1 = optim.SGD(model_1.parameters(), lr=learningRate, momentum=momentum, weight_decay=1e-5)
opt_2 = optim.SGD(model_2.parameters(), lr=learningRate, momentum=momentum, weight_decay=1e-5)


def train(epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        if device == 'cuda':
            data = Variable(data).cuda()
            targets = Variable(targets).cuda()
        else:
            data = Variable(data)
            targets = Variable(targets)
        opt_1.zero_grad()
        opt_2.zero_grad()
#         out_1 = model_1(data)
#         out_2 = model_2(data)
#         loss = F.nll_loss(output_1, targets)
        FR = min(forgetRate, epoch*forgetRate/epochK)
        loss_1, loss_2 = new_coteaching(data, targets, model_1, model_2, FR, epoch, init_epoch)
        loss_1.backward()
        loss_2.backward()
        opt_1.step()
        opt_2.step()
        if batch_idx % 200 == 0:
            print('[TRAIN] Train Epoch: {} [{}/{}]\tLoss: {:.6f} {:.6f}'.format(\
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),\
                loss_1.item(), loss_2.item()))

def test():
    model_2.eval()
    test_loss = 0
    correct = 0
    for data, targets in test_loader:
        if device == 'cuda':
            data = Variable(data).cuda()
            targets = Variable(targets).cuda()
        else:
            data = Variable(data)
            targets = Variable(targets)
        output = model_2(data)

        test_loss += F.nll_loss(output, targets, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print('\n[TEST]Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(\
        test_loss, correct, len(test_loader.dataset),\
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

acc = []
print('[INFO] Start training!')
for epoch in range(1, epochs):
    train(epoch)
    tmp = test()
    
    if epoch == 60:
        opt_1 = optim.SGD(model_1.parameters(), lr=learningRate*0.8, momentum=momentum, weight_decay=1e-5)
        opt_2 = optim.SGD(model_2.parameters(), lr=learningRate*0.8, momentum=momentum, weight_decay=1e-5)
    elif epoch == 100:
        opt_1 = optim.SGD(model_1.parameters(), lr=learningRate*0.5, momentum=momentum, weight_decay=1e-5)
        opt_2 = optim.SGD(model_2.parameters(), lr=learningRate*0.5, momentum=momentum, weight_decay=1e-5)
    elif epoch == 150:
        opt_1 = optim.SGD(model_1.parameters(), lr=learningRate*0.2, momentum=momentum, weight_decay=1e-5)
        opt_2 = optim.SGD(model_2.parameters(), lr=learningRate*0.2, momentum=momentum, weight_decay=1e-5)
    
    acc.append(tmp)
    np.save('./test_{}_NR{}_FR{}_SEED{}_type_{}.npy'.format(\
        args.data, 100*noisyRate, 100*forgetRate, SEED, N_type), acc)

print('Finished!!!')
