from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from models import *
from funcs import *
from count_ops import *
'''
path = '/home/vinita/Downloads/cifar10/'
#out_file = open("/home/vinita/Downloads/pytorch-blockswap-master/models/"+"Inference_Time"+".txt", "w+")
transform_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

trainset = torchvision.datasets.CIFAR10(root=path, train=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=path, train=False, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
'''

device = torch.device('cpu')

import sys
import re

out_file = open("/home/vinita/pytorch-blockswap-master/"+"Validation_results"+ ".txt", "w+")
base_dir = "/home/vinita/pytorch-blockswap-master/genotypes/"
max_itr = 1000
for fle in os.listdir(base_dir):
    if fle.endswith(".csv"):
        in_file = open(os.path.join(base_dir,fle))
        row = pd.read_csv(in_file)
        cs = row['convs'][0][1:-1].rstrip().replace(',', '')
        cs = cs.replace('\n', ' ').split()

        r = [w for w in cs if '<class' not in w]

        prefix = len("'models.blocks.")
        suffix = len("'>")
        new_convs = [c[prefix:-suffix] for c in r]
        print(fle)
        print(new_convs)
        print("\n\n\n")
        out_file.write("model"+str(fle)+'\n')
        net = WideResNet(40, 2, Conv, BasicBlock)
        for i, c in enumerate(new_convs):
            net = update_block(i, net, string_to_conv[c], mask=False)
        average_time = 0
        for i in range(0, max_itr):
            start_time = time.time()
            total_correct = 0
            total_images = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs, _  = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_images += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                    break
                average_time = average_time + (time.time()-start_time) * 100

        count_ops, count_param = measure_model(net, 40, 2)
        out_file.write('       param_count = %d    count_ops = %d '%(count_param, count_ops))
        out_file.write('Inference time: %d   millsecond'%(round(average_time/max_itr)))
        out_file.write('\n\n')
        in_file.close()

out_file.close()
