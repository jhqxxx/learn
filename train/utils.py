'''
Descripttion: 
version: 
Author: jhq
Date: 2022-10-22 19:09:32
LastEditors: jhq
LastEditTime: 2022-10-22 20:02:00
'''

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader
import os


def getMeanAndStd(train_data):
    '''
    Compute mean and varience for training data
    train_data: Dataset
    retur (mean, std)
    '''
    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:,d, :, :].mean()
            std[d] += X[:, d, :, :].std()


train_data = datasets.ImageFolder(root=r'F:\dataset\classify\flower_photos')