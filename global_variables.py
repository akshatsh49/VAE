import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import statistics
import matplotlib.pyplot as plt
import pickle
import os 

if(torch.cuda.is_available()):
    device='cuda:0'
else :
    device='cpu'

batch_size=64
train_loader=torch.utils.data.DataLoader(torchvision.datasets.MNIST('./root',train=True,download=True,transform=torchvision.transforms.ToTensor()),batch_size=batch_size,drop_last=True)
test_loader=torch.utils.data.DataLoader(torchvision.datasets.MNIST('./root',train=False,transform=torchvision.transforms.ToTensor()),batch_size=batch_size,drop_last=True)

sample_folder='sample_folder'
saved_models='saved_models'
loss='loss'
test_folder='test_folder'
reconst_folder='reconst_folder'
space_interpolations='space_interpolations'

print("Device : {}".format(device))