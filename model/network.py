# -- coding:utf-8 --
# Time:18 9月 2026 18:49
# Author YI LIAO(Steven Leo)
# File:network.py
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import math
import random
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler, SGD

class BNReLU(nn.Module):
    def __init__(self, in_channel=32, dropout=0.0):
        super(BNReLU, self).__init__()
        self.in_channel=in_channel
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(in_channel)
        self.relu=nn.ReLU(inplace=True)
        self.dropout_layer=nn.Dropout(dropout)

    def forward(self,x):
        x=self.bn(x)
        x=self.relu(x)
        if self.dropout>0.0:
            x=self.dropout_layer(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=16, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=self.compute_padding(stride)
        self.conv=nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding=self.padding)
    def compute_padding(self,stride):
        if stride==1:
            #padding=int((self.kernel_size-1)/2) if (self.kernel_size-1) is ood, use "same" to keep the size unchanged.
            return "same"
        else:
            return int((self.kernel_size-2)/2)
    def forward(self, x):
        x=self.conv(x)
        return x
class ResidualBlock1(nn.Module):
    def __init__(self, num_kernels=32, subsample_length=1, params_dict=None):
        super(ResidualBlock1, self).__init__()
        self.window_size=subsample_length
        self.pooling = nn.MaxPool1d(kernel_size=self.window_size, stride=self.window_size,padding=0, return_indices=False)
        self.layer=nn.Sequential(
            ConvLayer(num_kernels, num_kernels, params_dict["conv_filter_length"], stride=subsample_length),
            BNReLU(num_kernels, dropout=params_dict["conv_dropout"]),
            ConvLayer(num_kernels,num_kernels,params_dict["conv_filter_length"], stride=1),
        )
    def forward(self,x):
        residual=self.pooling(x)
        y=self.layer(x)
        y=y+residual
        return y

class ResidualBlock2(nn.Module):
    def __init__(self, block_index=1,in_channel=32, num_kernels=16, subsample_length=1, params_dict=None):
        super(ResidualBlock2, self).__init__()
        self.block_index=block_index
        self.num_kernels=num_kernels
        self.pooling=nn.MaxPool1d(kernel_size=subsample_length,stride=subsample_length,padding=0, return_indices=False)
        self.layer=nn.Sequential(
             BNReLU(in_channel, dropout=0.0),
             ConvLayer(in_channel, num_kernels, params_dict["conv_filter_length"], stride=subsample_length),
             BNReLU(num_kernels, dropout=params_dict["conv_dropout"]),
             ConvLayer(num_kernels, num_kernels, params_dict["conv_filter_length"], stride=1)
        )
    def zeropad(self, x):
        #x: (bs, num_channel, L)
        temp=torch.zeros_like(x,device=x.device)
        x=torch.concat([x,temp], dim=1) # x:(bs, 2*num_channel, L)
        return x
    def forward(self,x):
        residual=self.pooling(x)
        zero_pad=(self.block_index%4)==0 and self.block_index>0
        if zero_pad:
            # because the maxpooling can reduce the size ot the half, but it can't increase the number of channels
            # so once the ouput from convolution layer has the double number of input channels, use zero to fill.
            residual=self.zeropad(residual)

        y=self.layer(x)
        y=residual+y

        return y

class ConvNet(nn.Module):
    def __init__(self, in_channel=1,length_list=None, start_in=32, params_dict=None):
        super(ConvNet, self).__init__()

        layer_list=[ConvLayer(in_channel,start_in,kernel_size=params_dict["conv_filter_length"], stride=1),
                    BNReLU(start_in, dropout=0.0)]
        layer_list.append(ResidualBlock1(num_kernels=start_in, subsample_length=length_list[0], params_dict=params_dict))
        num_channels=start_in
        for index, length in enumerate(length_list[1:],start=1):
            new_num_kernels=self.computer_num_kernels(index, start_in)
            layer_list.append(ResidualBlock2(index,num_channels,new_num_kernels,length, params_dict))
            num_channels=new_num_kernels
        layer_list.append(BNReLU(num_channels, dropout=0.0))
        self.layer_list=nn.ModuleList(layer_list)

    def computer_num_kernels(self,index,num_kernels_start):
        num_kernels=2**int(index/4)*num_kernels_start # according to the paper, if block index satisfies the condition, the number of kernels will be changed.
        return num_kernels

    def forward(self,x):
        for layer in self.layer_list:
            x=layer(x)
        return x

class Net(nn.Module):
    def __init__(self, params_dict=None):
        super(Net, self).__init__()
        self.num_classes=params_dict["num_categories"]
        self.step=params_dict["step"]
        self.backbone=nn.Sequential(ConvNet(in_channel=1, length_list=params_dict["conv_subsample_lengths"],
                                            start_in=params_dict["conv_num_filters_start"], params_dict=params_dict))
        self.classifier=nn.Linear(in_features=self.step, out_features=self.num_classes, bias=False)
        # if use CrossEntropy as the loss function, it contains a softmax function. so it is not necessary to add a softmax layer.
        self.initiate_weight()# use kaiming he's method to initialize the learnable parameters of the neural network

    def initiate_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,x):
        y=self.backbone(x) #(bs, 256, 35)
        y=y.permute(0,2,1)#(bs, 35, 256)-->(bs, 35, 4)
        score=self.classifier(y)

        return score
# if you want to import local directory as a module, you should add "sys.append(str local_dirctory)
def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);

if __name__ == "__main__":
    params_dict = {"conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "conv_filter_length": 16,
        "conv_num_filters_start": 32,
        "conv_init": "he_normal",
        "conv_dropout": 0.2,
        "conv_num_skip": 2,
        "conv_increase_channels_at": 4,
        "step":256,
        "num_categories": 4
    }

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu;
    args.gpu_id = int(args.gpu)
    x_tensor=torch.rand(size=(2,1, 8960), dtype=torch.float32).to(device)
    print("input:",x_tensor.shape)
    model=Net(params_dict).to(device)
    y=model(x_tensor)
    print("output:",y.shape)

    print("Finish!");
