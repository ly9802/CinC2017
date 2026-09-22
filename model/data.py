# -- coding:utf-8 --
# Time:17 9月 2026 10:20
# Author YI LIAO(Steven Leo)
# File:data.py
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
import tqdm
from PIL import Image
from tqdm import tqdm
import collections
import json
import scipy.stats as sst
import scipy.io as sio
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

# if you want to import local directory as a module, you should add "sys.append(str local_dirctory)
def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);
def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))
def pad(x, val=0, dtype=np.float32):
    # for dynamic length of 1D vector, output the fixed length by filling "val" value (val=0, default)
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def load_ecg(record):
    # str record: path
    STEP=256
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":# record is a path, to see whether the suffix is "mat" or not
        ecg = sio.loadmat(record)['val'].squeeze()
    else:
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    trunc_ecg=ecg[:trunc_samp] # make sure length of output can be divided by STEP
    return trunc_ecg

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data_list = [json.loads(l) for l in fid]#-->data:list[dict, dict]
    labels_list = []; ecgs_list = []
    for dict_item in tqdm(data_list):
        labels_list.append(dict_item['labels'])
        temp=dict_item['ecg']# str path
        trunc_ecg=load_ecg(temp)
        ecgs_list.append(trunc_ecg)
        #ecgs_list.append(load_ecg(dict_item['ecg']))
    return ecgs_list, labels_list

def data_generator(batch_size, preproc, x, y):
    # x: list, y: list
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size] for i in range(0, end, batch_size)]
    random.shuffle(batches)

    for batch in batches:
        x, y = zip(*batch) # x:tuple(ndarray .... )
        yield preproc.process(x, y)

class Preproc:
    def __init__(self, ecg, labels):
        # ecg: list, labels:list
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)  # x :tuple(ndarray, ...) len(tuple)=batchsize
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        x = torch.from_numpy(x).float()
        x = x.permute(0, 2, 1)
        return x

    def process_y(self, y):
        temp = [[self.class_to_int[c] for c in s] for s in y]
        y = pad(temp, val=3, dtype=np.int32)  # 3:noise use 3 to fill
        y = torch.from_numpy(y).long()
        return y

class ECGSet(Dataset):
    def __init__(self,json_path, params_dict):
        self.data_root=json_path;
        self.step=params_dict['step']
        self.ecg_list,self.label_list=self.load_dataset(self.data_root)
        self.mean, self.std = self.compute_mean_std(self.ecg_list)
        self.classes = sorted(set(l for label in self.label_list for l in label))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def compute_mean_std(self,x):
        x = np.hstack(x)
        return (np.mean(x).astype(np.float32),
                np.std(x).astype(np.float32))

    def load_ecg(self, record):
        # str record: path
        STEP=self.step #STEP = 256
        if os.path.splitext(record)[1] == ".npy":
            ecg = np.load(record)
        elif os.path.splitext(record)[1] == ".mat":  # record is a path, to see whether the suffix is "mat" or not
            ecg = sio.loadmat(record)['val'].squeeze()
        else:
            with open(record, 'r') as fid:
                ecg = np.fromfile(fid, dtype=np.int16)

        trunc_samp = STEP * int(len(ecg) / STEP)
        trunc_ecg = ecg[:trunc_samp]  # make sure length of output can be divided by STEP -->ndarray
        return trunc_ecg
    def load_dataset(self,data_json):
        with open(data_json, 'r') as fid:
            data_list = [json.loads(l) for l in fid]  # -->data:list[dict, dict]
        labels_list = [];
        ecgs_list = []
        for dict_item in tqdm(data_list):
            labels_list.append(dict_item['labels'])
            ecgs_list.append(self.load_ecg(dict_item['ecg']))
        return ecgs_list, labels_list

    def __len__(self):
        return len(self.label_list)
    def __getitem__(self,idx):
        ecg_ndarray=self.ecg_list[idx]
        ecg_ndarray = (ecg_ndarray - self.mean) / self.std

        label_=self.label_list[idx]# each element is string not an integer
        label_ndarray=np.array([self.class_to_int[c] for c in label_])#string-->integer

        return ecg_ndarray, label_ndarray

def pad_series(sample_list):
    #sample_list:[(data1, label1),(data2,label2),(...),(...)] len(sample_list)=batchsize
    data_list, label_list = zip(*sample_list)
    x = pad(data_list)  # x :tuple(ndarray, ...) len(tuple)=batchsize-->2D ndarray
    x = x[:, :, None]
    x = torch.from_numpy(x).float()
    x = x.permute(0, 2, 1)

    y=pad(label_list,val=3, dtype=np.int64)#
    y = torch.from_numpy(y).long()
    return x,y

def data_generate(data_path, params_dict, batch_size, drop_last=True):
    ecg_set = ECGSet(data_path, params_dict)
    dataloader=DataLoader(ecg_set, batch_size=batch_size, shuffle=True, num_workers=1,collate_fn=pad_series,drop_last=drop_last)
    return ecg_set, dataloader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "../config/dev.json"
    ecg_list, label_list = load_dataset(data_path)  # tuple(list, list)
    preproc = Preproc(ecg_list, label_list)
    gen = data_generator(32, preproc, ecg_list, label_list)

    for x, y in gen:
        # x: (32=batch_size, 1=in_channel, 8960=trunc_ecg_length)
        # y: (32=batch_size, 35=num_label, 4=num_classes) 35=8960/STEP(256)
        print(x.shape, y.shape) #y(32,35)

        break

    print("Finish!");
