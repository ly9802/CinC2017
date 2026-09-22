# -- coding:utf-8 --
# Time:20 9月 2026 23:12
# Author YI LIAO(Steven Leo)
# File:statistic.py
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import math
import random
import collections
import json
import time
import argparse
import numpy as np
import pandas as pd
import scipy.stats as sst
import scipy.io as sio
import sklearn.metrics as skm
import h5py
from tqdm import tqdm
import tqdm
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


from model.network import Net
from model.data import data_generate
from main import get_args, eval

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
def load_pretrained_weights(model, pretrained_weights, checkpoint_key=None):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        #state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Pretrained weights is not found at {}".format(pretrained_weights))
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
    for dict_item in tqdm.tqdm(data_list):
        labels_list.append(dict_item['labels'])
        ecgs_list.append(load_ecg(dict_item['ecg']))
    return ecgs_list, labels_list
def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded
def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)
class Preproc:
    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):

        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)

        return y
def compute_prior(train_path, preproc):
    with open(train_path, 'r') as fid:
        train_labels = [json.loads(l)['labels'] for l in fid]
    counts = collections.Counter(preproc.class_to_int[l[0]] for l in train_labels)
    counts = sorted(counts.most_common(), key=lambda x: x[0])#->list(tuple, tuple, tuple, tuple)
    temp=zip(*counts) #zip(tuple_A, tuple_B, tuple_C,tuple_D)--iterator
    #temp is iterator
    counts=list(temp)[1]
    #counts = zip(*counts)[1]
    smooth = 500
    counts = np.array(counts)[None, None, :]
    total = np.sum(counts) + counts.shape[1]
    prior = (counts + smooth) / float(total)
    print("prior:",prior) # prior: [[[0.15448743 0.66301941 0.34596848 0.09691286]]]
    return prior

def statis(model, args, params_dict, data_loader):
    model.eval()
    total_eval = 0
    correct_eval = 0.0
    with torch.no_grad():
        prob_list=[]
        label_list=[]
        for no_batch, (data, label) in enumerate(data_loader):
            if args.use_cuda:
                data, label = data.to(args.device), label.to(args.device)

            score= model(data)#(bs, 35, 4)
            prob=F.softmax(score,dim=-1)

            prob_list.append(prob.cpu().numpy())
            label_list.append(label.cpu().numpy())

            prob_ = prob.contiguous().view(-1, 4)
            total_eval += prob_.size(dim=0)

            label_ = label.contiguous().view(-1)

            predict = prob_.argmax(dim=-1)

            correct_eval += (predict == label_).sum().item()

        test_acc = 100 * (correct_eval / total_eval)
        print("Test Accuracy: {:.2f}%".format(test_acc))
    return prob_list,label_list
  

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args=get_args();
    eval_path = args.eval_set_path
    train_path = args.training_set_path
    params_dict = json.load(open(args.config_file, 'r'))

    ecg_list, label_list = load_dataset(eval_path)
    preproc = Preproc(ecg_list, label_list)
    prior=compute_prior(train_path, preproc)

    #model_path = os.path.join(args.save_dir, args.data_set, "best_model.pth")
    model_path = os.path.join("./checkpoints/", "best_model.pth")
    model = Net(params_dict)
    load_pretrained_weights(model, model_path)
    model.to(args.device)

    test_set, test_loader = data_generate(eval_path, params_dict, 1, drop_last=False)

    prob_list,label_list=statis(model, args, params_dict, test_loader)
    #prob_list:list[ndarray:(1,35,4)]
    #label_list: list[ndarray:(1,35)]
    
    pred_list = []
    ground_truth_list = []
    for p, g in zip(prob_list, label_list):
        pred_list.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0]) #np.argmax(p / prior, axis=2).squeeze() (1,35)-->ndarray(35,)
        ground_truth_list.append(sst.mode(g.squeeze())[0][0])
    
    report = skm.classification_report(ground_truth_list, pred_list, target_names=preproc.classes, digits=3)
    scores = skm.precision_recall_fscore_support(ground_truth_list, pred_list, average=None)
    
    print(report)
    print("CINC Average {:3f}".format(np.mean(scores[2][:3])))


    print("Finish!");
