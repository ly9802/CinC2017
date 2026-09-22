# -- coding:utf-8 --
# Time:20 9月 2026 10:47
# Author YI LIAO(Steven Leo)
# File:main.py
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
import json
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

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from model.network import Net
from model.data import data_generate

# if you want to import local directory as a module, you should add "sys.append(str local_dirctory)
class EarlyStopping:
    def __init__(self, patience=2, verbose=False, delta=0, device=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.device=device

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Training loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.cpu().state_dict(), os.path.join(path, 'best_model.pth'))
        model.to(self.device)
        self.val_loss_min = val_loss

def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);

def generate_optimizer(model, optimizer_name="Adam", lr=0.001):
    from torch.optim import SGD, RMSprop, Adagrad, Adam, AdamW
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer_name == "Adagrad":
        optimizer = Adagrad(model.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay=1e-5)
    elif optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    else:
        raise NotImplementedError

    return optimizer

def generate_scheduler(optimizer, schedulr_name="ReduceLROnPlateau"):
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, MultiStepLR, \
        CyclicLR, LambdaLR, CosineAnnealingWarmRestarts
    if schedulr_name == "StepLR":
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    elif schedulr_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer)
    elif schedulr_name == "ExponentialLR":
        scheduler = ExponentialLR(optimizer)
    elif schedulr_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1,patience=2,min_lr=0.001*0.001)
    elif schedulr_name == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    elif schedulr_name == "CyclicLR":
        scheduler = CyclicLR(optimizer)
    elif schedulr_name == "LambdaLR":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 ** epoch)
    elif schedulr_name == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer)
    else:
        raise NotImplementedError
    return scheduler

def save_model(model, save_dir):
    import os;
    save_path = os.path.join(save_dir, "best_model.pth");
    torch.save(model.cpu().state_dict(), save_path);
    print("The model is saved at {}".format(save_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return save_path

def eval(model, args, params_dict, data_loader):
    num_batches=len(data_loader)
    print("How many batches in testing data:{}".format(num_batches))
    model.eval()
    total_eval = 0
    correct_eval = 0.0

    with torch.no_grad():
        test_loss = 0.0
        for no_batch, (data, label) in enumerate(data_loader):
            if args.use_cuda:
                data, label = data.to(args.device), label.to(args.device)

            score= model(data)#(bs, 35, 4)
            prob=F.softmax(score,dim=-1)
            prob = prob.contiguous().view(-1, 4)
            y = label.contiguous().view(-1)

            predict = prob.argmax(dim=-1)
            total_eval += prob.size(dim=0)
            correct_eval += (predict == y).sum().item()

        test_acc = 100 * (correct_eval / total_eval)
    return test_acc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="CinC2017", type=str, help="Dataset name")
    parser.add_argument("--training-set-path", default="./config/train.json", type=str)
    parser.add_argument("--eval-set-path",default="./config/dev.json",type=str)
    parser.add_argument("--config-file",default="./config/modelconfig.json")
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"])
    parser.add_argument("--pretrained", default="../../../PretrainedModels/", type=str);
    parser.add_argument("--save-dir", default="../../../ExperimentResults/", type=str);
    parser.add_argument("--num-epoches", default=20,type=int)
    parser.add_argument("--batch-size", default=32,type=int) #128 in paper
    parser.add_argument("--init-lr",default=0.001, type=float)
    parser.add_argument("--optimizer-name",default="Adam",type=str, choices=["SGD","Adam","RMSprop","Adagrad","AdamW"])
    parser.add_argument("--scheduler-name", default="ReduceLROnPlateau", type=str,
                        choices=["StepLR","CosineAnnealingLR","ExponentialLR","ReduceLROnPlateau","MultiStepLR","CyclicLR","LambdaLR","CosineAnnealingWarmRestarts"])
    parser.add_argument("--patience", default=2, type=int)
    parser.add_argument("--eval", default=True, action='store_true')
    parser.add_argument("--device", default=None, type=str);
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu;
    args.gpu_id = int(args.gpu)

    params_dict = json.load(open(args.config_file, 'r'))

    batch_size = args.batch_size
    num_epoches = args.num_epoches
    data_path = args.training_set_path
    save_dir = os.path.join(args.save_dir, args.data_set)
    check_path(save_dir)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, device=args.device)

    #prepare dataset
    ecg_set,data_loader=data_generate(data_path, params_dict, batch_size)
    test_set, test_loader = data_generate(args.eval_set_path, params_dict, batch_size, drop_last=False)
    #{"A":0,"N":1,"O":2,"~":3}
    # A=AF , N=Normal, O=other rhythm, ~=Nosiy
    num_batches = len(ecg_set)//batch_size

    print("How many epoch:{} during the training.".format(num_epoches))
    print("How many batches:{} per epoch during the training.".format(num_batches))
    #prepare model
    model = Net(params_dict).to(device)
    # prepare optimizer, scheduler, and loss function.
    optimizer = generate_optimizer(model, optimizer_name=args.optimizer_name, lr=args.init_lr)
    scheduler = generate_scheduler(optimizer, schedulr_name=args.scheduler_name)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    # begin training
    for epoch in range(num_epoches):
        model.train()
        total = 0
        correct = 0
        train_loss_list = []
        print("-----------Epoch:{}-------------------".format(epoch+1))

        for batch_idx, (ecg_data, target) in enumerate(data_loader):
            x = ecg_data.to(device)
            y = target.to(device)
            score = model(x)
            score = score.contiguous().view(-1, 4)
            y = y.contiguous().view(-1)

            loss = criterion(score, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += score.size(dim=0)
            pred = score.argmax(dim=-1)
            correct += (pred == y).sum().item()

            print("Trainding Batch:{}|{}\t Loss:{:.6f}".format(batch_idx + 1, num_batches, loss.item()))
            train_loss_list.append(loss.cpu().item())

        train_acc = 100 * (correct / total)
        train_loss = np.average(train_loss_list)
        scheduler.step(train_loss)

        print("Epoch:{}|{}, Training accuray :{:6f}%, next learning rate:{:6f}".format(epoch + 1, num_epoches, train_acc,
                                                                                     optimizer.param_groups[0]['lr']))

        if args.eval: #during training, whether to do test
            test_acc=eval(model, args,params_dict, test_loader)
            print("The testing accuracy: {}%".format(test_acc))

        early_stopping(train_loss, model, save_dir) # compare training loss, save the model when the training loss is the smallest.

        if early_stopping.early_stop: # according to the paper, once the training loss doesn't decrease in two consecutive epoches, stop!
            print("Early stopping")
            break


    print("Finish!");
