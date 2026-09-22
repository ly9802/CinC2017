# -- coding:utf-8 --
# Time:20 9月 2026 12:48
# Author YI LIAO(Steven Leo)
# File:test.py
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import math
import random
import cv2
import time
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
import glob
import json
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
from main import get_args, eval
# if you want to import local directory as a module, you should add "sys.append(str local_dirctory)
def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);

def load_pretrained_weights(model, pretrained_weights, checkpoint_key=None):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Pretrained weights is not found at {}".format(pretrained_weights))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu;
    params_dict = json.load(open(args.config_file, 'r'))

    test_set, test_loader = data_generate(args.eval_set_path, params_dict, args.batch_size, drop_last=False)

    model_path=os.path.join(args.save_dir, args.data_set, "best_model.pth")
    model = Net(params_dict)
    load_pretrained_weights(model, model_path)
    model.to(args.device)
    test_acc=eval(model, args, params_dict,test_loader)
    print("The testing accuracy:{}%".format(test_acc))

    print("Finish!");
