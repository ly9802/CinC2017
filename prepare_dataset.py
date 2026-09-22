# -- coding:utf-8 --
# Time:20 9月 2026 13:04
# Author YI LIAO(Steven Leo)
# File:prepare_dataset.py
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import json
import random
import argparse
import numpy as np
import scipy.io as sio
import tqdm
# if you want to import local directory as a module, you should add "sys.append(str local_dirctory)
def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);

def load_ecg_mat(ecg_file):

    return sio.loadmat(ecg_file)['val'].squeeze()

def load_all(data_path, reference_path,STEP=256):
    label_file = os.path.join(reference_path)
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]
    # records: list[[str,str],[],..]
    dataset_list = []
    for record, label in tqdm.tqdm(records):
        # record: str, label: str
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file) # mat_file_path
        ecg = load_ecg_mat(ecg_file)#mat-->ndarray(9000,) [int, int,...]
        length_ecg=ecg.shape[0]
        #num_labels = length_ecg // STEP # ecg
        num_labels=int(length_ecg/STEP)
        dataset_list.append((ecg_file, [label]*num_labels))
    return dataset_list

def split(dataset_list, dev_frac):
    #float dev_frac
    dev_cut = int(dev_frac * len(dataset_list))
    random.shuffle(dataset_list)
    dev = dataset_list[:dev_cut] #10% for evlation
    train = dataset_list[dev_cut:] # 90% for training
    return train, dev

def make_json(save_path, dataset_list):
    with open(save_path, 'w') as fid:
        for d in dataset_list:
            datum = {'ecg' : d[0], # d[0]: str ecg_path
                     'labels' : d[1]} #d[1]: str label
            json.dump(datum, fid)
            fid.write('\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="CinC2017", type=str, help="Dataset name")
    parser.add_argument("--dataset-path", default="../../../dataset/CinC/training2017/", type=str)
    parser.add_argument("--reference-file",default="../../../dataset/CinC/REFERENCE-v3.csv",type=str)
    parser.add_argument("--config-file", default="./config/modelconfig.json")
    parser.add_argument("--ratio",default=0.1, type=float, help="how many samples are used for evaluation")
    parser.add_argument("--json-save-dir", default="./config/", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    random.seed(2018)
    args = get_args()
    params_dict = json.load(open(args.config_file, 'r'))
    dev_frac = args.ratio
    data_path = args.dataset_path
    reference_path=args.reference_file

    dataset_list = load_all(data_path,reference_path,STEP=params_dict["step"])  # -->[str ecg_path, str label]
    train_list, dev_list = split(dataset_list, dev_frac)
    # train: list; dev: list
    check_path(args.json_save_dir)
    make_json(os.path.join(args.json_save_dir,"train.json"), train_list)
    make_json(os.path.join(args.json_save_dir, "dev.json"), dev_list)


    print("Finish!");
