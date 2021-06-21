import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import multiprocessing
import matplotlib.pyplot as plt
import h5py
import time
import csv
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.network import D3
from utils.sparse import NN_fill, generate_mask
from utils import depth_tools
from scripts import config

''' Synthetic data '''
class render_wave(Dataset):
    def __init__(self, data_size=600, trn_tst='train', return_rec=False):
        datapath = config.dir_data + 'synthetic/'
        self.gtfile = datapath + 'gt_512/{:05d}.bmp'
        self.recfile = datapath + 'rec_512/{:05d}.bmp'
        self.shadefile = datapath + 'shade_512/{:05d}.png'
        self.projfile = datapath + 'proj_512/{:05d}.png'
        # self.spfile = datapath + 'sparse_512/{:05d}.npy'
        self.spfile = datapath + 'sparse_512_sample-8/{:05d}.npy'
        self.data_len = data_size
        self.train_rate = 0.7
        self.return_rec = return_rec

        if trn_tst == 'train':
            self.start = 0
            self.end = int(self.data_len * self.train_rate)
        elif trn_tst == 'test':
            self.start = int(self.data_len * self.train_rate)
            self.end = self.data_len

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx += self.start

        shade = cv2.imread(self.shadefile.format(idx), 1) / 255.
        gt_img = cv2.imread(self.gtfile.format(idx), -1)
        gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)

        shade = torch.from_numpy(shade)
        gt = torch.from_numpy(gt)

        if self.return_rec:
            rec_img = cv2.imread(self.recfile.format(idx), -1)
            rec = depth_tools.unpack_bmp_bgra_to_float(rec_img)
            proj = cv2.imread(self.projfile.format(idx), 0) / 225.
            rec = np.dstack([rec, proj])
            rec = torch.from_numpy(rec)
            rec = rec.permute(2, 0, 1)
            return shade.float(), gt.float(), rec.float()
        else:
            sp = np.load(self.spfile.format(idx))
            sp = torch.from_numpy(sp)
            return shade.float(), gt.float(), sp.float()

''' Cardboard data '''
class board_data(Dataset):
    def __init__(self, trn_tst='test', return_rec=False, res=512, patch_res=None):
        datapath = config.dir_data + 'cardboard/'
        self.gtfile = datapath + 'gt_{}/{:05d}.bmp'
        self.recfile = datapath + 'rec_{}/{:05d}.bmp'
        self.shadefile = datapath + 'shade_{}/{:05d}.png'
        self.projfile = datapath + 'proj_{}/{:05d}.png'
        self.spfile = datapath + 'sparse_{}/{:05d}.npy'
        # self.spfile = datapath + 'sparse_{}_sample-8/{:05d}.npy'
        self.return_rec = return_rec
        self.res = res
        self.idxs = list(range(16)) + list(range(40, 56))
        self.data_len = len(self.idxs)
        self.is_patch = True if patch_res is not None else False
        if self.is_patch:
            self.patch_res = patch_res
            self.patch_num = res // patch_res

        if trn_tst == 'train':
            self.start = 0
            self.end = 20
        elif trn_tst == 'test':
            self.start = 20
            self.end = 32

    def __len__(self):
        if self.is_patch:
            return (self.end - self.start) * self.patch_num**2
        else:
            return self.end - self.start

    def __getitem__(self, idx):
        if self.is_patch:
            p = self.patch_num
            idx_patch = idx % p**2
            idx = int(idx / p**2)
        idx = self.idxs[self.start + idx]
        res = self.res

        gt_img = cv2.imread(self.gtfile.format(res, idx), -1)
        shade = cv2.imread(self.shadefile.format(res, idx), 1) / 255.
        proj = cv2.imread(self.projfile.format(res, idx), 0) / 225.

        gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)

        gt = torch.from_numpy(gt)
        shade = torch.from_numpy(shade)

        if self.return_rec:
            rec_img = cv2.imread(self.recfile.format(res, idx), -1)
            proj = cv2.imread(self.projfile.format(res, idx), 0) / 225.

            rec = depth_tools.unpack_bmp_bgra_to_float(rec_img)
            
            sp = np.dstack([rec, proj])
            sp = torch.from_numpy(sp)
            sp = sp.permute(2, 0, 1)
        else:
            sp = np.load(self.spfile.format(res, idx))
            sp = torch.from_numpy(sp)

        if self.is_patch:
            h = int(idx_patch / p)
            w = idx_patch % p
            r = self.patch_res
            gt = gt[h*r:(h+1)*r, w*r:(w+1)*r]
            sp = sp[:, h*r:(h+1)*r, w*r:(w+1)*r]
            shade = shade[h*r:(h+1)*r, w*r:(w+1)*r, :]

        return shade.float(), gt.float(), sp.float()

''' Real data '''
class real_data(Dataset):
    def __init__(self, trn_tst='test', return_rec=False, res=512, patch_res=None):
        datapath = config.dir_data + 'reals/'
        self.gtfile = datapath + 'gt_{}/{:05d}.bmp'
        self.recfile = datapath + 'rec_{}/{:05d}.bmp'
        self.shadefile = datapath + 'shade_{}/{:05d}.png'
        self.projfile = datapath + 'proj_{}/{:05d}.png'
        self.spfile = datapath + 'sparse_{}/{:05d}.npy'
        # self.spfile = datapath + 'sparse_{}_sample-8/{:05d}.npy'
        self.return_rec = return_rec
        self.res = res
        self.data_len = 19
        self.is_patch = True if patch_res is not None else False
        if self.is_patch:
            self.patch_res = patch_res
            self.patch_num = res // patch_res

        if trn_tst == 'train':
            self.idxs = [0, 1, 6, 7, 8, 9, 16]
        elif trn_tst == 'test':
            self.idxs = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 17, 18]

    def __len__(self):
        if self.is_patch:
            return len(self.idxs) * self.patch_num**2
        else:
            return len(self.idxs)

    def __getitem__(self, idx):
        if self.is_patch:
            p = self.patch_num
            idx_patch = idx % p**2
            idx = int(idx / p**2)
        idx = self.idxs[idx]
        res = self.res

        gt_img = cv2.imread(self.gtfile.format(res, idx), -1)
        shade = cv2.imread(self.shadefile.format(res, idx), 1) / 255.
        proj = cv2.imread(self.projfile.format(res, idx), 0) / 225.

        gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)

        gt = torch.from_numpy(gt)
        shade = torch.from_numpy(shade)

        if self.return_rec:
            rec_img = cv2.imread(self.recfile.format(res, idx), -1)
            proj = cv2.imread(self.projfile.format(res, idx), 0) / 225.

            rec = depth_tools.unpack_bmp_bgra_to_float(rec_img)

            sp = np.dstack([rec, proj])
            sp = torch.from_numpy(sp)
            sp = sp.permute(2, 0, 1)
        else:
            sp = np.load(self.spfile.format(res, idx))
            sp = torch.from_numpy(sp)

        if self.is_patch:
            h = int(idx_patch / p)
            w = idx_patch % p
            r = self.patch_res
            gt = gt[h*r:(h+1)*r, w*r:(w+1)*r]
            sp = sp[:, h*r:(h+1)*r, w*r:(w+1)*r]
            shade = shade[h*r:(h+1)*r, w*r:(w+1)*r, :]

        return shade.float(), gt.float(), sp.float()

''' NYU '''
class NYU_V2(Dataset):
    def __init__(self, trn_tst='train', transform=None):
        data = h5py.File(config.dir_data + 'NYU/nyu_depth_v2_labeled.mat', mode='r')
        self.spfile = config.dir_data + 'NYU/sparse/{:05d}.npy'

        if trn_tst == 'train':
            #train
            start = 0
            end = 420
        elif trn_tst == 'val':
            #validation
            start = 420
            end = 600
        elif trn_tst == 'test':
            #test
            start = 600
            end = 650
        elif trn_tst == 'all':
            start = 0
            end = 1449

        self.images = data["images"][start:end]
        self.depths  = data["depths"][start:end]
        self.idxs = range(start,end)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.images[idx, :]
        s_depth = self.depths[idx, :]
        sample = torch.from_numpy(np.transpose(sample, (2, 1, 0)))
        s_depth=  torch.from_numpy(np.transpose(s_depth, (1, 0)))

        sp = np.load(self.spfile.format(self.idxs[idx]))
        sp = torch.from_numpy(sp)

        return sample.float() / 255., s_depth.float(), sp.float()