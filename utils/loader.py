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

""" Initializing Dataset class """
class NYU_V2(Dataset):
    def __init__(self, trn_tst=0, transform=None):
        data = h5py.File(config.dir_data + 'NYU/nyu_depth_v2_labeled.mat', mode='r')

        if trn_tst == 0:
            #trainloader
            # self.images = data["images"][0:1400]
            # self.depths  = data["depths"][0:1400]
            self.images = data["images"][:10]
            self.depths  = data["depths"][:10]
        else:
            #testloader
            # self.images = data["images"][1401:]
            # self.depths = data["depths"][1401:]
            self.images = data["images"][10:20]
            self.depths = data["depths"][10:20]

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

        return sample.float(), s_depth.float()



class render_wave(Dataset):
    def __init__(self, trn_tst=0, return_rec=False):
        datapath = config.dir_data + 'render_wave2-pose_600/'
        self.gtfile = datapath + 'gt_512/{:05d}.bmp'
        self.recfile = datapath + 'rec_512/{:05d}.bmp'
        self.shadefile = datapath + 'shade_512/{:05d}.png'
        self.projfile = datapath + 'proj_512/{:05d}.png'
        self.sparse = datapath + 'sparse_512/{:05d}.npy'
        self.data_len = 600
        self.train_rate = 0.7
        self.return_rec = return_rec

        if trn_tst == 0:
            self.start = 0
            self.end = int(self.data_len * self.train_rate)
        else:
            self.start = int(self.data_len * self.train_rate)
            self.end = self.data_len

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx += self.start

        shade = cv2.imread(self.shadefile.format(idx), 1) / 255
        gt_img = cv2.imread(self.gtfile.format(idx), -1)
        gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)
        sp = np.load(self.sparse.format(idx))

        shade = torch.from_numpy(np.transpose(shade, (0, 1, 2)))
        gt = torch.from_numpy(np.transpose(gt, (0, 1)))
        sp = torch.from_numpy(sp)

        if self.return_rec:
            rec_img = cv2.imread(self.recfile.format(idx), -1)
            rec = torch.from_numpy(depth_tools.unpack_bmp_bgra_to_float(rec_img))
            return shade.float(), gt.float(), rec.float()
        else:
            return shade.float(), gt.float(), sp.float()

class board_data(Dataset):
    def __init__(self):
        datapath = config.dir_data + 'board/'
        self.gtfile = datapath + 'clip_gt/{:05d}.bmp'
        self.recfile = datapath + 'clip_rec/{:05d}.bmp'
        self.shadefile = datapath + 'clip_shade/{:05d}.png'
        self.projfile = datapath + 'clip_proj/{:05d}.png'
        # self.sparse = datapath + 'sparse_512/{:05d}.npy'
        self.idxs = list(range(16)) + list(range(40, 56))
        self.data_len = len(self.idxs)
        # self.train_rate = 0.7
        # self.return_rec = return_rec

        # if trn_tst == 0:
        #     self.start = 0
        #     self.end = int(self.data_len * self.train_rate)
        # else:
        #     self.start = int(self.data_len * self.train_rate)
        #     self.end = self.data_len

    def __len__(self):
        # return self.end - self.start
        return self.data_len

    def __getitem__(self, idx):
        idx = self.idxs[idx]

        shade = cv2.imread(self.shadefile.format(idx), 1) / 255
        gt_img = cv2.imread(self.gtfile.format(idx), -1)
        gt = tools.unpack_bmp_bgra_to_float(gt_img)
        # sp = np.load(self.sparse.format(idx))

        shade = torch.from_numpy(np.transpose(shade, (0, 1, 2)))
        gt = torch.from_numpy(np.transpose(gt, (0, 1)))
        # sp = torch.from_numpy(sp)

        return shade.float(), gt.float()#, sp.float()

class real_data(Dataset):
    def __init__(self):
        datapath = config.dir_data + 'real/'
        self.gtfile = datapath + 'clip_gt/{:05d}.bmp'
        self.recfile = datapath + 'clip_rec/{:05d}.bmp'
        self.shadefile = datapath + 'clip_shade/{:05d}.png'
        self.projfile = datapath + 'clip_proj/{:05d}.png'
        self.data_len = 19

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        shade = cv2.imread(self.shadefile.format(idx), 1) / 255
        gt_img = cv2.imread(self.gtfile.format(idx), -1)
        gt = tools.unpack_bmp_bgra_to_float(gt_img)

        shade = torch.from_numpy(np.transpose(shade, (0, 1, 2)))
        gt = torch.from_numpy(np.transpose(gt, (0, 1)))

        return shade.float(), gt.float()