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
from tqdm import tqdm
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.network import D3
from utils.sparse import NN_fill, generate_mask
from utils.loader import render_wave, board_data, real_data
from utils import depth_tools, evaluate
import config

''' ARG '''
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='model name to use training and test')
parser.add_argument('--pred', help='predict folder name')
args = parser.parse_args()


''' MODEL NAME '''
model_name = 'wave2_100'
model_name = 'overfit'

pred_name = 'pred_train'
# pred_name = 'pred_real'

''' OPTIOIN '''
data_size = 12
data_size = 3

''' SETTING '''
rgb_threshold = 0
gt_threshold = 0.1
res = 512
# 24x24 downsampling
# mask = generate_mask(24, 24, 480, 640)
# mask = generate_mask(24, 24, res, res)


if args.name is not None:
    model_name = args.name
if args.pred is not None:
    pred_name = args.pred

dir_model = config.dir_models + model_name + '/'
dir_pred = dir_model + pred_name + '/'
os.makedirs(dir_pred, exist_ok=True)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


""" Creating Train loaders """
train_set = render_wave(data_size=data_size, trn_tst=0, return_rec=True)
# test_set = render_wave(data_size=data_size, trn_tst=1)

# test_set = board_data()
# test_set = real_data()

print(f'Number of training examples: {len(train_set)}')
# print(f'Number of testing examples: {len(test_set)}')

testloader = DataLoader(train_set, batch_size=1, shuffle=False)
# testloader = DataLoader(test_set, batch_size=1, shuffle=False)
print('Loader built')


""" Testing and visualising data """
model = D3()
model.load_state_dict(torch.load(dir_model + 'saved_model.pt'))
# testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

print('Test...')
with torch.no_grad():
    model.eval()
    model.to(device)

    # for idx, (img, depth) in enumerate(testloader):
    #     NN = []
    #     #concat x with spatial data
    #     for j in range(img.shape[0]):
    #         sp = NN_fill(img[j].numpy(), depth[j].numpy(), mask)
    #         NN.append(sp)
    #     NN = torch.tensor(NN)

    for idx, (img, depth, sp, rec) in enumerate(testloader):
        NN = sp

        # NN[:, 1, :, :] = 1 / NN[:, 1, :, :]
        # NN[:, 1, :, :] /= NN[:, 1, :, :].max()
        # NN = torch.cat((rec, rec), 1)


        img = img.permute(0, 3, 1, 2)
        img = img.to(device) 
        depth = depth.to(device)
        NN = NN.to(device)
        # print(NN.shape)
        # print(img.shape)
        # print(depth.shape)

        fx = model(img, NN)

        img = img[0].permute(1, 2, 0).cpu().numpy()
        depth = depth[0].cpu().numpy()
        fx = fx[0][0].cpu().numpy()
        NN = NN.cpu().numpy()

        s1 = NN[:, 0, :, :]
        s2 = NN[:, 1, :, :]

        pred = fx
        sparse = s1[0]
        # pred += s1[0]

        ''' Normalize prediction '''
        mask_rgb = img[:, :, 0] > rgb_threshold
        mask_depth = depth > gt_threshold
        mask = mask_rgb * mask_depth
        mask = mask.astype(np.float32)
        # rec = rec[0][0].cpu().numpy()

        # gt = (depth - rec) * mask
        gt = (depth - sparse) * mask
        length = np.sum(mask)
        mean_gt = np.sum(gt) / length
        max_gt = np.max(np.abs(gt - mean_gt))
        max_pred = np.max(np.abs(pred))
        # diff = (pred / max_pred) * max_gt + mean_gt
        diff = pred * max_gt + mean_gt
        pred = sparse + diff

        pred, mask = evaluate.norm_diff(pred, depth, sparse, mask)
        # pred, mask = evaluate.norm_diff(pred, depth, rec, mask)
        # pred, mask = evaluate.norm_diff(pred, gt, rec, mask)

        depth *= mask
        pred *= mask

        pred_img = depth_tools.pack_float_to_bmp_bgra(pred)
        cv2.imwrite(dir_pred + 'pred_{:03d}.bmp'.format(idx), pred_img)

        """ Plot """
        mean_depth = np.sum(depth) / np.sum(mask)
        depth_range = 0.02
        vmin, vmax = mean_depth - depth_range, mean_depth + depth_range

        plt.figure(figsize = (15, 5))
        plt.subplot(1, 3, 1)
        # plt.imshow(img[:, :, ::-1]/255)
        plt.imshow(img[:, :, ::-1])
        plt.title("RGB")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(depth, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Depth Ground Thruth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Depth prediction")
        plt.axis("off")

        plt.savefig(dir_pred + 'result_{:03d}.png'.format(idx))
        # plt.show()
        plt.close()