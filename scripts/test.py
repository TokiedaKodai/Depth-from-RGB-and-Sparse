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

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.model import D3
from utils.sparse import NN_fill, generate_mask
from utils.loader import NYU_V2, render_wave, board_data, real_data
from utils import tools
import config

''' MODEL NAME '''
model_name = 'wave2'
pred_name = 'pred_board'
# pred_name = 'pred_real'

dir_model = config.dir_models + model_name + '/'
dir_pred = dir_model + pred_name + '/'
os.makedirs(dir_pred, exist_ok=True)

res = 512
# 24x24 downsampling
# mask = generate_mask(24, 24, 480, 640)
mask = generate_mask(24, 24, res, res)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


""" Creating Train loaders """
# train_set = NYU_V2(trn_tst=0)
# test_set = NYU_V2(trn_tst=1)
# train_set = render_wave(trn_tst=0)
# test_set = render_wave(trn_tst=1)

test_set = board_data()
# test_set = real_data()

# print(f'Number of training examples: {len(train_set)}')
print(f'Number of testing examples: {len(test_set)}')

testloader = DataLoader(test_set, batch_size=1, shuffle=True)
print('Loader built')


""" Testing and visualising data """
model = D3()
model.load_state_dict(torch.load(dir_model + 'saved_model.pt'))
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

print('Test...')
with torch.no_grad():
    model.eval()
    model.to(device)

    for idx, (img, depth) in enumerate(testloader):
        NN = []
        #concat x with spatial data
        for j in range(img.shape[0]):
            sp = NN_fill(img[j].numpy(), depth[j].numpy(), mask)
            NN.append(sp)
        NN = torch.tensor(NN)

    # for idx, (img, depth, sp) in enumerate(testloader):
        # NN = sp

        img = img.permute(0, 3, 1, 2)
        img = img.to(device) 
        depth = depth.to(device)
        NN = NN.to(device)
        # print(NN.shape)
        # print(img.shape)
        # print(depth.shape)

        fx = model(img, NN)

        tets = img[0].permute(1, 2, 0).cpu().numpy()
        depth = depth[0].cpu().numpy()
        fx = fx[0][0].cpu().numpy()

        fx_img = tools.pack_float_to_bmp_bgra(fx)
        cv2.imwrite(dir_pred + 'pred_{:03d}.bmp'.format(idx), fx_img)

        """ Plot """
        mean_depth = np.mean(fx)
        depth_range = 0.02
        vmin, vmax = mean_depth - depth_range, mean_depth + depth_range

        plt.figure(figsize = (15, 5))
        plt.subplot(1, 3, 1)
        # plt.imshow(tets/255)
        plt.imshow(tets[:, :, ::-1])
        plt.title("RGB")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(depth, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Depth Ground Thruth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(fx, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Depth prediction")
        plt.axis("off")

        plt.savefig(dir_pred + 'result_{:03d}.png'.format(idx))
        # plt.show()