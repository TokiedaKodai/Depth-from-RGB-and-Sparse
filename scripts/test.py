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
from utils.loader import render_wave, board_data, real_data, NYU_V2
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
model_name = 'test'
# model_name = 'test/finetune'
model_name = 'wave2_600'
model_name = 'wave2_600/finetune'
model_name = 'wave2_600_sample-8'
model_name = 'wave2_600_depth'
model_name = 'wave2_600_depth/finetune'
model_name = 'nyu'
model_name = 'nyu/finetune'

pred_name = 'pred_'
pred_name = 'pred_train'
pred_name = 'pred_val'
# pred_name = 'pred_test'
pred_name = 'pred_board'
pred_name = 'pred_real'

''' GT '''
is_learn_diff = False
is_norm_diff = True
is_use_rec = True
# is_use_rec = False

''' OPTIOIN '''
data_size = 600

''' SETTING '''
rgb_threshold = 0
depth_threshold = 0.1
# depth_threshold = 0
difference_threshold = 0.005
difference_threshold = 0.01
# difference_threshold = 1
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
# train_set = render_wave(data_size=data_size, trn_tst='train', return_rec=is_use_rec)
# test_set = render_wave(data_size=data_size, trn_tst='test', return_rec=is_use_rec)
res = 512
test_set = board_data(trn_tst='test', res=res, return_rec=is_use_rec)
test_set = real_data(trn_tst='test', res=res, return_rec=is_use_rec)

# test_set = NYU_V2(trn_tst='val')


# print(f'Number of training examples: {len(train_set)}')
# print(f'Number of testing examples: {len(test_set)}')

''' Training data '''
# testloader = DataLoader(train_set, batch_size=1, shuffle=False)
''' Test data '''
testloader = DataLoader(test_set, batch_size=1, shuffle=False)
print('Loader built')


""" Testing and visualising data """
model = D3()
# model.load_state_dict(torch.load(dir_model + 'saved_model.pt'))
model.load_state_dict(torch.load(dir_model + 'saved_model_best.pt'))
# testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

result = 'idx,RMSE\n'

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

    for idx, (img, depth, sp) in enumerate(testloader):
        print('{}/{}'.format(idx+1, len(testloader)), end=' ')
        start_time = time.time()

        NN = sp

        img = img.permute(0, 3, 1, 2)
        img = img.to(device) 
        depth = depth.to(device)
        NN = NN.to(device)

        fx = model(img, NN)

        img = img[0].permute(1, 2, 0).cpu().numpy()
        depth = depth[0].cpu().numpy()
        fx = fx[0][0].cpu().numpy()
        NN = NN.cpu().numpy()

        s1 = NN[:, 0, :, :]
        s2 = NN[:, 1, :, :]

        diff = fx
        sparse = s1[0]
        # pred += s1[0]

        ''' Normalize prediction '''
        img_r = img[:, :, 0]
        mask_rgb = img[:, :, 0] > rgb_threshold
        mask_depth = depth > depth_threshold
        mask_sparse = sparse > depth_threshold
        mask_close = np.abs(sparse - depth) < difference_threshold
        mask = mask_depth * mask_sparse * mask_close
        # mask = mask_rgb * mask_depth * mask_sparse #* mask_close
        mask = mask.astype(np.float32)
        # cv2.imwrite(dir_pred+'mask.png', mask.astype(np.int8)*255)
        # rec = rec[0][0].cpu().numpy()

        if is_learn_diff:
            gt = (depth - sparse) * mask
            length = np.sum(mask)
            mean_gt = np.sum(gt) / length
            max_gt = np.max(np.abs(gt - mean_gt))
            diff = diff * max_gt + mean_gt
            pred = sparse + diff
        else:
            pred = diff

        ''' Normalize '''
        pred, mask = evaluate.norm_diff(pred, depth, sparse, mask)

        depth *= mask
        pred *= mask

        ''' Evaluation '''
        rmse = evaluate.evaluate_rmse(pred, depth, mask)
        result += f'{idx},{rmse}\n'

        ''' Save Prediction '''
        pred_img = depth_tools.pack_float_to_bmp_bgra(pred)
        cv2.imwrite(dir_pred + 'pred_{:03d}.bmp'.format(idx), pred_img)

        """ Plot """
        mean_depth = np.sum(depth) / np.sum(mask)
        depth_range = 0.02
        # mean_depth = np.mean(depth)
        # depth_range = 1
        vmin, vmax = mean_depth - depth_range, mean_depth + depth_range

        plt.figure(figsize = (15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(sparse, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Sparse Depth")
        plt.axis("off")
        
        plt.subplot(2, 3, 2)
        plt.imshow(depth, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Depth Ground Thruth")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Depth prediction")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        # plt.imshow(img[:, :, ::-1]/255)
        plt.imshow(img[:, :, ::-1])
        plt.title("RGB")
        plt.axis("off")

        sparse_err = np.abs(sparse - depth) * mask
        pred_err = np.abs(pred - depth) * mask
        err_range = 0.001
        # sparse_err = np.abs(sparse - depth)
        # pred_err = np.abs(pred - depth)
        # err_range = 0.5

        plt.subplot(2, 3, 5)
        plt.imshow(sparse_err, cmap='jet', vmin=0, vmax=err_range)
        plt.title("Sparse Depth Error")
        plt.axis("off")
        
        plt.subplot(2, 3, 6)
        plt.imshow(pred_err, cmap='jet', vmin=0, vmax=err_range)
        plt.title("Prediction Error")
        plt.axis("off")

        plt.savefig(dir_pred + 'result_{:03d}.png'.format(idx))
        # plt.show()
        plt.close()

        print('Time: {} s'.format(time.time() - start_time))

with open(dir_pred + 'result.txt', mode='w') as f:
    f.write(result)