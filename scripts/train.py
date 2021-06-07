import numpy as np
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
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.network import D3
from utils.sparse import NN_fill, generate_mask
from utils.loader import render_wave
import config
###############################################################################################

''' MODEL NAME '''
model_name = 'wave2_100'
model_name = 'overfit'

''' ARG '''
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='model name to use training and test')
parser.add_argument('--epoch', type=int, help='epoch num')
args = parser.parse_args()

if args.name is not None:
    model_name = args.name

dir_model = config.dir_models + model_name + '/'
os.makedirs(dir_model, exist_ok=True)

res = 512
# 24x24 downsampling
# mask = generate_mask(24, 24, 480, 640)
# mask = generate_mask(24, 24, res, res)
# mask = generate_mask(48, 48, res, res)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

''' Training Parameters '''
data_size      = 600
# data_size      = 100
data_size      = 12
# data_size      = 2
batch_size     = 8
# batch_size     = 1
num_epochs     = 10000
learning_rate  = 1e-3
# learning_rate  = 1e-4
lr_decay_rate  = 0.5
lr_decay_epoch = 1000

# n_workers = multiprocessing.cpu_count()
# print('Worker num: ', n_workers)

if args.epoch is not None:
    num_epochs = args.epoch

rgb_threshold = 0
gt_threshold = 0.1

###############################################################################################
''' Loss function '''
def MaskMSE(outputs, targets, mask):
        err = torch.square(outputs - targets) * mask
        loss =  torch.sum(err) / torch.sum(mask)
        return loss

""" Training funcion (per epoch) """
def train(net, device, loader, optimizer, Loss_fun):
    #initialise counters
    running_loss = 0.0
    loss = []
    net.train()
    torch.no_grad()

    # train batch
    start_time = time.time()
    for i, (x, y, sp, rec) in enumerate(loader):
        optimizer.zero_grad()

        # NN = []
        # # concat x with spatial data
        # for j in range(x.shape[0]):
        #     sp = NN_fill(x[j].numpy(), y[j].numpy(), mask)
        #     NN.append(sp)
        # NN = torch.tensor(NN)
        NN = sp
        # NN = torch.cat((rec, rec), 1)
        
        x = x.permute(0, 3, 1, 2)
        x = x.to(device) 
        y = y.to(device)
        NN = NN.to(device)
        ''' Predict '''
        fx = net(x, NN)
        fx = fx.permute(1, 0, 2, 3)
        # pred = fx[0]
        # pred += sparse

        mask_rgb = x[:, 0, :, :] > rgb_threshold
        mask_depth = y > gt_threshold
        mask = mask_rgb * mask_depth
        mask = mask.float()
        mask = mask.to(device)

        rec = rec.to(device)[0]

        s1 = NN[:, 0, :, :]
        s2 = NN[:, 1, :, :]
        gt = (y - s1) * mask
        # gt = (y - rec) * mask
        length = torch.sum(mask)
        mean_gt = torch.sum(gt) / length
        std_gt = torch.sqrt(torch.sum(torch.square(gt - mean_gt)*mask) / length)
        gt = (gt - mean_gt) / std_gt
        gt = gt.to(device)

        # loss = Loss_fun(fx[0], y)

        # loss = Loss_fun(fx[0], y, mask)
        loss = Loss_fun(fx[0], gt, mask)
        
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    #end training 
    end_time = time.time() 
    running_loss /= len(loader)

    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's', end='        \r')
    torch.save(net.state_dict(), dir_model + "saved_model.pt") 

    return running_loss
###############################################################################################


""" Creating Train loaders """
train_set = render_wave(data_size=data_size, trn_tst=0, return_rec=True)
# test_set = render_wave(data_size=data_size, trn_tst=1)

print(f'Number of training examples: {len(train_set)}')
# print(f'Number of testing examples: {len(test_set)}')

#initialising data loaders
# trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
# testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print('Loader built')

img_batch, depth_batch, sp_batch, rec_batch = next(iter(trainloader))

# trainloader = [[img_batch[:], depth_batch[:], sp_batch[:]]]

""" Plot dataset """
plot_num = min(4, batch_size)
plt.figure(figsize = (plot_num*2, 4))
for tmp in range(plot_num):  
    plt.subplot(2,plot_num,tmp+1)
    # plt.imshow(img_batch[tmp]/255)
    plt.imshow(img_batch[tmp])
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2,plot_num,tmp+plot_num+1)
    plt.imshow(depth_batch[tmp])
    plt.title("Depth")
    plt.axis("off")
plt.savefig(dir_model + 'dataset.png')
# plt.show()


""" Training loop (for multiple epochs)"""
"""
Steps for skipping training:
1) check if saved_model.pt is available 
2) comment training loop
3) uncomment testing loop
"""

model = D3().float()
model = model.to(device)
# Loss_fun  = nn.MSELoss()
# Loss_fun  = nn.L1Loss()
Loss_fun  = MaskMSE
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

''' Re-Training '''
# model.load_state_dict(torch.load(dir_model + 'saved_model.pt'))

print('Training...')
for epoch in range(num_epochs):
    ''' Learning Rate Decay '''
    if epoch % lr_decay_epoch == 0:
        learning_rate *= lr_decay_rate
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Epoch:", epoch + 1, end=' --- ')
    train_loss= train(model, device, trainloader, optimizer, Loss_fun)
    df_log = pd.DataFrame({'Epoch': [epoch+1], 'Loss': [train_loss]})
    try:
        filepath = dir_model + "training_log.csv"
        is_header = not os.path.isfile(filepath)
        df_log.to_csv(filepath, mode='a', header=is_header)
    except Exception as e:
        print("log_data Error: " + str(e))
print('\nTraining end.')