import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import multiprocessing
import matplotlib.pyplot as plt
import time
import csv
import sys
import os
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.network import D3
from utils.sparse import NN_fill, generate_mask
from utils.loader import render_wave, board_data, real_data, NYU_V2
import config
###############################################################################################

''' MODEL NAME '''
model_name = 'test'
model_name = 'wave2_600'
model_name = 'wave2_600_sample-8'
model_name = 'wave2_600_depth'
model_name = 'nyu'

''' ARG '''
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='model name to use training and test')
parser.add_argument('--epoch', type=int, help='epoch num')
parser.add_argument('--finetune', action='store_true', help='add to fine-tune')
args = parser.parse_args()

if args.name is not None:
    model_name = args.name

dir_model = config.dir_models + model_name + '/'
os.makedirs(dir_model, exist_ok=True)

'''#################### Training Parameters ####################'''
''' Learn '''
is_learn_diff = False
is_norm_diff = True
is_use_rec = True

''' Fine-tune '''
is_finetune = False
is_finetune = True

''' Data Size '''
data_size      = 600
# data_size      = 100
# data_size      = 12
batch_size     = 8

# data_size      = 2
# batch_size     = 1
''' Epoch '''
start_epoch    = 0
end_epoch      = None
# end_epoch      = 1000
num_epochs     = None
num_epochs     = 500
save_model_per_epoch = 100
''' Learning Rate '''
learning_rate  = 1e-3
# learning_rate  = 1e-4
# learning_rate  = 2e-5
# learning_rate  = 1.8014e-05
lr_decay_rate  = 0.5
lr_decay_epoch = 100000

''' Setting '''
rgb_threshold = 0
depth_threshold = 0.1
# depth_threshold = 0
verbose = '\n'
# verbose = '        \r'
res = 512

''' Sampling setting '''
# 24x24 downsampling
# mask = generate_mask(24, 24, 480, 640)
# mask = generate_mask(24, 24, res, res)
''' ########################### Check ################################'''
''' Fine-tune '''
if is_finetune:
    dir_finetune = dir_model + 'finetune/'
    os.makedirs(dir_finetune, exist_ok=True)
    log_filepath = dir_finetune + 'training_log.csv'
else:
    log_filepath = dir_model + 'training_log.csv'

''' Exist model '''
try:
    df_log = pd.read_csv(log_filepath)
    # Epoch
    last_epoch = int(df_log.iloc[-1]['Epoch'])
    start_epoch = last_epoch
    # Learning Rate
    last_lr = float(df_log.iloc[-1]['Learning_Rate'])
    lr = last_lr
    is_retrain = True
except:
    is_retrain = False
    lr = learning_rate
''' Epoch '''
if args.epoch is not None:
    num_epochs = args.epoch
if num_epochs is not None:
    end_epoch = start_epoch + num_epochs

''' Check CUDA '''
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# n_workers = multiprocessing.cpu_count()
# print('Worker num: ', n_workers)

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

        # NN = sp
        NN = rec.permute(0, 3, 1, 2)
        # NN = torch.cat((rec, rec), 1)
        
        x = x.permute(0, 3, 1, 2)
        x = x.to(device) 
        y = y.to(device)
        NN = NN.to(device)
        ''' Output '''
        fx = net(x, NN)
        fx = fx.permute(1, 0, 2, 3)
        pred = fx[0]
        # pred += sparse

        mask_rgb = x[:, 0, :, :] > rgb_threshold
        mask_depth = y > depth_threshold
        mask = mask_rgb * mask_depth
        mask = mask.float()
        
        mask = mask.to(device)

        # rec = rec.to(device)[0]

        s1 = NN[:, 0, :, :]
        s2 = NN[:, 1, :, :]
        gt = (y - s1) * mask
        # gt = (y - rec) * mask
        length = torch.sum(mask, (1, 2))
        mean_gt = torch.sum(gt, (1, 2)) / length
        for j in range(batch_size):
            gt[j, :, :] -= mean_gt[j]
        std_gt = torch.sqrt(
            torch.sum(
                torch.square(
                    gt
                ) * mask, (1, 2)
            ) / length
        )
        for j in range(batch_size):
            gt[j, :, :] /= std_gt[j]
        gt = gt.to(device)

        # loss = Loss_fun(fx[0], y)

        # loss = Loss_fun(fx[0], y, mask)
        loss = Loss_fun(pred, gt, mask)
        
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    #end training 
    end_time = time.time() 
    running_loss /= len(loader)

    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's', end='        \r')
    torch.save(net.state_dict(), dir_model + "saved_model.pt")

    return running_loss

def train_loaded(net, device, loader, optimizer, Loss_fun):
    #initialise counters
    running_loss = 0.0
    loss = []
    net.train()
    # torch.no_grad()

    # train batch
    start_time = time.time()
    for i, (x, sp, gt, mask) in enumerate(loader):
        optimizer.zero_grad()

        NN = sp
        
        x = x.to(device)
        NN = NN.to(device)
        ''' Output '''
        fx = net(x, NN)
        fx = fx.permute(1, 0, 2, 3)
        pred = fx[0]

        mask = mask.to(device)
        gt = gt.to(device)

        loss = Loss_fun(pred, gt, mask)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    #end training 
    end_time = time.time()
    running_loss /= len(loader)

    print('  Training Loss :', running_loss, '-- Time:', end_time - start_time, 's', end=verbose)
    torch.save(net.state_dict(), dir_model + "saved_model.pt")

    return running_loss

''' Validation '''
def validate_loaded(net, device, loader, optimizer, Loss_fun):
    net.eval()
    #initialise counters
    running_loss = 0.0
    loss = []
    
    with torch.no_grad():
        start_time = time.time()
        for i, (x, sp, gt, mask) in enumerate(loader):
            optimizer.zero_grad()

            NN = sp

            x = x.to(device)
            NN = NN.to(device)
            ''' Predict '''
            fx = net(x, NN)
            fx = fx.permute(1, 0, 2, 3)
            pred = fx[0]

            mask = mask.to(device)
            gt = gt.to(device)

            loss = Loss_fun(pred, gt, mask)
            running_loss += loss.item()

    end_time = time.time() 
    running_loss /= len(loader)

    print(' '*23, 'Validation Loss :', running_loss, '-- Time:', end_time - start_time, 's', end=verbose)
    return running_loss
''' Load Data on Memory '''
def load_data(loader):
    batch_data = []

    for i, (x, y, sp) in enumerate(loader):
        x = x.permute(0, 3, 1, 2)
        # sp = sp.permute(0, 3, 1, 2)

        mask_rgb = x[:, 0, :, :] > rgb_threshold
        mask_depth = y > depth_threshold
        mask_sp = sp[:, 0, :, :] > depth_threshold
        mask = mask_rgb * mask_depth * mask_sp
        mask = mask.float()

        s1 = sp[:, 0, :, :]
        if is_learn_diff:
            gt = (y - s1) * mask
            if is_norm_diff:
                length = torch.sum(mask, (1, 2))
                mean_gt = torch.sum(gt, (1, 2)) / length
                gt[0, :, :] -= mean_gt[0]
                std_gt = torch.sqrt(
                    torch.sum(
                        torch.square(
                            gt
                        ) * mask
                    ) / length
                )
                gt[0, :, :] /= std_gt[0]
        else:
            gt = y * mask

        batch_data.append([x[0], sp[0], gt[0], mask[0]])
    return batch_data
###############################################################################################


""" Create loaders """
train_set = render_wave(data_size=data_size, trn_tst='train', return_rec=is_use_rec)
val_set = render_wave(data_size=data_size, trn_tst='test', return_rec=is_use_rec)
finetune_board_set = board_data(trn_tst='train', return_rec=is_use_rec, patch_res=256)
finetune_real_set = real_data(trn_tst='train', return_rec=is_use_rec, patch_res=256)

''' NYU '''
train_set = NYU_V2(trn_tst='train')
val_set = NYU_V2(trn_tst='val')


#initialising data loaders
# trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

print('Loading data...')
if is_finetune:
    print('Fine-tuning')
    ''' Fine-tune Loader '''
    finetune_board_loader = DataLoader(finetune_board_set, batch_size=1, shuffle=False)
    finetune_real_loader = DataLoader(finetune_real_set, batch_size=1, shuffle=False)
    finetune_board_data = load_data(finetune_board_loader)
    finetune_real_data = load_data(finetune_real_loader)
    finetune_data = finetune_board_data + finetune_real_data
    finetune_train_data, finetune_val_data = train_test_split(
        finetune_data, 
        test_size=0.3, 
        shuffle=True,
        random_state=0)
    print(f'Number of training examples: {len(finetune_train_data)}')
    print(f'Number of validation examples: {len(finetune_val_data)}')
    finetune_batch_size = 4
    trainloader = DataLoader(finetune_train_data, batch_size=finetune_batch_size, shuffle=True)
    valloader = DataLoader(finetune_val_data, batch_size=finetune_batch_size, shuffle=True)
    print('Fine-tuning Loaders built')
else:
    print(f'Number of training examples: {len(train_set)}')
    print(f'Number of validation examples: {len(val_set)}')

    trainloader = DataLoader(train_set, batch_size=1, shuffle=False)
    train_data = load_data(trainloader)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print('Training Loader built')

    valloader = DataLoader(val_set, batch_size=1, shuffle=False)
    val_data = load_data(valloader)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    print('Validation Loader built')

print(f'Number of training batches: {len(trainloader)}')
print(f'Number of validation batches: {len(valloader)}')

""" Plot dataset """
# img_batch, depth_batch, sp_batch, rec_batch = next(iter(trainloader))

# plot_num = min(4, batch_size)
# plt.figure(figsize = (plot_num*2, 4))
# for tmp in range(plot_num):  
#     plt.subplot(2,plot_num,tmp+1)
#     # plt.imshow(img_batch[tmp]/255)
#     plt.imshow(img_batch[tmp])
#     plt.title("Image")
#     plt.axis("off")

#     plt.subplot(2,plot_num,tmp+plot_num+1)
#     plt.imshow(depth_batch[tmp])
#     plt.title("Depth")
#     plt.axis("off")
# plt.savefig(dir_model + 'dataset.png')
# # plt.show()

''' Model build '''
model = D3().float()
model = model.to(device)
Loss_fun  = MaskMSE
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8, patience=20, verbose=True)

''' Load Model for Re-Training '''
if is_finetune and is_retrain:
    model.load_state_dict(torch.load(dir_finetune + 'saved_model.pt'))
elif is_finetune:
    model.load_state_dict(torch.load(dir_model + 'saved_model_best.pt'))
elif is_retrain:
    model.load_state_dict(torch.load(dir_model + 'saved_model.pt'))

''' Change model dir for Fine-tune '''
if is_finetune:
    dir_model = dir_finetune
''' Init min val loss '''
if is_retrain:
    min_loss = float(df_log['Val_Loss'].min())
else:
    min_loss = float('inf')

'''
#####################################################################
####################### TRAINING ####################################
#####################################################################
'''
print('Training...')
for epoch in range(start_epoch, end_epoch):
    ''' Learning Rate Decay '''
    if epoch % lr_decay_epoch == 0:
        learning_rate *= lr_decay_rate
        optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Epoch: {:5d}/{:5d}".format(epoch + 1, end_epoch), end=' ---- ')
    # train_loss= train(model, device, trainloader, optimizer, Loss_fun)
    train_loss= train_loaded(model, device, trainloader, optimizer, Loss_fun)
    val_loss= validate_loaded(model, device, valloader, optimizer, Loss_fun)
    scheduler.step(val_loss)
    log_lr = scheduler.state_dict()['_last_lr']

    df_log = pd.DataFrame({
        'Epoch': [epoch+1], 
        'Loss': [train_loss], 
        'Val_Loss': [val_loss],
        'Learning_Rate':log_lr
        })
    df_log.set_index('Epoch', inplace=True)
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), dir_model + "saved_model_best.pt")
    if (epoch+1) % save_model_per_epoch == 0:
        torch.save(model.state_dict(), dir_model + "saved_model_{}.pt".format(epoch+1))
    try:
        is_header = not os.path.exists(log_filepath)
        df_log.to_csv(log_filepath, mode='a', header=is_header)
    except Exception as e:
        print("log_data Error: " + str(e))
print('\nTraining end.')