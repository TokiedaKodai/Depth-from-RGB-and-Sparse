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

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.model import D3
from utils.sparse import NN_fill, generate_mask
from utils.loader import NYU_V2, render_wave
import config

''' MODEL NAME '''
model_name = 'wave2'

dir_model = config.dir_models + model_name + '/'
os.makedirs(dir_model, exist_ok=True)

res = 512
# 24x24 downsampling
# mask = generate_mask(24, 24, 480, 640)
mask = generate_mask(24, 24, res, res)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

""" Training funcion (per epoch) """
def train(net, device, loader, optimizer, Loss_fun):
    #initialise counters
    running_loss = 0.0
    loss = []
    net.train()
    torch.no_grad()

    # train batch
    start_time = time.time()
    for i, (x, y, sp) in enumerate(loader):
        # NN = []
        # # concat x with spatial data
        # for j in range(x.shape[0]):
        #     sp = NN_fill(x[j].numpy(), y[j].numpy(), mask)
        #     NN.append(sp)
        # NN = torch.tensor(NN)
        NN = sp
        
        optimizer.zero_grad()

        x = x.permute(0, 3, 1, 2)
        x = x.to(device) 
        y = y.to(device)
        NN = NN.to(device)
        fx = net(x, NN)
        fx = fx.permute(1, 0, 2, 3)
        loss = Loss_fun(fx[0], y)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    #end training 
    end_time = time.time() 
    running_loss /= len(loader)

    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    torch.save(net.state_dict(), dir_model + "saved_model.pt") 

    return running_loss


""" Creating Train loaders """
# train_set = NYU_V2(trn_tst=0)
# test_set = NYU_V2(trn_tst=1)
train_set = render_wave(trn_tst=0)
test_set = render_wave(trn_tst=1)

print(f'Number of training examples: {len(train_set)}')
print(f'Number of testing examples: {len(test_set)}')

batch_size     = 8
# batch_size     = 16
num_epochs     = 1000
learning_rate  = 0.001
# n_workers = multiprocessing.cpu_count()
# print('Worker num: ', n_workers)

#initialising data loaders
# trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
# testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
print('Loader built')

img_batch, depth_batch, sp_batch = next(iter(trainloader))

""" Plot dataset """
plot_num = 4
plt.figure(figsize = (plot_num*2, 4))
for tmp in range(plot_num):  
    plt.subplot(2,plot_num,tmp+1)
    # plt.imshow(img_batch[tmp]/255)
    plt.imshow(img_batch[tmp])
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2,plot_num,tmp+5)
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
Loss_fun  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

print('Training...')
for epoch in range(num_epochs):
    print("Epoch", epoch + 1)
    train_loss= train(model, device, trainloader, optimizer, Loss_fun)
    df_log = pd.DataFrame({'Epoch': [epoch+1], 'Loss': [train_loss]})
    try:
        filepath = dir_model + "training_log.csv"
        is_header = not os.path.isfile(filepath)
        df_log.to_csv(filepath, mode='a', header=is_header)
    except Exception as e:
        print("log_data Error: " + str(e))
