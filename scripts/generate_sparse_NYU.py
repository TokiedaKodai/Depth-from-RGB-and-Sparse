import numpy as np
import cv2
import sys
import os
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.sparse import NN_fill, generate_mask
from utils.loader import NYU_V2
from utils import depth_tools
from scripts import config

# 24x24 downsampling
mask = generate_mask(24, 24, 480, 640)

''' NYU Loader '''
loader = NYU_V2(trn_tst='all')

dir_save = config.dir_data + 'NYU/sparse/'
os.makedirs(dir_save, exist_ok=True)

for idx, (img, depth) in enumerate(tqdm(loader)):
    img = img.numpy()
    depth = depth.numpy()

    sp = NN_fill(img, depth, mask)

    np.save(dir_save + '{:05d}.npy'.format(idx), sp)