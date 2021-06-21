import numpy as np
import cv2
import sys
import os
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.sparse import NN_fill, generate_mask
from utils import depth_tools
from scripts import config

dir_data = config.dir_data + 'render_wave2-pose_600/'
idxs = range(600)

dir_data = config.dir_data + 'cardboard/'
idxs = list(range(16)) + list(range(40, 56))

dir_data = config.dir_data + 'reals/'
idxs = range(1, 19)

res = 512
# res = 1024

# 24x24 downsampling
mask = generate_mask(24, 24, res, res)
mask = generate_mask(8, 8, res, res)

dir_save = dir_data + 'sparse_{}_sample-8/'.format(res)
os.makedirs(dir_save, exist_ok=True)

for i in tqdm(idxs):
    gt_img = cv2.imread(dir_data+'gt_{}/{:05d}.bmp'.format(res, i), -1)
    shade = cv2.imread(dir_data+'shade_{}/{:05d}.png'.format(res, i), 1)

    gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)

    sp = NN_fill(shade, gt, mask)

    np.save(dir_save + '{:05d}.npy'.format(i), sp)