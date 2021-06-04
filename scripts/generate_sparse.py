import numpy as np
import cv2
import sys
import os
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.sparse import NN_fill, generate_mask
from utils import tools
from codes import config

dir_data = config.dir_data + 'render_wave2-pose_600/'
res = 512

# 24x24 downsampling
mask = generate_mask(24, 24, res, res)

for i in tqdm(range(310, 600)):
    gt_img = cv2.imread(dir_data+'gt_512/{:05d}.bmp'.format(i), -1)
    shade = cv2.imread(dir_data+'shade_512/{:05d}.png'.format(i), 1)

    gt = tools.unpack_bmp_bgra_to_float(gt_img)

    sp = NN_fill(shade, gt, mask)

    np.save(dir_data + 'sparse_512/{:05d}.npy'.format(i), sp)