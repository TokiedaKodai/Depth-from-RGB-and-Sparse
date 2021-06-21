import numpy as np
import cv2
import pandas as pd
import sys
import os
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.depth_tools import unpack_bmp_bgra_to_float, pack_float_to_bmp_bgra
from utils.evaluate import norm_diff
import config


dir_data = config.dir_data + 'cardboard/'
dir_supres = '../../../Lab/PixTransform/compare/cardboard_result_512/'
idxs = list(range(44, 56))
dir_save = config.dir_result + '/supres_board_512/'

dir_data = config.dir_data + 'reals/'
dir_supres = '../../../Lab/PixTransform/compare/reals_result_512/'
idxs = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 17, 18]
dir_save = config.dir_result + '/supres_real_512/'

img_res = 512
res = 256

def get_supres(dire, idx):
    pred = np.zeros((img_res, img_res))
    for i in range(2):
        for j in range(2):
            patch = np.load(dire+'predict-{:03d}.npy'.format(idx*4 + i*2 + j))
            pred[i*res:(i+1)*res, j*res:(j+1)*res] = patch
    return pred

os.makedirs(dir_save, exist_ok=True)

for i, idx in enumerate(tqdm(idxs)):
    gt = cv2.imread(dir_data+'gt_512/{:05d}.bmp'.format(idx), -1)
    rec = cv2.imread(dir_data+'rec_512/{:05d}.bmp'.format(idx), -1)
    gt = unpack_bmp_bgra_to_float(gt)
    rec = unpack_bmp_bgra_to_float(rec)

    # pred_supres = get_supres(dir_supres, i)
    pred_supres = np.load(dir_supres + 'predict-{:03d}.npy'.format(i))

    mask_gt = (gt > 0.1)*1.0
    mask_rec = (rec > 0.1)*1.0
    mask_supres = (pred_supres > 0.1)*1.0
    mask_supres *= mask_gt * mask_rec
    pred_supres, mask = norm_diff(pred_supres, gt, rec, mask_supres)

    supres_img = pack_float_to_bmp_bgra(pred_supres*mask)
    cv2.imwrite(dir_save+'{:03d}.bmp'.format(i), supres_img)