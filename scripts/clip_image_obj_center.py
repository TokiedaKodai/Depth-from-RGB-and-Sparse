import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

import config
from utils.depth_tools import unpack_bmp_bgra_to_float

# dir_in = config.dir_data + 'board/'
# dir_out = config.dir_data + 'cardboard/'
# dir_in = config.dir_data + 'real/'
# dir_out = config.dir_data + 'reals/'
dir_in = config.dir_data + 'render_wave2-pose_600/'
dir_out = config.dir_data + 'synthetic/'

# in_gt = dir_in + 'gt/gt{:03d}.bmp'
# in_rec = dir_in + 'rec/depth{:03d}.bmp'
# in_shade = dir_in + 'shading/shading{:03d}.bmp'
# in_proj = dir_in + 'frame/frame{:03d}.png'
in_gt = dir_in + 'gt/{:05d}.bmp'
in_rec = dir_in + 'rec/{:05d}.bmp'
in_shade = dir_in + 'shade/{:05d}.png'
in_proj = dir_in + 'proj/{:05d}.png'

out_gt_folder = dir_out + 'gt_512/'
out_rec_folder = dir_out + 'rec_512/'
out_shade_folder = dir_out + 'shade_512/'
out_proj_folder = dir_out + 'proj_512/'
out_mask_folder = dir_out + 'mask_512/'

# out_gt_folder = dir_out + 'gt_1024/'
# out_rec_folder = dir_out + 'rec_1024/'
# out_shade_folder = dir_out + 'shade_1024/'
# out_proj_folder = dir_out + 'proj_1024/'
# out_mask_folder = dir_out + 'mask_1024/'

os.makedirs(out_gt_folder, exist_ok=True)
os.makedirs(out_rec_folder, exist_ok=True)
os.makedirs(out_shade_folder, exist_ok=True)
os.makedirs(out_proj_folder, exist_ok=True)
os.makedirs(out_mask_folder, exist_ok=True)

out_gt = out_gt_folder + '{:05d}.bmp'
out_rec = out_rec_folder + '{:05d}.bmp'
out_shade = out_shade_folder + '{:05d}.png'
out_proj = out_proj_folder + '{:05d}.png'
out_mask = out_mask_folder + '{:05d}.png'

res_x = 1200
res_y = 1200
clip = 512
# clip = 1024
clip_x = clip
clip_y = clip
clip_center_x = clip_x // 2
clip_center_y = clip_y // 2

# idxs = list(range(16)) + list(range(40, 56))
idxs = range(600)

for i in tqdm(idxs):
    gt_img = cv2.imread(in_gt.format(i), -1)[:res_y, :res_x, :]
    rec_img = cv2.imread(in_rec.format(i), -1)[:res_y, :res_x, :]
    shade = cv2.imread(in_shade.format(i), 1)[:res_y, :res_x, :]
    proj = cv2.imread(in_proj.format(i), 1)[:res_y, :res_x, :]

    gt = unpack_bmp_bgra_to_float(gt_img)
    rec = unpack_bmp_bgra_to_float(rec_img)

    valid_gt = gt > 0.2
    valid_rec = rec > 0.2
    close_depth = np.abs(gt - rec) < 0.01
    mask = np.logical_and(valid_gt, valid_rec, close_depth) * 1.0
    length = np.sum(mask)

    grid_x, grid_y = np.meshgrid(
        list(range(res_x)),
        list(range(res_y))
    )
    center_x = int(np.sum(grid_x * mask) / length)
    center_y = int(np.sum(grid_y * mask) / length)

    if center_x < clip_center_x:
        center_x = clip_center_x
    if center_y < clip_center_y:
        center_y = clip_center_y
    if center_x > res_x - clip_center_x:
        center_x = res_x - clip_center_x
    if center_y > res_y - clip_center_y:
        center_y = res_y - clip_center_y
    
    start_x = center_x - clip_center_x
    start_y = center_y - clip_center_y
    end_x = center_x + clip_center_x
    end_y = center_y + clip_center_y

    gt_clip = gt_img[start_y: end_y, start_x: end_x, :]
    rec_clip = rec_img[start_y: end_y, start_x: end_x, :]
    shade_clip = shade[start_y: end_y, start_x: end_x, :]
    proj_clip = proj[start_y: end_y, start_x: end_x, :]
    mask_clip = mask[start_y: end_y, start_x: end_x] * 255

    cv2.imwrite(out_gt.format(i), gt_clip)
    cv2.imwrite(out_rec.format(i), rec_clip)
    cv2.imwrite(out_shade.format(i), shade_clip)
    cv2.imwrite(out_proj.format(i), proj_clip)
    cv2.imwrite(out_mask.format(i), mask_clip)
