import cv2
import sys
import os
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from scripts import config

dir_data = config.dir_data + 'render_wave2-pose_600/'
res = 512
H = 1200
W = 1600
w_start = (W - res) // 2
w_end = (W + res) // 2

for i in tqdm(range(600)):
    gt = cv2.imread(dir_data+'gt/{:05d}.bmp'.format(i), -1)
    rec = cv2.imread(dir_data+'rec/{:05d}.bmp'.format(i), -1)
    shade = cv2.imread(dir_data+'shade/{:05d}.png'.format(i), 1)
    proj = cv2.imread(dir_data+'proj/{:05d}.png'.format(i), 1)

    cv2.imwrite(dir_data+'gt_512/{:05d}.bmp'.format(i), gt[-res:, w_start:w_end, :])
    cv2.imwrite(dir_data+'rec_512/{:05d}.bmp'.format(i), rec[-res:, w_start:w_end, :])
    cv2.imwrite(dir_data+'shade_512/{:05d}.png'.format(i), shade[-res:, w_start:w_end, :])
    cv2.imwrite(dir_data+'proj_512/{:05d}.png'.format(i), proj[-res:, w_start:w_end, :])
