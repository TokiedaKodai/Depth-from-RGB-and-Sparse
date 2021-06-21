import numpy as np
import cv2
import pandas as pd
import sys
import os
import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from utils.depth_tools import unpack_bmp_bgra_to_float
from utils.evaluate import norm_diff, evaluate_rmse
import config

dir_data = config.dir_data + 'cardboard/'
dir_supres = config.dir_result + '/supres_board_512/'
# dir_d3 = '../../sp_models/test/pred_board-512/'
# dir_d3 = '../../sp_models/test/finetune/pred_board-512/'
# dir_d3 = '../../sp_models/wave2_600/finetune/pred_board/'
dir_d3 = '../../sp_models/wave2_600_depth/finetune/pred_board/'
dir_d3 = '../../sp_models/nyu/finetune/pred_board/'
# dir_ours = '../../output/output_wave2-pose_lumi/predict_400_cardboard_norm-local-pix=24_rate=50_crop=2_vloss_min/'
dir_ours = '../../output/output_wave2-pose_lumi_FT_s2/predict_450_cardboard_norm-local-pix=24_rate=50_crop=2_vloss_min/'
idxs = list(range(44, 56))
dir_save = config.dir_result + '/board_depth_supres512/'
dir_save = config.dir_result + '/board_depth_supres512_nyu/'

# dir_data = config.dir_data + 'reals/'
# dir_supres = config.dir_result + '/supres_real_512/'
# dir_d3 = '../../sp_models/test/pred_real-512/'
# # dir_d3 = '../../sp_models/test/finetune/pred_real-512/'
# dir_d3 = '../../sp_models/wave2_600/finetune/pred_real/'
# dir_d3 = '../../sp_models/wave2_600_depth/finetune/pred_real/'
# dir_d3 = '../../sp_models/nyu/finetune/pred_real/'
# # dir_ours = '../../output/output_wave2-pose_lumi/predict_400_reals_norm-local-pix=24_rate=50_crop=2_vloss_min/'
# dir_ours = '../../output/output_wave2-pose_lumi_FT_s2/predict_450_reals_norm-local-pix=24_rate=50_crop=2_vloss_min/'
# idxs = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 17, 18]
# dir_save = config.dir_result + '/real_depth_supres512_nyu/'

img_res = 512
res = 256
edge = 25
edge = 128

depth_range = 0.02
depth_range = 0.01
err_range = 0.002

list_rmse_rec = []
list_rmse_supres = []
list_rmse_d3 = []
list_rmse_ours = []

os.makedirs(dir_save, exist_ok=True)

for i, idx in enumerate(tqdm(idxs)):
    shade = cv2.imread(dir_data+'shade_512/{:05d}.png'.format(idx), 1)
    gt = cv2.imread(dir_data+'gt_512/{:05d}.bmp'.format(idx), -1)
    rec = cv2.imread(dir_data+'rec_512/{:05d}.bmp'.format(idx), -1)
    gt = unpack_bmp_bgra_to_float(gt)
    rec = unpack_bmp_bgra_to_float(rec)

    pred_supres = cv2.imread(dir_supres+'{:03d}.bmp'.format(i), -1)
    pred_supres = unpack_bmp_bgra_to_float(pred_supres)

    pred_d3 = cv2.imread(dir_d3+'pred_{:03d}.bmp'.format(i), -1)
    pred_d3 = unpack_bmp_bgra_to_float(pred_d3)

    pred_ours = cv2.imread(dir_ours+'predict-{:03d}.bmp'.format(idx), -1)
    pred_ours = unpack_bmp_bgra_to_float(pred_ours)

    mask_supres = (pred_supres > 0.1)*1.0
    mask_d3 = (pred_d3 > 0.1)*1.0
    mask_ours = (pred_ours > 0.1)*1.0
    mask = mask_supres*mask_d3*mask_ours

    shade = shade[edge:-edge, edge:-edge, :]
    gt = gt[edge:-edge, edge:-edge]
    rec = rec[edge:-edge, edge:-edge]
    mask = mask[edge:-edge, edge:-edge]
    pred_supres = pred_supres[edge:-edge, edge:-edge]
    pred_d3 = pred_d3[edge:-edge, edge:-edge]
    pred_ours = pred_ours[edge:-edge, edge:-edge]
    # err_rec = err_rec[edge:-edge, edge:-edge]
    # err_supres = err_supres[edge:-edge, edge:-edge]
    # err_d3 = err_d3[edge:-edge, edge:-edge]
    # err_ours = err_ours[edge:-edge, edge:-edge]

    gt = gt*mask
    rec = rec*mask
    pred_supres = pred_supres*mask
    pred_d3 = pred_d3*mask
    pred_ours = pred_ours*mask

    rmse_rec = evaluate_rmse(rec, gt, mask)
    rmse_supres = evaluate_rmse(pred_supres, gt, mask)
    rmse_d3 = evaluate_rmse(pred_d3, gt, mask)
    rmse_ours = evaluate_rmse(pred_ours, gt, mask)
    list_rmse_rec.append(rmse_rec)
    list_rmse_supres.append(rmse_supres)
    list_rmse_d3.append(rmse_d3)
    list_rmse_ours.append(rmse_ours)

    err_rec = np.abs(rec - gt)
    err_supres = np.abs(pred_supres - gt)
    err_d3 = np.abs(pred_d3 - gt)
    err_ours = np.abs(pred_ours - gt)

    dir_save_each = dir_save + '{:03d}/'.format(idx)
    os.makedirs(dir_save_each, exist_ok=True)

    with open(dir_save_each+'rmse-{:03d}.txt'.format(idx), mode='w') as f:
        f.write(f'Low-res,Super-res,D3,Ours\n{rmse_rec},{rmse_supres},{rmse_d3},{rmse_ours}')

    mean_gt = np.sum(gt) / np.sum(mask)
    v_min, v_max = mean_gt-depth_range, mean_gt+depth_range
    e_max = err_range

    # dir_save += '{:03d}/'.format(idx)
    # os.makedirs(dir_save)

    # cv2.imwrite(dir_save+'shade.png', shade)
    # cv2.imwrite(dir_save+'shade.png', gt)

    fig = plt.figure(figsize=(16, 6))
    plt.rcParams["font.size"] = 18
    gs_master = GridSpec(nrows=2,
                        ncols=2,
                        height_ratios=[1, 1],
                        width_ratios=[5, 0.1],
                        wspace=0.05,
                        hspace=0.05)
    gs_1 = GridSpecFromSubplotSpec(nrows=1,
                                ncols=5,
                                subplot_spec=gs_master[0, 0],
                                wspace=0.05,
                                hspace=0)
    gs_2 = GridSpecFromSubplotSpec(nrows=1,
                                ncols=5,
                                subplot_spec=gs_master[1, 0],
                                wspace=0.05,
                                hspace=0)
    gs_3 = GridSpecFromSubplotSpec(nrows=2,
                                ncols=1,
                                subplot_spec=gs_master[0:1, 1],
                                wspace=0,
                                hspace=0.1)
    ax_enh0 = fig.add_subplot(gs_1[0, 0])
    ax_enh1 = fig.add_subplot(gs_1[0, 1])
    ax_enh2 = fig.add_subplot(gs_1[0, 2])
    ax_enh3 = fig.add_subplot(gs_1[0, 3])
    ax_enh4 = fig.add_subplot(gs_1[0, 4])

    ax_misc0 = fig.add_subplot(gs_2[0, 0])
    ax_err_rec = fig.add_subplot(gs_2[0, 1])
    ax_err_supres = fig.add_subplot(gs_2[0, 2])
    ax_err_d3 = fig.add_subplot(gs_2[0, 3])
    ax_err_ours = fig.add_subplot(gs_2[0, 4])

    ax_cb0 = fig.add_subplot(gs_3[0, 0])
    ax_cb1 = fig.add_subplot(gs_3[1, 0])

    for ax in [
            ax_enh0, ax_enh1, ax_enh2, ax_enh3, ax_enh4,
            ax_misc0, ax_err_rec, ax_err_supres, ax_err_d3, ax_err_ours
    ]:
        ax.axis('off')

    ''' Depth '''
    ax_enh0.imshow(gt, cmap='jet', vmin=v_min, vmax=v_max)
    ax_enh1.imshow(rec, cmap='jet', vmin=v_min, vmax=v_max)
    ax_enh2.imshow(pred_supres, cmap='jet', vmin=v_min, vmax=v_max)
    ax_enh3.imshow(pred_d3, cmap='jet', vmin=v_min, vmax=v_max)
    ax_enh4.imshow(pred_ours, cmap='jet', vmin=v_min, vmax=v_max)

    ax_enh0.set_title('Ground Truth')
    ax_enh1.set_title('Low-res')
    # ax_enh2.set_title('Super-res')
    ax_enh2.set_title('[10]')
    # ax_enh3.set_title('D3')
    ax_enh3.set_title('[13]')
    ax_enh4.set_title('Ours')

    ''' Shading '''
    ax_misc0.imshow(shade[:, :, ::-1])
    ''' Error map '''
    scale = 1000
    e_max *= scale
    ax_err_rec.imshow(err_rec*scale, cmap='jet', vmin=0, vmax=e_max)
    ax_err_supres.imshow(err_supres*scale, cmap='jet', vmin=0, vmax=e_max)
    ax_err_d3.imshow(err_d3*scale, cmap='jet', vmin=0, vmax=e_max)
    ax_err_ours.imshow(err_ours*scale, cmap='jet', vmin=0, vmax=e_max)

    ''' Colorbar '''
    # plt.tight_layout()
    fig.savefig(io.BytesIO())
    # cb_offset = -0.05
    cb_offset = 0

    plt.colorbar(ScalarMappable(colors.Normalize(vmin=v_min, vmax=v_max),
                                cmap='jet'),
                cax=ax_cb0)
    im_pos, cb_pos = ax_enh0.get_position(), ax_cb1.get_position()
    ax_cb0.set_position([
        cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
        im_pos.y1 - im_pos.y0
    ])
    ax_cb0.set_ylabel('Depth [m]')

    plt.colorbar(ScalarMappable(colors.Normalize(vmin=0, vmax=e_max),
                                cmap='jet'),
                cax=ax_cb1)
    im_pos, cb_pos = ax_err_ours.get_position(), ax_cb1.get_position()
    ax_cb1.set_position([
        cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
        im_pos.y1 - im_pos.y0
    ])
    ax_cb1.set_ylabel('Error [mm]')

    ''' Save '''
    plt.savefig(dir_save + '{:03d}.png'.format(idx), dpi=300)
    plt.savefig(dir_save_each + '{:03d}.pdf'.format(idx), dpi=300)
    plt.savefig(dir_save_each + '{:03d}.svg'.format(idx), dpi=300)
    plt.close()

result = pd.DataFrame({
    'Index': idxs,
    'Low-res': list_rmse_rec,
    'Super-res': list_rmse_supres,
    'D3': list_rmse_d3,
    'Ours': list_rmse_ours
})
result.set_index('Index', inplace=True)
result.to_csv(dir_save+'rmse-result.csv', mode='w', header=True)