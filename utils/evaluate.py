import numpy as np

# Normalize patch size
p = 24
patch_rate = 50

def norm_diff(pred, gt, sp, mask):
    shapes = pred.shape
    normed = np.zeros_like(pred)
    mask[:p, :] = 0
    mask[shapes[0]-p:, :] = 0
    mask[:, :p] = 0
    mask[:, shapes[0]-p:] = 0
    new_mask = mask.copy()

    # diff_pred = (pred - sp)*mask
    diff_pred = pred * mask
    diff_gt = (gt - sp)*mask
    cnt = 0
    for i in range(p, shapes[0]-p):
        for j in range(p, shapes[1]-p):
            cnt += 1
            if not mask[i,j]:
                normed[i,j] = 0
                continue

            local_mask = mask[i-p:i+p+1, j-p:j+p+1]
            local_gt = diff_gt[i-p:i+p+1, j-p:j+p+1]
            local_pred = diff_pred[i-p:i+p+1, j-p:j+p+1]
            local_mask_len = np.sum(local_mask)
            patch_len = (p*2 + 1) ** 2
            if local_mask_len < patch_len*patch_rate/100:
                normed[i, j] = 0
                new_mask[i, j] = 0
                continue
            local_mean_gt = np.sum(local_gt) / local_mask_len
            local_mean_pred = np.sum(local_pred) / local_mask_len
            local_sd_gt = np.sqrt(np.sum(np.square(local_gt - local_mean_gt)) / local_mask_len)
            local_sd_pred = np.sqrt(np.sum(np.square(local_pred - local_mean_pred)) / local_mask_len)
            normed[i, j] = (diff_pred[i, j] - local_mean_pred) * (local_sd_gt / local_sd_pred) + local_mean_gt

    return normed+sp, new_mask