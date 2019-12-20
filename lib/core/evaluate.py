# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def OKS(pred, gt, gt_vis, area):
        if len(pred) == 0 or len(gt) == 0:
            return []
        sigmas = np.array([
            0.012 , 0.0158, 0.0169, 0.0165, 0.0169, 0.0158, 0.0298, 0.0329,
            0.0321, 0.0323, 0.034 , 0.0388, 0.0452, 0.0574, 0.0492, 0.0352,
            0.0492, 0.0574, 0.0452, 0.0388, 0.034 , 0.0323, 0.0321, 0.0329,
            0.0298, 0.0194, 0.017 , 0.0185, 0.0193, 0.0185, 0.017 , 0.0286,
            0.0471, 0.0547, 0.0526, 0.043 , 0.0392, 0.0513, 0.0566, 0.0509,
            0.0564, 0.0542, 0.0604, 0.0599, 0.052 , 0.0599, 0.0604, 0.0542,
            0.0564, 0.0509, 0.0566, 0.0513, 0.0392, 0.043 , 0.0526, 0.0547,
            0.0471, 0.0286, 0.0074, 0.0085, 0.0165, 0.0248, 0.0165, 0.0085,
            0.0156, 0.0231, 0.0296, 0.0137, 0.0195, 0.025 , 0.0347, 0.038 ,
            0.0257, 0.037 , 0.0257, 0.038 , 0.0347, 0.025 , 0.0195, 0.0137,
            0.0296, 0.0231, 0.0156, 0.0248, 0.0469, 0.0632, 0.037 , 0.0469,
            0.0632, 0.0137, 0.0153, 0.0243, 0.0377, 0.0243, 0.0153, 0.0203,
            0.0366, 0.0467, 0.0433, 0.0393, 0.0329, 0.0418, 0.0477, 0.0399,
            0.0331, 0.042 , 0.0492, 0.0436, 0.0478, 0.0436, 0.0492, 0.042 ,
            0.0331, 0.0399, 0.0477, 0.0418, 0.0329, 0.0393, 0.0433, 0.0467,
            0.0366, 0.0203, 0.0377, 0.0645, 0.0573, 0.0478, 0.0645, 0.0573,
            0.0352, 0.0158, 0.021 , 0.0214, 0.021 , 0.0158, 0.0196, 0.05  ,
            0.0489, 0.0404, 0.0401, 0.0404, 0.0489, 0.05  , 0.0196, 0.0276,
            0.0548, 0.0283, 0.0204, 0.0283, 0.0548, 0.0369, 0.0726, 0.0677,
            0.064 , 0.0251, 0.064 , 0.0677, 0.0726, 0.0369, 0.0308, 0.0216,
            0.0308, 0.0506, 0.0494, 0.0463, 0.0477, 0.0463, 0.0494, 0.0506,
            0.0275, 0.0202, 0.0275, 0.0651, 0.0451, 0.035 , 0.028 , 0.0392,
            0.0362, 0.0392, 0.028 , 0.035 , 0.0451, 0.0651, 0.0253, 0.0195,
            0.0253, 0.0513, 0.0543, 0.0415, 0.0543, 0.0513, 0.0153, 0.023 ,
            0.0167, 0.0145, 0.0167, 0.023 , 0.0332, 0.0391, 0.0391, 0.0396,
            0.044 , 0.0452, 0.0498, 0.0514, 0.0585, 0.0655, 0.0635, 0.0602,
            0.0635, 0.0655, 0.0585, 0.0514, 0.0498, 0.0452, 0.044 , 0.0396,
            0.0391, 0.0391, 0.0332, 0.0121, 0.0134, 0.0158, 0.0162, 0.0158,
            0.0134, 0.0246, 0.0406, 0.047 , 0.0404, 0.0463, 0.0466, 0.0435,
            0.0499, 0.0455, 0.044 , 0.0411, 0.049 , 0.0576, 0.0685, 0.0618,
            0.0483, 0.0618, 0.0685, 0.0576, 0.049 , 0.0411, 0.044 , 0.0486,
            0.0499, 0.0435, 0.0466, 0.0463, 0.0404, 0.047 , 0.0406, 0.0246,
            0.0116, 0.0167, 0.016 , 0.018 , 0.016 , 0.0167, 0.0196, 0.0385,
            0.0421, 0.0497, 0.0562, 0.0528, 0.0428, 0.0528, 0.0562, 0.0497,
            0.0421, 0.0385, 0.0196, 0.0244, 0.0297, 0.0244, 0.0208, 0.0244,
            0.0297, 0.0173, 0.0616, 0.0659, 0.0712, 0.0707, 0.0685, 0.0339,
            0.0685, 0.0707, 0.0712, 0.0659, 0.0616, 0.0173])

        vars = (sigmas * 2)**2
        k = len(sigmas)
        # create bounds for ignore regions(double the gt bbox)
        xg, yg = gt[:, 0], gt[:, 1]
        vg = gt_vis[:,0]
        k1 = np.count_nonzero(vg > 0)

        xd, yd = pred[:, 0], pred[:, 1]
        if k1>0:
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
        else:
            print('No visible points.')
            dx, dy = 0, 0
        e = (dx**2 + dy**2) /vars/ (area+np.spacing(1)) / 2
        if k1 > 0:
            e= e[vg > 0]
        
        oks = np.sum(np.exp(-e)) / e.shape[0]
        return oks

def meanOKS(pred, gt, gt_vis, area):
    batch_size = pred.shape[0]
    oks = []
    for i in range(batch_size):
        oks += [OKS(pred[i], gt[i], gt_vis[i], area[i])]
    return np.array(oks).mean(), oks, batch_size



def accuracy(output, target, target_type='gaussian', thr=0.5, h=96, w=72):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if target_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    elif target_type == 'coordinate':
        pred = output
        target = target
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


