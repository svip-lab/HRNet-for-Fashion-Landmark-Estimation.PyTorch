# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, output, center, scale, coord_heatmaps=None):
    heatmap_height = config.MODEL.HEATMAP_SIZE[1]
    heatmap_width = config.MODEL.HEATMAP_SIZE[0]

    if config.MODEL.TARGET_TYPE == 'gaussian':
        batch_heatmaps = output
        coords, maxvals = get_max_preds(batch_heatmaps)
        # post-processing
        if config.TEST.POST_PROCESS:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                        diff = np.array(
                            [
                                hm[py][px+1] - hm[py][px-1],
                                hm[py+1][px]-hm[py-1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25

    elif config.MODEL.TARGET_TYPE == 'coordinate':
        coords= output
        batch_size, num_kpoints, _ = coords.shape

        idx = np.round(coords.reshape(-1, 2)).astype(np.int)
        coord_heatmaps = coord_heatmaps.reshape(-1, heatmap_height, heatmap_width)
        maxvals = []
        for i, heatmap in enumerate(coord_heatmaps):
            maxvals.append(heatmap[idx[i][1], idx[i][0]])

        maxvals = np.array(maxvals).reshape(batch_size, num_kpoints, 1)
            
    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )
    return preds, maxvals

def get_original_gts(config, output, center, scale):
    heatmap_height = config.MODEL.HEATMAP_SIZE[1]
    heatmap_width = config.MODEL.HEATMAP_SIZE[0]

    coords, maxvals = output, 1
    
    gts = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):
        gts[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )
    return gts

