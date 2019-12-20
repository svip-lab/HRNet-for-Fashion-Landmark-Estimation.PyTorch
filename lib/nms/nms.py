# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .cpu_nms import cpu_nms
from .gpu_nms import gpu_nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    return _nms


def gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return gpu_nms(dets, thresh, device_id)
    return _nms


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        # # for COCO
        # sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        
        # # for deepfashion2
        sigmas = np.array([0.012 , 0.0158, 0.0169, 0.0165, 0.0169, 0.0158, 0.0298, 0.0329,
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

    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores


def soft_oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]
    scores = scores[order]

    # max_dets = order.size
    max_dets = 20
    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        order = order[1:]
        scores = rescore(oks_ovr, scores[1:], thresh)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep
    # kpts_db = kpts_db[:keep_cnt]

    # return kpts_db
