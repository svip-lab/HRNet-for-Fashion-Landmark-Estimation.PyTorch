# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, cfg, target_type, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.cfg = cfg
        self.criterion = nn.MSELoss(reduction='sum')
        self.use_target_weight = use_target_weight
        self.target_type = target_type

    def forward(self, output, target, target_weight):
        if self.target_type == 'gaussian':
            '''Modified for DeepFashion Landmark Detection Task, without for loop'''
            batch_size = output.size(0)
            num_kpoints = output.size(1)
            num_pixels = output.size(2) * output.size(3)

            heatmaps_pred = output.reshape((batch_size, num_kpoints, -1))
            heatmaps_gt = target.reshape((batch_size, num_kpoints, -1))

            sum_MSE = ((((heatmaps_pred - heatmaps_gt).mul(target_weight)) ** 2).reshape(batch_size, -1) / num_pixels).sum(dim=1, keepdim=True)
            MSE = (sum_MSE / target_weight.sum(1)).sum() / batch_size
        elif self.target_type == 'coordinate':
            batch_size = output.size(0)
            w, h = self.cfg.MODEL.HEATMAP_SIZE
            sum_MSE = ((output - target)/(h+w) * target_weight) ** 2
            sum_MSE = (sum_MSE.reshape(batch_size, -1) / 2).sum(dim=1, keepdim=True)
            MSE = (sum_MSE / target_weight.sum(1)).sum() / batch_size

        return MSE

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
