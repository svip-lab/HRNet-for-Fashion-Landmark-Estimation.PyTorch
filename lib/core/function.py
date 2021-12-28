# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back, transform_preds
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, train_dataset, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True).float()
        target_weight = target_weight.cuda(non_blocking=True).float()

        cat_ids = meta['category_id']
        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()

        channel_mask = torch.zeros_like(target_weight).float()
        for j, cat_id in enumerate(cat_ids):
            rg = train_dataset.gt_class_keypoints_dict[int(cat_id)]
            index = torch.tensor([list(range(rg[0], rg[1]))], device=channel_mask.device, dtype=channel_mask.dtype).transpose(1,0).long()
            channel_mask[j].scatter_(0, index, 1)

        # compute output
        output = model(input)
        
        if config.MODEL.TARGET_TYPE == 'gaussian':
            # block irrelevant channels in output
            output = output * channel_mask.unsqueeze(3)
            preds, maxvals = get_final_preds(config, output.detach().cpu().numpy(), c, s)
        
        elif config.MODEL.TARGET_TYPE == 'coordinate':
            heatmap, output = output
            
            # block irrelevant channels in output
            output = output * channel_mask
            preds_local, maxvals = get_final_preds(config, output.detach().cpu().numpy(), c, s, heatmap.detach().cpu().numpy())

            # Transform back from heatmap coordinate to image coordinate
            preds = preds_local.copy()
            for i in range(preds_local.shape[0]):
                preds[i] = transform_preds(
                    preds_local[i], c[i], s[i], 
                    [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]]
                )
        else:
            raise NotImplementedError('{} is not implemented'.format(config.MODEL.TARGET_TYPE))
        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), 
                                        target.detach().cpu().numpy(),
                                        train_dataset.target_type)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0)/batch_time.val,
                    data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

            save_debug_images(config, input, meta, target, preds_local, output, prefix)
        

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 7))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            
            target = target.cuda(non_blocking=True).float()
            target_weight = target_weight.cuda(non_blocking=True).float()

            cat_ids = meta['category_id']
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            channel_mask = torch.zeros_like(target_weight).float()
            for j, cat_id in enumerate(cat_ids):
                rg = val_dataset.gt_class_keypoints_dict[int(cat_id)]
                index = torch.tensor([list(range(rg[0], rg[1]))], device=channel_mask.device, dtype=channel_mask.dtype).transpose(1,0).long()
                channel_mask[j].scatter_(0, index, 1)
                
            # compute output
            output = model(input)

            if config.MODEL.TARGET_TYPE == 'gaussian':
                if config.TEST.FLIP_TEST:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped
                    output_flipped = output_flipped.cpu().numpy()
                    
                    category_id_list = meta['category_id'].cpu().numpy().copy()
                    for j, category_id in enumerate(category_id_list):
                        output_flipped[j, :, :, :] = flip_back(output_flipped[j, None],
                                                val_dataset.flip_pairs[category_id-1],
                                                config.MODEL.HEATMAP_SIZE[0])
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                # block irrelevant channels in output
                output = output * channel_mask.unsqueeze(3)
                preds_local, maxvals = get_final_preds(config, output.detach().cpu().numpy(), c, s)

                # Transform back from heatmap coordinate to image coordinate
                preds = preds_local.copy()
                for i in range(preds_local.shape[0]):
                    preds[i] = transform_preds(
                        preds_local[i], c[i], s[i], 
                        [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]]
                    )
                
            else:
                raise NotImplementedError('{} is not implemented'.format(config.MODEL.TARGET_TYPE))
            
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), 
                                        target.detach().cpu().numpy(),
                                        val_dataset.target_type)
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            all_boxes[idx:idx + num_images, 6] = meta['category_id'].cpu().numpy().astype(int)

            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )

                save_debug_images(config, input, meta, target, preds_local, output, prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            # if isinstance(name_values, list):
            #     for name_value in name_values:
            #         writer.add_scalars(
            #             'valid',
            #             dict(name_value),
            #             global_steps
            #         )
            # else:
            #     writer.add_scalars(
            #         'valid',
            #         dict(name_values),
            #         global_steps
            #     )
            writer.add_scalar(
                'valid_AP',
                perf_indicator,
                global_steps
            )
            
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.4f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
