# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import cv2
import math
import numpy as np
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from util import box_ops

from torch import Tensor
from util.events import get_event_storage, EventStorage, TensorboardXWriter
from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda


def visualize_single_layer_outputs(storage, outputs, targets, layer: int, img_name: str):
    img = targets[0]['ori_img'].permute(1, 2, 0).contiguous()
    h, w = img.shape[:2]
    gt_boxes = box_ops.box_cxcywh_to_xyxy(targets[0]['boxes'])
    gt_boxes[:, ::2] *= w
    gt_boxes[:, 1::2] *= h
    # gt_boxes = targets[0]['ori_boxes']
    vis_img = draw_boxes(img, gt_boxes, color=(0, 1, 0))

    # dt_boxes = outputs['pred_boxes'][0].detach().clone()
    dt_boxes = outputs['boxes_all'][layer, 0].detach().clone()
    # dt_scores = outputs['pred_logits'][0].detach().clone().sigmoid().max(dim=-1)[0]
    dt_scores = outputs['logits_all'][layer, 0].detach().clone().sigmoid().max(dim=-1)[0]

    keep = dt_scores > 0.4
    dt_boxes = dt_boxes[keep]
    dt_scores = dt_scores[keep]

    if 'ref_pts' in outputs:
        ref_pts = outputs['ref_pts'][layer, 0, :, :2].detach().clone()
        ref_pts[:, 0] *= w
        ref_pts[:, 1] *= h
        ref_pts = torch.cat([ref_pts, keep.unsqueeze(-1).to(torch.float32)], dim=-1)
        print("ref_pts.shape={}".format(ref_pts.shape))
        draw_ref_pts(vis_img, ref_pts)
    # if len(dt_boxes) > 0:
    print("dt_boxes.shape={}".format(dt_boxes.shape))
    dt_boxes[:, 0::2] *= w
    dt_boxes[:, 1::2] *= h
    # print("gt_boxes={} dt_boxes={}".format(gt_boxes, dt_boxes))
    dt_boxes = box_ops.box_cxcywh_to_xyxy(dt_boxes)
    vis_img = draw_boxes(vis_img, dt_boxes, color=(0, 0, 1),
                         texts=['{:3f}'.format(score.item()) for score in dt_scores])
    vis_img = image_hwc2chw(vis_img)
    storage.put_image(img_name, vis_img)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer: TensorboardXWriter=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    storage = get_event_storage()
    step = 0

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
        step += 1
        if writer is not None:
            # if step % 20 == 1 and 'ori_img' in targets[0]:
            #     for i in range(6):
            #         visualize_single_layer_outputs(storage, outputs, targets, layer=i, img_name='vis_layer_{}'.format(i))
            storage.put_scalar("loss", loss_value)
            for loss_name, loss_value in loss_dict_reduced_scaled.items():
                storage.put_scalar(loss_name, loss_value.item())
            for loss_name, loss_value in loss_dict_reduced_unscaled.items():
                storage.put_scalar(loss_name, loss_value.item())
            writer.write()
            if storage is not None:
                storage.step()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    storage = get_event_storage()
    step = 0

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)


        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        # gather the stats from all processes
        if writer is not None:
            if step % 20 == 0:
                img = data_dict['ori_img'][0].permute(1, 2, 0).contiguous()
                h, w = img.shape[:2]
                gt_boxes = box_ops.box_cxcywh_to_xyxy(data_dict['gt_instances'][0].boxes)
                gt_boxes[:, ::2] *= w
                gt_boxes[:, 1::2] *= h
                vis_img = draw_boxes(img, gt_boxes, color=(0, 1, 0))
                dt_boxes = outputs['pred_boxes'][0].detach().clone()
                dt_scores = outputs['pred_logits'][0].detach().clone().sigmoid().max(dim=-1)[0]
                keep = dt_scores > 0.4
                dt_boxes = dt_boxes[keep]
                dt_scores = dt_scores[keep]
                dt_boxes[:, 0::2] *= w
                dt_boxes[:, 1::2] *= h
                # print("gt_boxes={} dt_boxes={}".format(gt_boxes, dt_boxes))
                dt_boxes = box_ops.box_cxcywh_to_xyxy(dt_boxes)
                vis_img = draw_boxes(vis_img, dt_boxes, color=(0, 0, 1), texts=[str(score.item()) for score in dt_scores])
                vis_img = image_hwc2chw(vis_img)
                storage.put_image('image_with_gt', vis_img)
            storage.put_scalar("loss", loss_value)
            for loss_name, loss_value in loss_dict_reduced_scaled.items():
                storage.put_scalar(loss_name, loss_value.item())
            # for loss_name, loss_value in loss_dict_reduced_unscaled.items():
            #     storage.put_scalar(loss_name, loss_value.item())
            writer.write()
        if storage is not None:
            storage.step()
        step += 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             )
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
