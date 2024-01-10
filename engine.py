# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
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
import math
import os
import sys
from typing import Iterable

import torch
import torch.nn as nn
import util.misc as utils

from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from models.utils import compute_delta, mixup_imgs, mixup_domain_labels


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    postprocessors, cfg=None, **kwargs):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('delta', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    data_loader_len = len(data_loader)

    delta = 0
    current_iters = epoch * data_loader_len
    total_iters = cfg.TRAIN.EPOCHS * data_loader_len
    for iter_i in metric_logger.log_every(range(data_loader_len), print_freq, header):
        current_iters += 1
        delta = compute_delta(current_iters, total_iters, alpha=cfg.BLENDA.ALPHA, beta=cfg.BLENDA.BETA, min_delta=cfg.BLENDA.MIN_DELTA)

        src_img, int_img, tgt_img = samples.tensors
        src_target, tgt_target = targets
        if cfg.BLENDA.MIXUP_SRC_INT_IMGS:
            int_img = mixup_imgs(src_img, int_img, mixup_ratio=delta)
            int_img, src_target = data_loader.dataset.transform(int_img, src_target)
        if cfg.BLENDA.MIXUP_SRC_TGT_IMGS:
            tgt_img = mixup_imgs(src_img, tgt_img, mixup_ratio=delta)
            tgt_img, tgt_target = data_loader.dataset.transform(tgt_img, tgt_target)
        samples = utils.nested_tensor_from_tensor_list([int_img, tgt_img])
        targets = [src_target, tgt_target]

        domain_labels = [0, 1]
        if cfg.BLENDA.MIXUP_SRC_INT_DOMAIN_LABELS:
            int_domain_label = cfg.BLENDA.INT_DOMAIN_LABEL
            if int_domain_label == 0:
                raise ValueError('The interpolation domain label can not be the same with the src domain label')
            domain_labels[0] = mixup_domain_labels(0, int_domain_label, delta)
        if cfg.BLENDA.MIXUP_SRC_TGT_DOMAIN_LABELS:
            domain_labels[1] = mixup_domain_labels(0, 1, delta)

        outputs = model(samples)
        loss_dict = criterion(outputs, targets[:2], domain_labels)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

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
        if cfg.TRAIN.CLIP_MAX_NORM > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_MAX_NORM)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), cfg.TRAIN.CLIP_MAX_NORM)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(delta=delta)

        samples, targets = prefetcher.next()
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, outputs


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, cfg, **kwargs):
    """
    data_loader.dataset: `CocoDetection`, not `DADataset`
    base_ds: `pycocotools.coco.COCO`
    """

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluators_per_class = {}
    for cat_id in base_ds.getCatIds():
        if 'category_ids' in kwargs:
            if cat_id not in kwargs['category_ids']:
                continue
        evaluator = CocoEvaluator(base_ds, iou_types)
        for iou_type in iou_types:
            evaluator.coco_eval[iou_type].params.catIds = [cat_id]
        coco_evaluators_per_class[cat_id] = evaluator

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
        if coco_evaluators_per_class is not None:
            for evaluator in coco_evaluators_per_class.values():
                evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if coco_evaluators_per_class is not None:
        for evaluator in coco_evaluators_per_class.values():
            evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        print('=== Overall mAP ===')
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    if coco_evaluators_per_class is not None:
        for cat_id, evaluator in coco_evaluators_per_class.items():
            cat_name = base_ds.cats[cat_id]['name']
            print(f'=== Class mAP ({cat_id}, {cat_name}) ===')
            evaluator.accumulate()
            evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            ratio = 1
            if 'category_ids' in kwargs:
                ratio = (cfg.DATASET.NUM_CLASSES - 1) / len(cfg.DATASET.CATEGORY_IDS)
            stats['coco_eval_bbox'] = [v * ratio for v in coco_evaluator.coco_eval['bbox'].stats.tolist()]
            for cat_id, evaluator in coco_evaluators_per_class.items():
                cat_name = base_ds.cats[cat_id]['name']
                k = f'coco_eval_bbox_cat-id={cat_id}_cat-name={cat_name}'
                stats[k] = evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
