# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
import cv2
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from detectron2.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
from .backbone import build_backbone
from .matcher import build_matcher
from .motr import FrameMatcherV2, E2EMotDETRV3, TrackerPostProcess
from .deformable_transformer_plus import build_deforamble_transformer
from .track_embedding import build as build_track_embedding_layer
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss
from .dct import ProcessorDCT

class MultiClipFrameMatcherV2(FrameMatcherV2):
    def clear(self):
        self.num_samples = 0

    def initialize_for_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        # self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

class MultiClipE2EMotDETRV3(E2EMotDETRV3):
    def forward(self, data):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        clips: List = data['imgs']  # list of Tensor.
        gt_instances = data['gt_instances']
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        self.criterion.clear()
        assert self.training
        losses_dict = {}
        num_clips = len(clips)
        for i in range(num_clips):
            clip = clips[i]
            gt_instances_i = gt_instances[i]
            track_instances = self._generate_empty_tracks()
            if self.training:
                self.criterion.initialize_for_clip(gt_instances_i)
            for frame in clip:
                if not isinstance(frame, NestedTensor):
                    frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances)
                track_instances = frame_res['track_instances']
                for loss_name, loss in self.criterion.losses_dict.items():
                    losses_dict['clip{}_{}'.format(i, loss_name)] = loss

        outputs['losses_dict'] = losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'detmot': 1,
        'e2e_mot': 1,
        'ori_mot': 1,
        'e2e_static_mot': 1,
        'e2e_joint': 1,
        'ytvos': 40,
        'post_ytvos': 40,
        'old_ytvos': 40,
        'dev_old_ytvos': 40,
        'mosaic_ytvos': 40
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    if args.occlusion_class:
        # three classes: (positive track, occluded track, empty-track)
        num_classes += 1
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)

    if args.with_vector:
        processor_dct = ProcessorDCT(n_keep=args.n_keep, gt_mask_len=args.gt_mask_len)

    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    track_embedding_layer = build_track_embedding_layer(args, args.track_embedding_layer, d_model, hidden_dim, d_model*2)

    matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            })

        if args.with_vector:
            weight_dict.update({'frame_{}_loss_vector'.format(i): 1})

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })

                if args.with_vector:
                    print(f'Training with vector aux loss of decoder layer {args.vector_aux_loss_layer}.')
                    if j in args.vector_aux_loss_layer:
                        weight_dict.update({'frame_{}_aux{}_loss_vector'.format(i, j): 1})
                    else:
                        weight_dict.update({'frame_{}_aux{}_loss_vector'.format(i, j): 0})
    all_weight_dict = {}
    for i in range(args.batch_size):
        for key in list(weight_dict.keys()):
            all_weight_dict['clip{}_{}'.format(i, key)] = weight_dict[key]
    losses = ['labels', 'boxes']
    if args.masks:
        losses += ["masks"]
    criterion = MultiClipFrameMatcherV2(num_classes, matcher=matcher, weight_dict=all_weight_dict, losses=losses, random_drop=args.random_drop,
                                            with_vector=args.with_vector, 
                                            vector_loss_coef=args.vector_loss_coef,
                                            vector_loss_norm=args.vector_loss_norm)
    criterion.to(device)
    postprocessors = {}
    model = MultiClipE2EMotDETRV3(
        backbone,
        transformer,
        track_embed=track_embedding_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        with_vector=args.with_vector, 
        vector_hidden_dim=args.vector_hidden_dim,
        processor_dct=processor_dct if args.with_vector else None
    )
    return model, criterion, postprocessors
