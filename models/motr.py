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
from .deformable_transformer_plus import build_deforamble_transformer
from .track_embedding import build as build_track_embedding_layer
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss
from .dct import ProcessorDCT


class FrameMatcherV2(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        random_drop=0,
                        with_vector=False, 
                        vector_loss_coef=2,
                        vector_loss_norm=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.random_drop = random_drop
        self.with_vector = with_vector
        self.vector_loss_coef = vector_loss_coef
        self.vector_loss_norm = vector_loss_norm

        print(f"Training with vector_loss_coef {self.vector_loss_coef}.")
        if self.vector_loss_norm:
            print('Training with vector_loss_norm.')

    def initialize(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for track-disappear slot is set -1.
        # Thus, their labels is positive.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            # set labels to 1(means track-disappear).
            # labels_per_img = torch.ones_like(J)
            labels_per_img = torch.full_like(J, self.num_classes) # fixed
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :,
                               :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_masks(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        # We ignore the regression loss of the track-disappear slots.
        # TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_vectors = outputs['pred_vectors'][idx]
        target_vectors = torch.cat([gt_per_img.masks[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether (obj_id = -1)
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        mask = (target_obj_ids != -1)

        losses = {}
        loss_vector = F.l1_loss(src_vectors[mask], target_vectors[mask], reduction='none')
        if self.vector_loss_norm:
            losses['loss_vector'] = self.vector_loss_coef * loss_vector.mean(dim=1).sum() / num_boxes
        else:
            losses['loss_vector'] = self.vector_loss_coef * loss_vector.sum() / num_boxes

        return losses

    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        def select_unmatched_indexes(matched_indexes: torch.Tensor, num_total_indexes: int) -> torch.Tensor:
            matched_indexes_set = set(matched_indexes.detach().cpu().numpy().tolist())
            all_indexes_set = set(list(range(num_total_indexes)))
            unmatched_indexes_set = all_indexes_set - matched_indexes_set
            unmatched_indexes = torch.as_tensor(list(unmatched_indexes_set), dtype=torch.long).to(matched_indexes)
            return unmatched_indexes

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        if self.with_vector:
            pred_vectors_i = track_instances.pred_vectors # predicted vectors of i-th image.
            outputs_i.update({'pred_vectors': pred_vectors_i.unsqueeze(0)})

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        valid_track_mask = track_instances.obj_idxes >= 0
        valid_track_idxes = torch.arange(len(track_instances), device=pred_logits_i.device)[valid_track_mask]
        valid_obj_idxes = track_instances.obj_idxes[valid_track_idxes]
        for j in range(len(valid_obj_idxes)):
            obj_id = valid_obj_idxes[j].item()
            if obj_id in obj_idx_to_gt_idx:
                track_instances.matched_gt_idxes[valid_track_idxes[j]] = obj_idx_to_gt_idx[obj_id]
            else:
                num_disappear_track += 1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        # matched_track_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0) # no occu 
        # prev_track = track_instances[full_track_idxes[matched_track_idxes]]
        # print("frame={} {} prev tracks shape={} {}".format(self._current_frame_idx, len(prev_track), (prev_track.pred_boxes, prev_track.pred_logits.sigmoid()), gt_instances_i[track_instances.matched_gt_idxes[matched_track_idxes]].boxes))
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the fp tracks (obj_idxes == -2) will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the unmatched gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        unmatched_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        unmatched_gt_instances = gt_instances_i[unmatched_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [unmatched_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            # map the matched pair indexes to original index-space.
            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt for loss calculation.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], unmatched_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)
        # new_track = track_instances[new_matched_indices[:, 0]]
        # print("new_track{} = {}".format(len(new_track), (new_track.pred_boxes, new_track.pred_logits.sigmoid())))
        # print("new gt={}".format(gt_instances_i.boxes[new_matched_indices[:, 1]]))

        # step6. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step7. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step8. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step9. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    # if loss == 'masks':
                    #     # Intermediate masks losses are too costly to compute, we ignore them.
                    #     continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    # print("aux_{}_loss: {}".format(i, l_dict))
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, is_repeat=False):
        # print("max_score={}".format(track_instances.scores.max()))
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh and is_repeat is False:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, processor_dct=None):
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes
        if self.processor_dct is not None:
            out_vector = track_instances.pred_vectors

        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        # if self.processor_dct is not None:
        #     n_keep = self.processor_dct.n_keep
        #     vectors = out_vector
        #     masks = []
        #     n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
        #     r, c = vectors.shape
        #     outputs_masks_per_image = []
        #     for ri in range(r):
        #         # here visual for training
        #         idct = np.zeros((gt_mask_len ** 2))
        #         idct[:n_keep] = vectors[ri].cpu().numpy()
        #         idct = self.processor_dct.inverse_zigzag(idct, gt_mask_len, gt_mask_len)
        #         re_mask = cv2.idct(idct)
        #         max_v = np.max(re_mask)
        #         min_v = np.min(re_mask)
        #         re_mask = np.where(re_mask>(max_v+min_v) / 2., 1, 0)
        #         re_mask = torch.from_numpy(re_mask)[None].float()
        #         outputs_masks_per_image.append(re_mask)
        #     outputs_masks_per_image = torch.cat(outputs_masks_per_image, dim=0).to(out_vector.device)
        #     # here padding local mask to global mask
        #     outputs_masks_per_image = retry_if_cuda_oom(paste_masks_in_image)(
        #         outputs_masks_per_image,  # N, 1, M, M
        #         boxes,
        #         (img_h, img_w),
        #         threshold=0.5,
        #     )
        #     outputs_masks_per_image = outputs_masks_per_image.unsqueeze(1).cpu()

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        if self.processor_dct is not None:
            # track_instances.masks = outputs_masks_per_image
            track_instances.vectors = out_vector

        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class E2EMotDETRV3(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 with_vector=False, vector_hidden_dim=256, n_keep=256, gt_mask_len=128,
                 processor_dct=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        # mask settings
        self.with_vector = with_vector
        self.vector_hidden_dim = vector_hidden_dim
        self.n_keep = n_keep
        self.gt_mask_len = gt_mask_len

        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # init for mask head
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.n_keep, 3)

            # init
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)

            # aux head
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])

        self.post_process = TrackerPostProcess(processor_dct=processor_dct if self.with_vector else None)
        self.processor_dct = processor_dct
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embed.weight.shape
        device = self.query_embed.weight.device
        track_instances.ref_pts = self.transformer.reference_points(self.query_embed.weight[:, :dim // 2])
        track_instances.query_pos = self.query_embed.weight
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((num_queries, ), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((num_queries, 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((num_queries, self.num_classes), dtype=torch.float, device=device)
        track_instances.max_scores = torch.zeros((num_queries, ), dtype=torch.float, device=device)
        if self.with_vector:
            track_instances.pred_vectors = torch.zeros((len(track_instances), self.n_keep), dtype=torch.float, device=device)
        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_vector is None:
            return [{'pred_logits': a, 'pred_boxes': b, }
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_vectors': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, is_repeat=False):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)

        ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[5]}
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_vector if self.with_vector else None)

        with torch.no_grad():
            track_scores = outputs_class[-1, 0, :].sigmoid().max(dim=-1).values

        track_instances.scores = track_scores
        track_instances.pred_logits = outputs_class[-1, 0]
        track_instances.pred_boxes = outputs_coord[-1, 0]
        track_instances.output_embedding = hs[-1, 0]
        if self.with_vector:
                track_instances.pred_vectors = outputs_vector[-1, 0]
        if self.training:
            # the track id will be assigned by the mather.
            out['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(out)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances, is_repeat=is_repeat)
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        out_track_instances = self.track_embed(tmp)
        out['track_instances'] = out_track_instances
        return out

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, is_repeat=False):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks()
        res = self._forward_single_image(img,
                                         track_instances=track_instances, is_repeat=is_repeat)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

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
        if self.training:
            self.criterion.initialize(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }

        if self.with_vector:
            outputs.update({'pred_vectors': []})

        track_instances = self._generate_empty_tracks()
        for frame in frames:
            if not isinstance(frame, NestedTensor):
                frame = nested_tensor_from_tensor_list([frame])
            frame_res = self._forward_single_image(frame, track_instances)
            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

            if self.with_vector:
                outputs['pred_vectors'].append(frame_res['pred_vectors'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'bdd100k_mot': 8
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

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ["masks"]
    criterion = FrameMatcherV2(num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, random_drop=args.random_drop,
                                            with_vector=args.with_vector, 
                                            vector_loss_coef=args.vector_loss_coef,
                                            vector_loss_norm=args.vector_loss_norm)
    criterion.to(device)
    postprocessors = {}
    model = E2EMotDETRV3(
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
