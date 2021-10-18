# -*- coding: UTF-8 -*-
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List
from util.evaluation import Evaluator
import motmetrics as mm
import shutil
import json
import pycocotools.mask as mask_util
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom

np.random.seed(2020)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, mask=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # print("c1c2 = {} {}".format(c1, c2))
    if mask is not None:
        v = Visualizer(img, scale=1)
        vis_mask = v.draw_binary_mask(mask[0].cpu().numpy(), color="blue")
        img = vis_mask.get_image()
    return img


def draw_bboxes(ori_img, bbox, identities=None, mask=None, offset=(0, 0), cvt_color=False):
    # if cvt_color:
    #     ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        if mask is not None and mask.shape[0] > 0:
            m = mask[i]
        else:
            m = None
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
            label = int(box[5])
        else:
            score = None
            label = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label_str = '{:d}@{:d}'.format(id, label)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label_str, score=score, mask=m)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class TRTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.active_trackers = {}
        self.inactive_trackers = {}
        self.disappeared_tracks = []

    def _remove_track(self, slot_id):
        self.inactive_trackers.pop(slot_id)
        self.disappeared_tracks.append(slot_id)

    def clear_disappeared_track(self):
        self.disappeared_tracks = []

    def update(self, dt_instances: Instances, target_size=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        dt_idxes = set(dt_instances.obj_idxes.tolist())
        track_idxes = set(self.active_trackers.keys()).union(set(self.inactive_trackers.keys()))
        matched_idxes = dt_idxes.intersection(track_idxes)

        unmatched_tracker = track_idxes - matched_idxes
        for track_id in unmatched_tracker:
            # miss in this frame, move to inactive_trackers.
            if track_id in self.active_trackers:
                self.inactive_trackers[track_id] = self.active_trackers.pop(track_id)
            self.inactive_trackers[track_id].miss_one_frame()
            if self.inactive_trackers[track_id].miss > 10:
                self._remove_track(track_id)

        for i in range(len(dt_instances)):
            idx = dt_instances.obj_idxes[i]
            bbox = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i + 1]], axis=-1)
            label = dt_instances.labels[i]
            if label < 8:
                # get a positive track.
                if idx in self.inactive_trackers:
                    # set state of track active.
                    self.active_trackers[idx] = self.inactive_trackers.pop(idx)
                if idx not in self.active_trackers:
                    # create a new track.
                    self.active_trackers[idx] = Track(idx)
                self.active_trackers[idx].update(bbox)

        ret = []
        if dt_instances.has('masks'):
            mask = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label < 8:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate(
                    [dt_instances.boxes[i], dt_instances.scores[i:i + 1], dt_instances.labels[i:i + 1]], axis=-1)
                ret.append(
                    np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                if dt_instances.has('masks'):
                    mask.append(dt_instances.masks[i])

        img_h, img_w = target_size
        if len(ret) > 0:
            if dt_instances.has('masks'):
                return np.concatenate(ret), np.concatenate(mask)
            return np.concatenate(ret)
        if dt_instances.has('masks'):
            return np.empty((0, 7)), np.empty((0, 1, img_h, img_w))
        return np.empty((0, 7))


class Detector(object):
    def __init__(self, args, model=None, seq_num=2):

        self.args = args
        self.detr = model

        self.seq_num = seq_num
        file_names = os.listdir(os.path.join(args.img_path, seq_num))
        file_names.sort()
        self.img_list = [os.path.join(args.img_path, seq_num, item) for item in file_names]
        self.img_len = len(self.img_list)
        self.tr_tracker = TRTR()

        self.img_height = 800
        self.img_width = 1333

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.save_path = os.path.join(self.args.output_dir, 'results/{}'.format(seq_num))
        self.save_json_path = os.path.join(self.args.output_dir, 'data')
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_json_path, exist_ok=True)

        self.cls2label = {
            1: "pedestrian",
            2: "rider",
            3: "car",
            4: "truck",
            5: "bus",
            6: "train",
            7: "motorcycle",
            8: "bicycle"
        }

        self.outputs = [self.gen_empty_dict(fname, seq_num, fid) for fid,fname in enumerate(file_names)]

    def gen_empty_dict(self, fname, vid, fid):
        sample = {
                    "name": fname,
                    "labels": [],
                    "videoName": vid,
                    "frameIndex": fid
                }
        return sample

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def update_results(self, frame_id, bbox_xyxy, identities, labels, scores=None, masks=None):
        for i, (xyxy, track_id, label) in enumerate(zip(bbox_xyxy, identities, labels)):
            if track_id < 0 or track_id is None:
                continue
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            line = {'frame': int(frame_id), 'id': int(track_id), 'x1': x1, 'y1': y1, 'w': w, 'h': h, 'label': label}
            if masks is not None and masks.shape[0] > 0:
                line.update({'mask': masks[i], 'score': scores[i]})
            ret = {
                "id": str(int(track_id)),
                "category": self.cls2label[label+1],
                "box2d": {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
            }
            self.outputs[int(frame_id)-1]["labels"].append(ret)

    def vector2mask(self, dt_instances: Instances) -> Instances:
        vectors = dt_instances.vectors
        boxes = dt_instances.boxes

        if self.detr.processor_dct is not None:
            n_keep = self.detr.processor_dct.n_keep
            masks = []
            n_keep, gt_mask_len = self.detr.processor_dct.n_keep, self.detr.processor_dct.gt_mask_len
            r, c = vectors.shape
            outputs_masks_per_image = []
            for ri in range(r):
                # this a hack
                idct = np.ones((gt_mask_len ** 2)) * 0.001
                idct[:n_keep] = vectors[ri].cpu().numpy()
                idct = self.detr.processor_dct.inverse_zigzag(idct, gt_mask_len, gt_mask_len)
                re_mask = cv2.idct(idct)
                max_v = np.max(re_mask)
                min_v = np.min(re_mask)
                re_mask = np.where(re_mask > (max_v + min_v) / 2., 1, 0)
                re_mask = torch.from_numpy(re_mask)[None].float()
                outputs_masks_per_image.append(re_mask)
            if len(outputs_masks_per_image) == 0:
                dt_instances.masks = np.zeros((0, 1, self.seq_h, self.seq_w))
                return dt_instances
            outputs_masks_per_image = torch.cat(outputs_masks_per_image, dim=0).to(vectors.device)
            # here padding local mask to global mask
            outputs_masks_per_image = retry_if_cuda_oom(paste_masks_in_image)(
                outputs_masks_per_image,  # N, 1, M, M
                boxes,
                (self.seq_h, self.seq_w),
                threshold=0.5,
            )

            outputs_masks_per_image = outputs_masks_per_image.unsqueeze(1).cpu()
        dt_instances.masks = outputs_masks_per_image
        return dt_instances

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if dt_instances.has('scores'):
            if dt_instances.has('masks'):
                img_show = draw_bboxes(img, np.concatenate(
                    [dt_instances.boxes, dt_instances.scores.reshape(-1, 1), dt_instances.labels.reshape(-1, 1)],
                    axis=-1), dt_instances.obj_idxes, dt_instances.masks)
            else:
                img_show = draw_bboxes(img, np.concatenate(
                    [dt_instances.boxes, dt_instances.scores.reshape(-1, 1), dt_instances.labels.reshape(-1, 1)],
                    axis=-1), dt_instances.obj_idxes)
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        cv2.imwrite(img_path, img_show)

    def detect(self, prob_threshold=0.6, area_threshold=100, dump=False, vis=False):
        last_dt_embedding = None

        track_instances = None
        for i in range(0, self.img_len):
            img = cv2.imread(self.img_list[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cur_img, ori_img = self.init_img(img)

            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')

            res = self.detr.inference_single_image(cur_img.cuda().float(), (self.seq_h, self.seq_w), track_instances)
            track_instances = res['track_instances']

            all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
            dt_instances = track_instances.to(torch.device('cpu'))

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            # here perform dct and convert local mask to global mask
            if dt_instances.has('vectors'):
                dt_instances = self.vector2mask(dt_instances)

            cur_vis_img_path = os.path.join(self.save_path, 'frame_{}.jpg'.format(i))
            if vis:
                self.visualize_img_with_bbox(cur_vis_img_path, ori_img, dt_instances, ref_pts=all_ref_pts)
            if dt_instances.has('masks'):
                tracker_outputs, seg_outputs = self.tr_tracker.update(dt_instances,
                                                                      target_size=(self.seq_h, self.seq_w))
            else:
                tracker_outputs = self.tr_tracker.update(dt_instances, target_size=(self.seq_h, self.seq_w))
                seg_outputs = None

            self.update_results(frame_id=(i + 1),
                                bbox_xyxy=tracker_outputs[:, :4],
                                identities=tracker_outputs[:, 6],
                                labels=tracker_outputs[:, 5],
                                scores=tracker_outputs[:, 4],
                                masks=seg_outputs)

        # dump results
        if dump:
            with open(os.path.join(self.save_json_path, '{}.json'.format(self.seq_num)), 'w', encoding='utf-8') as f:
                json.dump(self.outputs, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    videos = os.listdir(args.img_path)
    for seq_num in tqdm(videos):
        det = Detector(args, model=detr, seq_num=seq_num)
        det.detect(dump=True, vis=False)