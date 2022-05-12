# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

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

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader

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


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
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
    return img


'''
deep sort 中的画图方法，在原图上进行作画
'''
def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
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


class MOTR(object):
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

    def update(self, dt_instances: Instances):
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
            bbox = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
            label = dt_instances.labels[i]
            if label == 0:
                # get a positive track.
                if idx in self.inactive_trackers:
                    # set state of track active.
                    self.active_trackers[idx] = self.inactive_trackers.pop(idx)
                if idx not in self.active_trackers:
                    # create a new track.
                    self.active_trackers[idx] = Track(idx)
                self.active_trackers[idx].update(bbox)
            elif label == 1:
                # get an occluded track.
                if idx in self.active_trackers:
                    # set state of track inactive.
                    self.inactive_trackers[idx] = self.active_trackers.pop(idx)
                if idx not in self.inactive_trackers:
                    # It's strange to obtain a new occluded track.
                    # TODO: think more rational disposal.
                    self.inactive_trackers[idx] = Track(idx)
                self.inactive_trackers[idx].miss_one_frame()
                if self.inactive_trackers[idx].miss > 10:
                    self._remove_track(idx)

        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label == 0:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))


def load_label(label_path: str, img_size: tuple) -> dict:
    labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
    h, w = img_size
    # Normalized cewh to pixel xyxy format
    labels = labels0.copy()
    labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
    labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
    labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
    labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
    targets = {'boxes': [], 'labels': [], 'area': []}
    num_boxes = len(labels)

    visited_ids = set()
    for label in labels[:num_boxes]:
        obj_id = label[1]
        if obj_id in visited_ids:
            continue
        visited_ids.add(obj_id)
        targets['boxes'].append(label[2:6].tolist())
        targets['area'].append(label[4] * label[5])
        targets['labels'].append(0)
    targets['boxes'] = np.asarray(targets['boxes'])
    targets['area'] = np.asarray(targets['area'])
    targets['labels'] = np.asarray(targets['labels'])
    return targets


def filter_pub_det(res_file, pub_det_file, filter_iou=False):
    frame_boxes = {}
    with open(pub_det_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            elements = line.strip().split(',')
            frame_id = int(elements[0])
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if frame_id not in frame_boxes:
                frame_boxes[frame_id] = []
            frame_boxes[frame_id].append([x1, y1, x2, y2])

    for frame, boxes in frame_boxes.items():
        frame_boxes[frame] = np.array(boxes)

    ids = {}
    num_filter_box = 0
    with open(res_file, 'r') as f:
        lines = f.readlines()
    with open(res_file, 'w') as f:
        for line in lines:
            if len(line) == 0:
                continue
            elements = line.strip().split(',')
            frame_id, obj_id = elements[:2]
            frame_id = int(frame_id)
            obj_id = int(obj_id)
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if obj_id not in ids:
                # track initialization.
                if frame_id not in frame_boxes:
                    num_filter_box += 1
                    print("filter init box {} {}".format(frame_id, obj_id))
                    continue
                pub_dt_boxes = frame_boxes[frame_id]
                dt_box = np.array([[x1, y1, x2, y2]])
                if filter_iou:
                    max_iou = bbox_iou(dt_box, pub_dt_boxes).max()
                    if max_iou < 0.5:
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                else:
                    pub_dt_centers = (pub_dt_boxes[:, :2] + pub_dt_boxes[:, 2:4]) * 0.5
                    x_inside = (dt_box[0, 0] <= pub_dt_centers[:, 0]) & (dt_box[0, 2] >= pub_dt_centers[:, 0])
                    y_inside = (dt_box[0, 1] <= pub_dt_centers[:, 1]) & (dt_box[0, 3] >= pub_dt_centers[:, 1])
                    center_inside:np.ndarray = x_inside & y_inside
                    if not center_inside.any():
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                print("save init track {} {}".format(frame_id, obj_id))
                ids[obj_id] = True
            f.write(line)

    print("totally {} boxes are filtered.".format(num_filter_box))

class ListImgDataset(Dataset):
    def __init__(self, img_list) -> None:
        super().__init__()
        self.img_list = img_list

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        label_path = f_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        targets = load_label(label_path, cur_img.shape[:2]) if os.path.exists(label_path) else None
        return cur_img, targets

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

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, targets = self.load_img_from_file(self.img_list[index])
        return self.init_img(img)


class Detector(object):
    def __init__(self, args, model=None, seq_num=2):

        self.args = args
        self.detr = model

        self.seq_num = seq_num
        img_list = os.listdir(os.path.join(self.args.mot_path, 'MOT17/images/test', self.seq_num, 'img1'))
        img_list = [os.path.join(self.args.mot_path, 'MOT17/images/test', self.seq_num, 'img1', _) for _ in img_list if
                    ('jpg' in _) or ('png' in _)]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)
        self.tr_tracker = MOTR()

        self.save_path = os.path.join(self.args.output_dir, 'results/{}'.format(seq_num))
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_path, f'{self.seq_num}.txt')):
            os.remove(os.path.join(self.predict_path, f'{self.seq_num}.txt'))

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

    @staticmethod
    def write_results(txt_path, frame_id, bbox_xyxy, identities):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        with open(txt_path, 'a') as f:
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
                f.write(line)

    def eval_seq(self):
        data_root = os.path.join(self.args.mot_path, 'MOT15/images/train')
        result_filename = os.path.join(self.predict_path, 'gt.txt')
        evaluator = Evaluator(data_root, self.seq_num)
        accs = evaluator.eval_file(result_filename)
        return accs

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes)
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        cv2.imwrite(img_path, img_show)

    def detect(self, prob_threshold=0.7, area_threshold=100):
        last_dt_embedding = None
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = None
        loader = DataLoader(ListImgDataset(self.img_list), 1, num_workers=2)
        for i, (cur_img, ori_img) in enumerate(tqdm(loader)):
            cur_img, ori_img = cur_img[0], ori_img[0]

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_image(cur_img.cuda().float(), (seq_h, seq_w), track_instances)
            track_instances = res['track_instances']

            all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
            dt_instances = track_instances.to(torch.device('cpu'))

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            num_occlusion = (dt_instances.labels == 1).sum()
            dt_instances.scores[dt_instances.labels == 1] *= -1
            total_dts += len(dt_instances)
            total_occlusion_dts += num_occlusion

            tracker_outputs = self.tr_tracker.update(dt_instances)

            self.write_results(txt_path=os.path.join(self.predict_path, f'{self.seq_num}.txt'),
                               frame_id=(i + 1),
                               bbox_xyxy=tracker_outputs[:, :4],
                               identities=tracker_outputs[:, 5])
        # filter_pub_det(os.path.join(self.predict_path, f'{self.seq_num}.txt'),
        #                 f'/data/Dataset/mot/MOT17/images/test/{self.seq_num}/det/det.txt')
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))


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

    # '''for MOT17 submit''' 
    sub_dir = 'MOT17/images/test'
    seq_nums = ['MOT17-01-SDP',
                'MOT17-03-SDP',
                'MOT17-06-SDP',
                'MOT17-07-SDP',
                'MOT17-08-SDP',
                'MOT17-12-SDP',
                'MOT17-14-SDP']

    for seq_num in seq_nums:
        det = Detector(args, model=detr, seq_num=seq_num)
        det.detect()

    """copy reuslts for same sequences"""
    repeated_seq_nums = ['MOT17-01-DPM',
                        'MOT17-03-DPM',
                        'MOT17-06-DPM',
                        'MOT17-07-DPM',
                        'MOT17-08-DPM',
                        'MOT17-12-DPM',
                        'MOT17-14-DPM',
                        'MOT17-01-FRCNN',
                        'MOT17-03-FRCNN',
                        'MOT17-06-FRCNN',
                        'MOT17-07-FRCNN',
                        'MOT17-08-FRCNN',
                        'MOT17-12-FRCNN',
                        'MOT17-14-FRCNN']
    
    print('copy reuslts for same sequences: ')
    predict_path = os.path.join(args.output_dir, args.exp_name)
    for repeated_seq_nums_i in repeated_seq_nums:
        u, v = repeated_seq_nums_i.split('-')[:-1]
        shutil.copyfile(os.path.join(predict_path, '{}-{}-SDP.txt'.format(u,v)),os.path.join(predict_path,f'{repeated_seq_nums_i}.txt'))

    sub_dir = 'MOT17/images/train'
    seq_nums = os.listdir('/data/Dataset/mot/MOT17/images/train')
    accs = []
    seqs = []
    for seq_num in seq_nums:
        shutil.copyfile(os.path.join(args.mot_path, sub_dir, f'{seq_num}/gt/gt.txt'),os.path.join(predict_path,f'{seq_num}.txt'))
