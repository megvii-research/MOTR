import os 
import numpy as np 
import json
import cv2
from tqdm import tqdm
from collections import defaultdict


def convert(img_dir, split, label_dir, save_label_dir, filter_crowd=False, filter_ignore=False):
    cat2id = {'train':6, 'car':3, 'bus':5, 'other person': 1, 'rider':2, 'pedestrian':1, 'other vehicle':3, 'motorcycle':7, 'bicycle':8, 'trailer':4, 'truck':4}

    coco = defaultdict(list)
    coco["categories"] = [
        {"supercategory": "human", "id": 1, "name": "pedestrian"},
        {"supercategory": "human", "id": 2, "name": "rider"},
        {"supercategory": "vehicle", "id": 3, "name": "car"},
        {"supercategory": "vehicle", "id": 4, "name": "truck"},
        {"supercategory": "vehicle", "id": 5, "name": "bus"},
        {"supercategory": "vehicle", "id": 6, "name": "train"},
        {"supercategory": "bike", "id": 7, "name": "motorcycle"},
        {"supercategory": "bike", "id": 8, "name": "bicycle"},
    ]
    attr_id_dict = {
        frame["name"]: frame["id"] for frame in coco["categories"]
    }

    all_categories = set()
    img_dir = os.path.join(img_dir, split)
    label_dir = os.path.join(label_dir, split)
    vids = os.listdir(img_dir)
    for vid in tqdm(vids):
        txt_label_dir = os.path.join(save_label_dir, split, vid)
        os.makedirs(txt_label_dir, exist_ok=True)
        annos = json.load(open(os.path.join(label_dir, vid+'.json'), 'r'))
        for anno in annos:
            name = anno['name']
            labels = anno['labels']
            videoName = anno['videoName']
            frameIndex = anno['frameIndex']
            img = cv2.imread(os.path.join(img_dir, vid, name))
            seq_height, seq_width, _ = img.shape
            if len(labels) < 1:
                continue
            # for label in labels:
            #     category = label['category']
            #     all_categories.add(category)
            with open(os.path.join(txt_label_dir, name.replace('jpg', 'txt')), 'w') as f:
                for label in labels:
                    obj_id = label['id']
                    category = label['category']
                    attributes = label['attributes']
                    is_crowd = attributes['crowd']

                    if filter_crowd and is_crowd:
                        continue
                    if filter_ignore and (category not in attr_id_dict.keys()):
                        continue

                    box2d = label['box2d']
                    x1 = box2d['x1']
                    x2 = box2d['x2']
                    y1 = box2d['y1']
                    y2 = box2d['y2']
                    w = x2-x1
                    h = y2-y1
                    cx = (x1+x2) / 2
                    cy = (y1+y2) / 2
                    label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        cat2id[category], int(obj_id), cx / seq_width, cy / seq_height, w / seq_width, h / seq_height)
                    f.write(label_str)
    # print(f'all categories are {all_categories}.')

def generate_txt(img_dir,label_dir,txt_path='bdd100k.train',split='train'):
    img_dir = os.path.join(img_dir, split)
    label_dir = os.path.join(label_dir, split)
    all_vids = os.listdir(img_dir)
    all_frames = []
    for vid in tqdm(all_vids):
        fids = os.listdir(os.path.join(img_dir, vid))
        fids.sort()
        for fid in fids:
            if os.path.exists(os.path.join(label_dir, vid, fid.replace('jpg', 'txt'))):
                all_frames.append(f'images/track/{split}/{vid}/{fid}')
    with open(txt_path, 'w') as f:
        for frame in all_frames:
            f.write(frame+'\n')

'''no filter'''
# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/labels/track'
# split = 'train'
# convert(img_dir, split, label_dir, save_label_dir)

# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/labels/track'
# split = 'val'
# convert(img_dir, split, label_dir, save_label_dir)

# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/labels/track'
# split = 'train'
# generate_txt(img_dir,save_label_dir,txt_path='bdd100k.train',split='train')

# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/labels/track'
# split = 'val'
# generate_txt(img_dir,save_label_dir,txt_path='bdd100k.val',split='val')


'''for filter'''
# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/filter_labels/track'
# split = 'train'
# convert(img_dir, split, label_dir, save_label_dir, filter_crowd=True, filter_ignore=True)

# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/filter_labels/track'
# split = 'val'
# convert(img_dir, split, label_dir, save_label_dir, filter_crowd=True, filter_ignore=True)

# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/filter_labels/track'
# split = 'train'
# generate_txt(img_dir,save_label_dir,txt_path='filter.bdd100k.train',split='train')

# img_dir = '/data/Dataset/bdd100k/bdd100k/images/track'
# label_dir = '/data/Dataset/bdd100k/bdd100k/labels/box_track_20'
# save_label_dir = '/data/Dataset/bdd100k/bdd100k/filter_labels/track'
# split = 'val'
# generate_txt(img_dir,save_label_dir,txt_path='filter.bdd100k.val',split='val')

