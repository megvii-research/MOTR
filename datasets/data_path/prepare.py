import os
from functools import partial
from typing import List


def solve_MOT_train(root, year):
    assert year in [15, 16, 17]
    dataset_path = 'MOT{}/images/train'.format(year)
    data_root = os.path.join(root, dataset_path)
    if year == 17:
        video_paths = []
        for video_name in os.listdir(data_root):
            if 'SDP' in video_name:
                video_paths.append(video_name)
    else:
        video_paths = os.listdir(data_root)

    frames = []
    for video_name in video_paths:
        files = os.listdir(os.path.join(data_root, video_name, 'img1'))
        files.sort()
        for i in range(1, len(files) + 1):
            frames.append(os.path.join(dataset_path, video_name, 'img1', '%06d.jpg' % i))
    return frames


def solve_CUHK(root):
    dataset_path = 'ethz/CUHK-SYSU'
    data_root = os.path.join(root, dataset_path)
    file_names = os.listdir(os.path.join(data_root, 'images'))
    file_names.sort()

    frames = []
    for i in range(len(file_names)):
        if os.path.exists(os.path.join(root, 'ethz/CUHK-SYSU/labels_with_ids', f's{i + 1}.txt')):
            if os.path.exists(os.path.join(root, 'ethz/CUHK-SYSU/images', f's{i + 1}.jpg')):
                frames.append(os.path.join('ethz/CUHK-SYSU/images', f's{i + 1}.jpg'))
    return frames

def solve_ETHZ(root):
    dataset_path = 'ethz/ETHZ'
    data_root = os.path.join(root, dataset_path)
    video_paths = []
    for name in os.listdir(data_root):
        if name not in ['eth01', 'eth03']:
            video_paths.append(name)

    frames = []
    for video_path in video_paths:
        files = os.listdir(os.path.join(data_root, video_path, 'images'))
        files.sort()
        for img_name in files:
            if os.path.exists(os.path.join(data_root, video_path, 'labels_with_ids', img_name.replace('.png', '.txt'))):
                if os.path.exists(os.path.join(data_root, video_path, 'images', img_name)):
                    frames.append(os.path.join('ethz/ETHZ', video_path, 'images', img_name))
    return frames


def solve_PRW(root):
    dataset_path = 'ethz/PRW'
    data_root = os.path.join(root, dataset_path)
    frame_paths = os.listdir(os.path.join(data_root, 'images'))
    frame_paths.sort()
    frames = []
    for i in range(len(frame_paths)):
        if os.path.exists(os.path.join(data_root, 'labels_with_ids', frame_paths[i].split('.')[0] + '.txt')):
            if os.path.exists(os.path.join(data_root, 'images', frame_paths[i])):
                frames.append(os.path.join(dataset_path, 'images', frame_paths[i]))
    return frames


dataset_catalog = {
    'MOT15': partial(solve_MOT_train, year=15),
    'MOT16': partial(solve_MOT_train, year=16),
    'MOT17': partial(solve_MOT_train, year=17),
    'CUHK-SYSU': solve_CUHK,
    'ETHZ': solve_ETHZ,
    'PRW': solve_PRW,
}


def solve(dataset_list: List[str], root, save_path):
    all_frames = []
    for dataset_name in dataset_list:
        dataset_frames = dataset_catalog[dataset_name](root)
        print("solve {} frames from dataset:{} ".format(len(dataset_frames), dataset_name))
        all_frames.extend(dataset_frames)
    print("totally {} frames are solved.".format(len(all_frames)))
    with open(save_path, 'w') as f:
        for u in all_frames:
            line = '{}'.format(u) + '\n'
            f.writelines(line)

root = '/data/workspace/datasets/mot' 
save_path = '/data/workspace/detr-mot/datasets/data_path/mot17.train' # for fangao
dataset_list = ['MOT17', ]

solve(dataset_list, root, save_path)
