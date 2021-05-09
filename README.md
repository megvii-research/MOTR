# MOTR: End-to-End Multiple-Object Tracking with TRansformer

This repository is an official implementation of the paper [MOTR: End-to-End Multiple-Object Tracking with TRansformer]().

## Introduction

**TL; DR.** MOTR is a fully end-to-end multiple-object tracking framework based on Transformer. It directly outputs the tracks within the video sequences without any association procedures.

<div style="align: center">
<img src=./figs/motr.png/>
</div>

**Abstract.** The key challenge in multiple-object tracking (MOT) task is temporal modeling of the object under track. Existing tracking-by-detection methods adopt simple heuristics, such as spatial or appearance similarity. Such methods, in spite of their commonality, are overly simple and insufficient to model complex variations, such as tracking through occlusion. Inherently, existing methods lack the ability to learn temporal variations from data. In this paper, we present MOTR, the first fully end-to-end multiple-object tracking framework. It learns to model the long-range temporal variation of the objects. It performs temporal association implicitly and avoids previous explicit heuristics. Built on Transformer and DETR, MOTR introduces the concept of “track query”. Each track query models the entire track of an object. It is transferred and updated frame-by-frame to perform object detection and tracking, in a seamless manner. Temporal aggregation network combined with multi-frame training is proposed to model the long-range temporal relation. Experimental results show that MOTR achieves state-of-the-art performance. Code will be released.


<!-- ## Citing Deformable DETR
If you find Deformable DETR useful in your research, please consider citing:
```bibtex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
``` -->

## Main Results

|  **Method**  |  **Dataset**  |  **MOTA**  |  **IDF1**  |  **IDS**  |  **URL**  |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| MOTR | MOT16 | 65.8 | 67.1 | 547 | [model](https://drive.google.com/file/d/1xqRRHvsHZ22icdWwGZbzonNa440Aqw_9/view?usp=sharing) |
| MOTR | MOT17 | 66.5 | 67.0 | 1884 | [model](https://drive.google.com/file/d/1xqRRHvsHZ22icdWwGZbzonNa440Aqw_9/view?usp=sharing) |

*Note:*

1. All models of MOTR are trained on NVIDIA Tesla V100 GPU.
2. The original implementation is based on our internal codebase. There are slight differences in the final accuracy and running time due to the plenty details in platform switch.


## Installation

The codebases are built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators

build MultiScaleDeformableAttention

```bash
cd ./models/ops
sh ./make.sh
```

## Usage

### Dataset preparation

Please download [MOT17 dataset](https://motchallenge.net/) and [CrowdHuman dataset](https://www.crowdhuman.org/) and organize them like [FairMOT](https://github.com/ifzhang/FairMOT) as following:

```
.
├── crowdhuman
│   ├── images
│   └── labels_with_ids
├── MOT15
│   ├── images
│   ├── labels_with_ids
│   ├── test
│   └── train
├── MOT17
│   ├── images
│   ├── labels_with_ids

```

### Training and Evaluation

#### Training on single node

You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). Then training MOTR on 8 GPUs as following:

```bash 
sh configs/r50_motr_train.sh

```

#### Evaluation on MOT15

You can download the pretrained model of MOTR (the link is in "Main Results" session), then run following command to evaluate it on MOT15 train dataset:

```bash 
sh configs/r50_motr_eval.sh

```

#### Evaluation on MOT17

You can download the pretrained model of MOTR (the link is in "Main Results" session), then run following command to evaluate it on MOT17 test dataset (submit to server):

```bash
sh configs/r50_motr_submit.sh

```