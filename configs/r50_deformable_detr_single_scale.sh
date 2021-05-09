#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
