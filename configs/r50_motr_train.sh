# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


# for MOT17

# PRETRAIN=coco_model_final.pth
# EXP_DIR=exps/e2e_motr_r50_joint
# python3 -m torch.distributed.launch --nproc_per_node=8 \
#     --use_env main.py \
#     --meta_arch motr \
#     --use_checkpoint \
#     --dataset_file e2e_joint \
#     --epoch 200 \
#     --with_box_refine \
#     --lr_drop 100 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${PRETRAIN} \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 10 \
#     --sampler_steps 50 90 150 \
#     --sampler_lengths 2 3 4 5 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --query_interaction_layer 'QIM' \
#     --extra_track_attn \
#     --data_txt_path_train ./datasets/data_path/joint.train \
#     --data_txt_path_val ./datasets/data_path/mot17.train \