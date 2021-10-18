

PRETRAIN=/data/dongbin/thirdparty/weights/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=exps/r50.bdd100k_mot.early_stop
python3.6 eval.py \
    --meta_arch motr \
    --dataset_file bdd100k_mot \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 17 \
    --save_period 1 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 4 \
    --sampler_steps 12 \
    --sampler_lengths 2 3  \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_embedding_layer 'AttentionMergerV4' \
    --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/bdd100k.train \
    --data_txt_path_val ./datasets/data_path/bdd100k.val \
    --mot_path /data/Dataset/bdd100k/bdd100k \
    --resume ${EXP_DIR}/checkpoint.pth \
    --img_path /data/Dataset/bdd100k/bdd100k/images/track/val

# PRETRAIN=/data/dongbin/thirdparty/weights/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
# EXP_DIR=exps/r50.bdd100k_mot.20e
# python3.6 eval.py \
#     --meta_arch motr \
#     --dataset_file bdd100k_mot \
#     --epoch 20 \
#     --with_box_refine \
#     --lr_drop 16 \
#     --save_period 2 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${PRETRAIN} \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 4 \
#     --sampler_steps 6 12 \
#     --sampler_lengths 2 3 4 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --track_embedding_layer 'AttentionMergerV4' \
#     --extra_track_attn \
#     --data_txt_path_train ./datasets/data_path/bdd100k.train \
#     --data_txt_path_val ./datasets/data_path/bdd100k.val \
#     --mot_path /data/Dataset/bdd100k/bdd100k \
#     --resume ${EXP_DIR}/checkpoint.pth \
#     --img_path /data/Dataset/bdd100k/bdd100k/images/track/val

# PRETRAIN=/data/dongbin/thirdparty/weights/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
# EXP_DIR=exps/r50.bdd100k_mot.filter
# python3.6 eval.py \
#     --meta_arch motr \
#     --dataset_file bdd100k_mot \
#     --epoch 50 \
#     --with_box_refine \
#     --lr_drop 40 \
#     --save_period 5 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${PRETRAIN} \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 4 \
#     --sampler_steps 12 22 36 \
#     --sampler_lengths 2 3 4 5 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --track_embedding_layer 'AttentionMergerV4' \
#     --extra_track_attn \
#     --filter_ignore \
#     --data_txt_path_train ./datasets/data_path/filter.bdd100k.train \
#     --data_txt_path_val ./datasets/data_path/filter.bdd100k.val \
#     --mot_path /data/Dataset/bdd100k/bdd100k \
#     --resume ${EXP_DIR}/checkpoint.pth \
#     --img_path /data/Dataset/bdd100k/bdd100k/images/track/val

# PRETRAIN=/data/dongbin/thirdparty/weights/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
# EXP_DIR=exps/r50.bdd100k_mot
# python3.6 eval.py \
#     --meta_arch motr \
#     --dataset_file bdd100k_mot \
#     --epoch 50 \
#     --with_box_refine \
#     --lr_drop 40 \
#     --save_period 5 \
#     --lr 2e-4 \
#     --lr_backbone 2e-5 \
#     --pretrained ${PRETRAIN} \
#     --output_dir ${EXP_DIR} \
#     --batch_size 1 \
#     --sample_mode 'random_interval' \
#     --sample_interval 4 \
#     --sampler_steps 12 22 36 \
#     --sampler_lengths 2 3 4 5 \
#     --update_query_pos \
#     --merger_dropout 0 \
#     --dropout 0 \
#     --random_drop 0.1 \
#     --fp_ratio 0.3 \
#     --track_embedding_layer 'AttentionMergerV4' \
#     --extra_track_attn \
#     --data_txt_path_train ./datasets/data_path/bdd100k.train \
#     --data_txt_path_val ./datasets/data_path/bdd100k.val \
#     --mot_path /data/Dataset/bdd100k/bdd100k \
#     --resume ${EXP_DIR}/checkpoint.pth \
#     --img_path /data/Dataset/bdd100k/bdd100k/images/track/val
