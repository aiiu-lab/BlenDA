CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ../../tools/run_dist_launch.sh 1 --master_port 29500 \
python ../../main.py \
--config_file config.yaml \
--opts \
    DATASET.DA_MODE blenda \
    BLENDA.MIXUP_SRC_INT_IMGS True \
    BLENDA.MIXUP_SRC_INT_DOMAIN_LABELS True \
    BLENDA.MIXUP_SRC_TGT_IMGS True \
    BLENDA.MIXUP_SRC_TGT_DOMAIN_LABELS True \
    BLENDA.INT_DOMAIN_LABEL 1 \
    BLENDA.ALPHA 20.0 \
    OUTPUT_DIR exps/c2fc \
    RESUME ../../released_weights/cityscapes_to_foggy_cityscapes.pth \
    TRAIN.EPOCHS 150 \
    TRAIN.BATCH_SIZE 3 \
    TRAIN.LR 2e-05 \
    TRAIN.LR_BACKBONE 2e-06 \
    TRAIN.LR_DROP 500 \
    TRAIN.FINETUNE True