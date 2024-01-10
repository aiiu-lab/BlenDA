CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29500 \
python main.py \
--config_file exps/c2fc/config.yaml \
--opts \
    DATASET.DA_MODE blenda \
    OUTPUT_DIR exps/c2fc \
    RESUME exps/c2fc/checkpoints/checkpoint_best.pth \
    TRAIN.BATCH_SIZE 3