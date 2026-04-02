set -x -e

export DATASET_PATH=examples/Booster/wave
export OUTPUT_DIR=/root/sde_ws/src/gr00t_inference/models/wave
export EXPERIMENT_NAME=exp3
export WANDB_ENTITY="groot-sde"

export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=1

# generate dataset statistics
# python3 \
#   examples/Booster/Wave/stats.py \
#   --dataset_path $DATASET_PATH \
#   --embodiment_tag NEW_EMBODIMENT

# torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
# CUDA_VISIBLE_DEVICES=2 python3 \
# --modality_config_path examples/Booster/booster_config.py \
python3 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path $DATASET_PATH  \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps 1000 \
    --save_total_limit 10 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --episode_sampling_rate 1.0 \
    --num_shards_per_epoch 20 \
    --shard_size 512 \
    --entity_name $WANDB_ENTITY \
    --experiment_name $EXPERIMENT_NAME \
    --global_batch_size 128 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4
