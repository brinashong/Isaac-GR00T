set -x -e

export DATASET_PATH=examples/Booster/meet-and-greet_50Hz_4Jun # meet-and-greet
export OUTPUT_DIR=/mnt/ssd-server/eai_dataset/groot_models/wave-n1d7
export EXPERIMENT_NAME=exp16-meet-and-greet-absolute-action-vanilla-50Hz-20Jun
export WANDB_ENTITY="groot-sde"

export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

# CUDA_VISIBLE_DEVICES=2 python3 \
#--base_model_path /mnt/ssd-server/eai_dataset/groot_models/meet-and-greet/exp36-meet-and-greet/checkpoint-20000 \
#CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=$NUM_GPUS \
python3 \
    gr00t/experiment/launch_finetune.py \
    --dataset_path $DATASET_PATH  \
    --base_model_path nvidia/GR00T-N1.7-3B \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path examples/Booster/booster_config.py \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps 2000 \
    --save_total_limit 20 \
    --max_steps 50000 \
    --warmup_ratio 0.02 \
    --weight_decay 1e-4 \
    --learning_rate 5e-5 \
    --lambda_smooth 0.0 \
    --lambda_accel 0.0 \
    --lambda_continuity 0.0 \
    --use_stats_norm_scale \
    --use_wandb \
    --episode_sampling_rate 1.0 \
    --state_dropout_prob 0.2 \
    --num_shards_per_epoch 20 \
    --shard_size 512 \
    --entity_name $WANDB_ENTITY \
    --experiment_name $EXPERIMENT_NAME \
    --global_batch_size 128 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4

    # --use_prev_action_conditioning \
