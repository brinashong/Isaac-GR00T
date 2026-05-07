set -x -e

export DATASET_PATH=examples/Booster/wave # meet-and-greet
export OUTPUT_DIR=/mnt/ssd-server/eai_dataset/groot_models/wave-n1d7
export EXPERIMENT_NAME=exp56-wave-relative-action-vanilla-7May
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
    --max_steps 40000 \
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

    # --use_prev_action_conditioning
