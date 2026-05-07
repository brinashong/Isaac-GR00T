set -x -e

export DATASET_PATH=examples/Booster/meet-and-greet
export MODEL_PATH=/mnt/ssd-server/eai_dataset/groot_models/meet-and-greet-n1d7/exp56-meet-and-greet-relative-action-vanilla-28Apr/checkpoint-40000
export OUTPUT_PATH=/mnt/ssd-server/eai_dataset/groot_results/meet-and-greet-n1d7/exp56

export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=1

python3 \
    gr00t/eval/open_loop_eval.py \
    --dataset_path $DATASET_PATH  \
    --embodiment_tag NEW_EMBODIMENT \
    --model-path $MODEL_PATH \
    --traj-ids 0 \
    --action-horizon 32 \
    --denoising-steps 4 \
    --steps 1000 \
    --modality-keys neck left_arm right_arm waist left_leg right_leg \
    --save-plot-path $OUTPUT_PATH
