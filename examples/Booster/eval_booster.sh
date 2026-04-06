set -x -e

export DATASET_PATH=examples/Booster/Wave
export MODEL_PATH=/root/sde_ws/src/gr00t_inference/models/Wave/exp1/checkpoint-20000
export OUTPUT_PATH=/root/sde_ws/src/gr00t_inference/plots/Wave/exp1

export CUDA_VISIBLE_DEVICES=1

python3 gr00t/eval/open_loop_eval.py \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path $MODEL_PATH \
    --traj-ids 0 \
    --action-horizon 16 \
    --denoising-steps 4 \
    --steps 1000 \
    --modality-keys neck left_arm right_arm waist left_leg right_leg \
    --save-plot-path $OUTPUT_PATH
