This instruction assumes you are using Docker container made by I2R - Dockerfile name: "Dockerfile-dev-sde-gpu_inference_humble" or image name: "u22_gpu_humble_inference_sde:latest"

1. Ensure env variables are set by checking: 
echo $FLASH_ATTN_CUDA_ARCHS
echo $MAX_JOBS  # PLEASE SET THIS CAREFULLY!
echo $FLASH_ATTN_CUDA_ARCHS

2. Run the following: 
uv pip install -e . --system

3. You are done with the setup! You may run the training and inference, etc scripts. 

Things to work on: 
1. Make ROS2-GR00T inference node
2. How to add embodiment tags for Booster T1 instead of using "NEW_EMBODIMENT"?
3. How to train one model for multiple humanoid robot embodiments using GR00T? 

