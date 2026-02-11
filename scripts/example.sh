#!/bin/bash
export TORCH_CUDA_ARCH_LIST="8.9"

conda activate flymyai-lora
source .venv/bin/activate

# --num_processes=1 表示使用 1 个 GPU、--gpu_ids=0 表示使用第 0 个 GPU
accelerate launch --num_processes=1 --gpu_ids=0 train_4090.py --config ./train_configs/train_lora_4090.yaml

# --multi_gpu 多 GPU 训练、num_processes 表示使用 x 个 GPU
accelerate launch --multi_gpu --num_processes=4 train_4090.py --config ./train_configs/train_lora_4090.yaml
accelerate launch --multi_gpu --num_processes=2 train_4090.py --config ./train_configs/train_lora_4090.yaml
accelerate launch --multi_gpu --num_processes=1 train_4090.py --config ./train_configs/train_lora_4090.yaml

# 终端展示 GPU 、CPU、MEM 利用率使用
uvx nvitop -m

uv run modelscope download --model Qwen/Qwen-Image-2512 --local_dir ./models/Qwen-Image-2512
