#!/bin/bash
export TORCH_CUDA_ARCH_LIST="8.9"

source .venv/bin/activate
accelerate launch --num_processes=1 --gpu_ids=0 train_4090.py --config ./train_configs/train_lora_4090.yaml
