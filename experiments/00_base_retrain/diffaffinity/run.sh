#!/bin/bash
export PTXAS_DIR="/data/yuwl/paper1/envs/env-diffaffinity/lib/python3.9/site-packages/triton/backends/nvidia/bin"
export CUDNN8_PATH="/data/yuwl/paper1/envs/env-diffaffinity/cudnn8_for_jax"
export CUDA_PATH="/data/yuwl/paper1/envs/env-diffaffinity/lib/python3.9/site-packages/nvidia"
export PATH="$PTXAS_DIR:$PATH"
export LD_LIBRARY_PATH="$CUDNN8_PATH:$CUDA_PATH/cuda_runtime/lib:$CUDA_PATH/cublas/lib:$CUDA_PATH/cufft/lib:$CUDA_PATH/cusolver/lib:$CUDA_PATH/cusparse/lib:$CUDA_PATH/nvjitlink/lib"
export CUDA_VISIBLE_DEVICES=${1:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05
export XLA_FLAGS="--xla_gpu_autotune_level=0"

cd /data/yuwl/paper1/CIR-DDG/baselines/DiffAffinity
exec /data/yuwl/paper1/envs/env-diffaffinity/bin/python -u /data/yuwl/paper1/experiments_v3/00_base_retrain/diffaffinity/train.py
