#!/bin/bash
# Shared environment helpers for SLURM job scripts.

common_set_distributed_env() {
    local world_size="$1"
    local master_port="$2"
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT="$master_port"
    export WORLD_SIZE="$world_size"
}

common_set_omp_env() {
    export OMP_NUM_THREADS=NUM_THREADS
}

common_set_nccl_env() {
    export NCCL_IB_DISABLE=IB_DISABLE_FLAG
}

common_set_nccl_network_env() {
    export NCCL_DEBUG=LOG_LEVEL
    export NCCL_SOCKET_IFNAME=NETWORK_INTERFACE
    export GLOO_SOCKET_IFNAME=NETWORK_INTERFACE
}

common_set_hf_minimal_env() {
    export HF_HOME=PATH_TO_HF_HOME
}

common_set_hf_cache_env() {
    common_set_hf_minimal_env
    export TRANSFORMERS_CACHE=$HF_HOME
    export HF_DATASETS_CACHE=$HF_HOME
    export DIFFUSERS_CACHE=$HF_HOME
}

common_set_hf_token_env() {
    export HF_TOKEN=HF_TOKEN_VALUE
}

common_set_py_runtime_env() {
    export PYTHONUNBUFFERED=PYTHONUNBUFFERED_VALUE
}

common_set_hf_offline_env() {
    export HF_HUB_DISABLE_PROGRESS_BARS=HF_HUB_DISABLE_PROGRESS_BARS_VALUE
    export HF_HUB_OFFLINE=HF_HUB_OFFLINE_VALUE
    export TRANSFORMERS_OFFLINE=TRANSFORMERS_OFFLINE_VALUE
}

common_set_cuda_alloc_env() {
    export PYTORCH_CUDA_ALLOC_CONF=PYTORCH_CUDA_ALLOC_CONF_VALUE
}
