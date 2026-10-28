#!/bin/bash
#SBATCH --account=SLURM_ACCOUNT
#SBATCH --partition=normal
#SBATCH --job-name=flux-2k-edit
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/2k_edit_%j.out
#SBATCH --error=logs/2k_edit_%j.err

mkdir -p logs

# ── Distributed env vars ──────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER
export WORLD_SIZE=NUM_PROCESSES
export OMP_NUM_THREADS=NUM_THREADS

# Cluster inter-node networking
export NCCL_DEBUG=LOG_LEVEL
export NCCL_IB_DISABLE=IB_DISABLE_FLAG
export NCCL_SOCKET_IFNAME=NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=NETWORK_INTERFACE

# ── HuggingFace cache ─────────────────────────────────────────────────────────
export HF_HOME=PATH_TO_HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME
export HF_TOKEN=HF_TOKEN_VALUE

echo "Job started: $(date)"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Running on nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

# ── ONE srun per node, container provides the full environment ────────────────
srun --label --kill-on-bad-exit=1 \
    --environment=qwen3vl-fa2 \
    bash run_2k_edits_worker.sh

echo "Job finished: $(date)"