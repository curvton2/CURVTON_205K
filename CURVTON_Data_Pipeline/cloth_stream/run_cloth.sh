#!/bin/bash
#SBATCH --account=SLURM_ACCOUNT
#SBATCH --partition=normal
#SBATCH --job-name=generate_cloth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=../../logs/gen_cloth_%j.out
#SBATCH --error=../../logs/gen_cloth_%j.err



# ── Distributed env vars ──────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER
export WORLD_SIZE=NUM_PROCESSES
export OMP_NUM_THREADS=NUM_THREADS

# Cluster inter-node networking (safeguard, though 1 node doesn't strictly need IB)
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
echo "Running on node: $(hostname)"

# ── One SRUN pointing to the worker script ────────────────────────────────────
# Note: Ensure you are using your container environment, assuming "qwen3vl-fa2" as in reference
srun --label --kill-on-bad-exit=1 \
    --environment=qwen3vl-fa2 \
    bash PATH_TO_WORKER_SCRIPT

echo "Job finished: $(date)"
