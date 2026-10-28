#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --job-name=vton_tryon_single
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=PATH_TO_LOG_DIR/%A_vton_tryon_single.log
#SBATCH --error=PATH_TO_LOG_DIR/%A_vton_tryon_single.err
#SBATCH --mem=400GB

# ── Environment ──────────────────────────────────────────────
export HF_HOME=PATH_TO_HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME
# DO NOT COMMIT THE TOKEN
export HF_TOKEN=HF_TOKEN_VALUE
export PYTHONUNBUFFERED=PYTHONUNBUFFERED_VALUE
export HF_HUB_DISABLE_PROGRESS_BARS=HF_HUB_DISABLE_PROGRESS_BARS_VALUE
export HF_HUB_OFFLINE=HF_HUB_OFFLINE_VALUE
export TRANSFORMERS_OFFLINE=TRANSFORMERS_OFFLINE_VALUE
export PYTORCH_CUDA_ALLOC_CONF=PYTORCH_CUDA_ALLOC_CONF_VALUE

# ── Distributed setup (needed for GPU isolation) ─────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER
export WORLD_SIZE=$((SLURM_NNODES * 4))

# Exit immediately if any command fails
set -e

# --- PATH CONFIGURATION ---
WORKING_DIR=PATH_TO_WORKDIR
cd ${WORKING_DIR}

mkdir -p PATH_TO_LOG_DIR

# --- CONFIGURATION (Sourced from vton/config.py) ---
export MODEL_PATH_FLUX=$(python3 -c "from vton.config import MODEL_PATH_FLUX; print(MODEL_PATH_FLUX)")
export DATASET_V3_DIR=PATH_TO_DATASET_V3

# --- CONTAINER CONFIGURATION ---
CONTAINER_ARG="--environment=qwen3vl-fa2"

JOB_START=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Nodes      : $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "MASTER     : $MASTER_ADDR:$MASTER_PORT"
echo "WORLD_SIZE : $WORLD_SIZE"
echo "START TIME : $JOB_START"
echo "============================================================"
echo "SOURCE     : ${DATASET_V3_DIR}/dataset_ultimate  (initial_person_image + cloth_image)"
echo "DEST       : ${DATASET_V3_DIR}/dataset_ultimate/{diff}/{gender}/tryon_image"
echo "============================================================"

# --- TEST MODE ---
TEST_LIMIT_ARG=""
if [ "$1" == "test" ]; then
    TEST_LIMIT_ARG="--category_limit 16"
    echo "RUNNING IN TEST MODE (LIMIT 16 per category)"
fi

# ==============================================================================
# TRY-ON GENERATION, SINGLE PASS (Flux.2-klein-9B)
# Input  : dataset_ultimate/{diff}/{gender}/initial_person_image/IMAGE_FILE.png
#          dataset_ultimate/{diff}/{gender}/cloth_image/IMAGE_FILE.png
# Output : dataset_ultimate/{diff}/{gender}/tryon_image/IMAGE_FILE.png
# Images fed to Flux in order: [person_image, cloth_image]
# Prompt : improved_prompt from an _incorrect.json if present, else DRESS_INFO / default prompt
# For the full generate -> evaluate -> rename loop, use job_tryon_loop.sh.
# ==============================================================================
echo "Starting Try-On Generation (single pass)..."
echo "Running Try-On Generator (Flux.2)..."
srun -u ${CONTAINER_ARG} bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID

    python vton/tryon/core/edit_generator.py \\
        --root_dir ${DATASET_V3_DIR} \\
        --dataset_v3_dir ${DATASET_V3_DIR} \\
        --model_path ${MODEL_PATH_FLUX} \\
        --batch_size 1 \\
        --num_workers 16 \\
        ${TEST_LIMIT_ARG}
"

JOB_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "START TIME : $JOB_START"
echo "END TIME   : $JOB_END"
echo "TRY-ON GENERATION COMPLETE."
echo "============================================================"
