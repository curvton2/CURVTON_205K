#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --job-name=vton_tryon_eval
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=PATH_TO_LOG_DIR/%A_vton_tryon_eval.log
#SBATCH --error=PATH_TO_LOG_DIR/%A_vton_tryon_eval.err
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

# ── Distributed setup ────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER
export WORLD_SIZE=$((SLURM_NNODES * 4))

# Exit immediately if any command fails
set -e

# --- PATH CONFIGURATION ---
WORKING_DIR=PATH_TO_WORKDIR
cd ${WORKING_DIR}

mkdir -p PATH_TO_LOG_DIR

# --- CONFIGURATION ---
export MODEL_PATH_QWEN=$(python3 -c "from vton.config import MODEL_PATH_QWEN; print(MODEL_PATH_QWEN)")
export TRYON_EVAL_PROMPT_FILE=$(python3 -c "from vton.config import TRYON_EVAL_PROMPT_FILE; print(TRYON_EVAL_PROMPT_FILE)")
export DATASET_V3_DIR=PATH_TO_DATASET_V3
export DATASET_ULTIMATE_1_DIR="${DATASET_V3_DIR}/dataset_ultimate_1"

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

# --- TEST MODE ---
TEST_LIMIT_ARG=""
if [ "$1" == "test" ]; then
    TEST_LIMIT_ARG="--category_limit 16"
    echo "RUNNING IN TEST MODE (LIMIT 16 per category)"
fi

# ==============================================================================
# EVALUATE TRY-ON (Qwen3-VL) — dataset_ultimate → correct/incorrect split
# Input  : dataset_ultimate/{diff}/{gender}/tryon_image/IMAGE_FILE.png
#          dataset_ultimate/{diff}/{gender}/initial_person_image/IMAGE_FILE.png
#          dataset_ultimate/{diff}/{gender}/cloth_image/IMAGE_FILE.png
# VLM order : Image A = initial_person_image | Image B = cloth_image | Image C = tryon_image
# Output : correct triplets stay in dataset_ultimate/
#          incorrect triplets + JSON (improved_prompt from VLM) move to dataset_ultimate_1/
# Threshold : 0.6
# ==============================================================================
echo "Running Try-On Evaluator (Qwen3-VL) on dataset_ultimate..."
srun -u ${CONTAINER_ARG} bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID

    python vton/tryon/core/edit_evaluator.py \\
        --dataset_v3_dir ${DATASET_V3_DIR} \\
        --dataset_ultimate_1_dir ${DATASET_ULTIMATE_1_DIR} \\
        --prompt_file ${TRYON_EVAL_PROMPT_FILE} \\
        --model_name ${MODEL_PATH_QWEN} \\
        --threshold 0.6 \\
        --batch_size 8 \\
        --max_new_tokens 300 \\
        ${TEST_LIMIT_ARG}
"

JOB_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "START TIME : $JOB_START"
echo "END TIME   : $JOB_END"
echo "TRY-ON EVALUATION COMPLETE."
echo "============================================================"
