#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --job-name=vton_phase1_edit
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=PATH_TO_LOG_DIR/%A_vton_edit.log
#SBATCH --error=PATH_TO_LOG_DIR/%A_vton_edit.err
#SBATCH --mem=400GB

# ── Environment ──────────────────────────────────────────────
export HF_HOME=PATH_TO_HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME
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
export ROOT_DIR=$(python3 -c "from vton.config import ROOT_DIR; print(ROOT_DIR)")
export HUMANS_DIR=$(python3 -c "from vton.config import HUMANS_DIR; print(HUMANS_DIR)")
export JSONL_SRC=$(python3 -c "from vton.config import JSONL_SRC_DIR; print(JSONL_SRC_DIR)")
export MODEL_PATH_QWEN=$(python3 -c "from vton.config import MODEL_PATH_QWEN; print(MODEL_PATH_QWEN)")
export MODEL_PATH_FLUX=$(python3 -c "from vton.config import MODEL_PATH_FLUX; print(MODEL_PATH_FLUX)")
export EDIT_EVAL_PROMPT_FILE=$(python3 -c "from vton.config import EDIT_EVAL_PROMPT_FILE; print(EDIT_EVAL_PROMPT_FILE)")

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

stage_wait() {
    echo "Stage complete. Syncing disk and waiting for GPU memory release..."
    sync
    sleep 10
    echo "Ready for next stage."
}

# --- TEST MODE ---
TEST_LIMIT_ARG=""
if [ "$1" == "test" ]; then
    TEST_LIMIT_ARG="--test_limit 2000"
fi

# ==============================================================================
# PHASE 1: PREPARATION & EDITING  (16 nodes × 4 GPUs = 64 workers)
# ==============================================================================
echo "Starting PHASE 1: Human Prep & Editing... (16 nodes, 64 GPUs)"

# 1.1 Consolidate Metadata (single process)
srun -u ${CONTAINER_ARG} --ntasks=1 python vton/edit/consolidate_metadata.py

# 1.2 Generate Edit Prompts (Qwen3-VL-32B on 64 GPUs)
#     --gpus-per-task=1 gives each task exactly 1 GPU with proper CUDA_VISIBLE_DEVICES
#     No manual CUDA_VISIBLE_DEVICES or sleep stagger needed!
srun -u ${CONTAINER_ARG} bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    echo \"[RANK \$RANK | GPU \$LOCAL_RANK] on \$(hostname)\"

    python vton/edit/generate_edit_prompts.py \\
        --humans_root ${HUMANS_DIR} \\
        --jsonl_dir ${JSONL_SRC} \\
        --output_dir ${ROOT_DIR}/edited/prompts \\
        --model_name ${MODEL_PATH_QWEN} \\
        --batch_size 8 \\
        --start_batch 0 \\
        --max_new_tokens 300 \\
        ${TEST_LIMIT_ARG}
"
stage_wait

# 1.3 Flux Generation (FLUX.2-klein-9B on 64 GPUs)
#     Flux model may not be in local cache, so disable offline mode for this step
srun -u ${CONTAINER_ARG} bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    unset HF_HUB_OFFLINE
    unset TRANSFORMERS_OFFLINE
    echo \"[RANK \$RANK | GPU \$LOCAL_RANK] on \$(hostname)\"

    python vton/edit/edit_generator.py \\
        --root_dir ${ROOT_DIR} \\
        --model_path ${MODEL_PATH_FLUX} \\
        --batch_size 1 \\
        ${TEST_LIMIT_ARG}
"
stage_wait

# 1.4 Quality Evaluation (Qwen3-VL-32B on 64 GPUs)
srun -u ${CONTAINER_ARG} bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    echo \"[RANK \$RANK | GPU \$LOCAL_RANK] on \$(hostname)\"

    python vton/edit/edit_evaluator.py \\
        --root_dir ${ROOT_DIR} \\
        --prompt_file ${EDIT_EVAL_PROMPT_FILE} \\
        --threshold 0.8 \\
        --model_name ${MODEL_PATH_QWEN} \\
        --batch_size 8 \\
        --max_new_tokens 300 \\
        ${TEST_LIMIT_ARG}
"
stage_wait

# 1.5 Finalize Names (single process)
srun -u ${CONTAINER_ARG} --ntasks=1 python vton/edit/rename.py

JOB_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "START TIME : $JOB_START"
echo "END TIME   : $JOB_END"
echo "PHASE 1 COMPLETE."
echo "============================================================"
