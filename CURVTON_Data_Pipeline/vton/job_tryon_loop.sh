#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --job-name=vton_tryon_loop
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=PATH_TO_LOG_DIR/%A_vton_tryon_loop.log
#SBATCH --error=PATH_TO_LOG_DIR/%A_vton_tryon_loop.err
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
export ROOT_DIR=$(python3 -c "from vton.config import ROOT_DIR; print(ROOT_DIR)")
export MODEL_PATH_QWEN=$(python3 -c "from vton.config import MODEL_PATH_QWEN; print(MODEL_PATH_QWEN)")
export MODEL_PATH_FLUX=$(python3 -c "from vton.config import MODEL_PATH_FLUX; print(MODEL_PATH_FLUX)")
export TRYON_EVAL_PROMPT_FILE=$(python3 -c "from vton.config import TRYON_EVAL_PROMPT_FILE; print(TRYON_EVAL_PROMPT_FILE)")
export CAMERA_CONTROL_JSON=$(python3 -c "from vton.config import CAMERA_CONTROL_JSON; print(CAMERA_CONTROL_JSON)")
export ROOT_DIR=PATH_TO_DATASET_V3
export DATASET_V3_DIR=PATH_TO_DATASET_V3
export DATASET_END_DIR=PATH_TO_DATASET_END
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

stage_wait() {
    echo "Stage complete. Syncing disk and waiting for GPU memory release..."
    sync
    sleep 10
    echo "Ready for next stage."
}

# --- TEST MODE ---
TEST_LIMIT_ARG=""
if [ "$1" == "test" ]; then
    TEST_LIMIT_ARG="--category_limit 16"
    echo "RUNNING IN TEST MODE (LIMIT 16 per category, total 96)"
fi

# ==============================================================================
# TRY-ON ITERATIVE LOOP (FLUX -> VLM -> ORGANIZE)
#   1. edit_generator.py : generate a try-on for every pair in dataset_ultimate (the pool)
#                          that has none; uses retry_prompt/{name}.json when present
#   2. edit_evaluator.py : accept (sidecar JSON) or reject (move to dataset_ultimate_1)
#   3. rename.py         : accepted -> DATASET_END_DIR; rejected -> back into the pool
#                          with the VLM improved_prompt; failed images archived in
#                          dataset_rejected/iter_N/
# Pairs still in the pool after MAX_ITERATIONS never passed evaluation.
# Phases 4-5 then run on DATASET_END_DIR (accepted triplets only).
# ==============================================================================
echo "Starting Try-On Generation & Evaluation Loop... (16 nodes, 64 GPUs)"
MAX_ITERATIONS=3

for ITER in $(seq 1 $MAX_ITERATIONS); do
    echo "============================================================"
    echo "ITERATION $ITER / $MAX_ITERATIONS"
    echo "============================================================"

    # --- 1. GENERATE (FLUX) ---
    echo "Phase 1: Running Try-On Generator (Flux.2)..."
    srun -u ${CONTAINER_ARG} bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        
        python vton/tryon/core/edit_generator.py \
            --root_dir ${ROOT_DIR} \
            --dataset_v3_dir ${DATASET_V3_DIR} \
            --model_path ${MODEL_PATH_FLUX} \
            --iteration $ITER \
            --batch_size 1 \
            --num_workers 16 \
            ${TEST_LIMIT_ARG}
    "
    stage_wait


    # --- 2. EVALUATE (QWEN VLM) ---
    echo "Phase 2: Running Try-On Evaluator (Qwen3-VL)..."
    srun -u ${CONTAINER_ARG} bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        
        python vton/tryon/core/edit_evaluator.py \
            --dataset_v3_dir ${DATASET_V3_DIR} \
            --dataset_ultimate_1_dir ${DATASET_ULTIMATE_1_DIR} \
            --prompt_file ${TRYON_EVAL_PROMPT_FILE} \
            --threshold 0.6 \
            --model_name ${MODEL_PATH_QWEN} \
            --batch_size 8 \
            --max_new_tokens 300 \
            ${TEST_LIMIT_ARG}
    "
    stage_wait

    # --- 3. ORGANIZE (RENAME SCRIPT) ---
    echo "Phase 3: Moving accepted try-ons to DATASET_END_DIR, re-queuing rejected ones..."
    srun -u ${CONTAINER_ARG} --nodes=1 --ntasks=1 bash -c "
        python vton/tryon/core/rename.py \
            --dataset_v3_dir ${DATASET_V3_DIR} \
            --dataset_ultimate_1_dir ${DATASET_ULTIMATE_1_DIR} \
            --dataset_end_dir ${DATASET_END_DIR} \
            --iteration $ITER
    "
    stage_wait

done

# ==============================================================================
# PHASE 4: POST TRY-ON CAMERA EDIT (QWEN) 15% SAMPLE
# ==============================================================================
echo "============================================================"
echo "Phase 4: Running Qwen Camera Edit on 15% Medium/Hard..."
echo "============================================================"
srun -u ${CONTAINER_ARG} bash -c "
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    
    python vton/tryon/core/post_camera_edit.py \
        --root_dir ${ROOT_DIR} \
        --dataset_end_dir ${DATASET_END_DIR} \
        --camera_control_json ${CAMERA_CONTROL_JSON} \
        --percentage 0.15 \
        --steps 4 \
        --num_workers 16
"
stage_wait

# ==============================================================================
# PHASE 5: FINAL CLOTH RENAMING (CLEANUP)
# ==============================================================================
echo "============================================================"
echo "Phase 5: Renaming Cloth Files to _upper, _lower, _dress..."
echo "============================================================"
srun -u ${CONTAINER_ARG} --nodes=1 --ntasks=1 bash -c "
    python vton/tryon/core/final_cloth_rename.py \
        --root_dir ${ROOT_DIR} \
        --dataset_end_dir ${DATASET_END_DIR}
"
stage_wait

JOB_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "START TIME : $JOB_START"
echo "END TIME   : $JOB_END"
echo "TRY-ON ITERATIVE LOOP COMPLETE."
echo "============================================================"
