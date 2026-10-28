#!/bin/bash
#SBATCH --job-name=full_dataset_pipeline
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --kill-on-bad-exit=1
#SBATCH --output=%x_%j.out

# --- CONFIGURATION (Sourced from vton/config.py via python) ---
export ROOT_DIR=$(python3 -c "from vton.config import ROOT_DIR; print(ROOT_DIR)")
export ULTIMATE_DIR=$(python3 -c "from vton.config import ULTIMATE_DIR; print(ULTIMATE_DIR)")
export INCORRECT_DIR=$(python3 -c "from vton.config import INCORRECT_DIR; print(INCORRECT_DIR)")
export JSONL_SRC=$(python3 -c "from vton.config import JSONL_SRC_DIR; print(JSONL_SRC_DIR)")

export MODEL_PATH_QWEN=$(python3 -c "from vton.config import MODEL_PATH_QWEN; print(MODEL_PATH_QWEN)")
export MODEL_PATH_FLUX=$(python3 -c "from vton.config import MODEL_PATH_FLUX; print(MODEL_PATH_FLUX)")
export MODEL_PATH_VTON=$(python3 -c "from vton.config import MODEL_PATH_VTON; print(MODEL_PATH_VTON)")
export EDIT_EVAL_PROMPT_FILE=$(python3 -c "from vton.config import EDIT_EVAL_PROMPT_FILE; print(EDIT_EVAL_PROMPT_FILE)")

# --- CONTAINER CONFIGURATION ---
CONTAINER_NAME="qwen3vl-fa2"
CONTAINER_ARG="--environment=${CONTAINER_NAME}"

# GPU Memory Cleanup Function
gpu_cleanup() {
    echo "Wait for Stage Completion and Cleaning up VRAM..."
    wait
    sleep 10
    srun ${CONTAINER_ARG} --ntasks=${SLURM_NTASKS} bash -c "pkill -u $USER -f python || true"
    srun ${CONTAINER_ARG} --ntasks=${SLURM_NTASKS} nvidia-smi > /dev/null
    echo "GPU State Synchronized."
}

# --- TEST MODE HANDLING ---
TEST_LIMIT_ARG=""
if [ "$1" == "test" ]; then
    echo "!!! TEST MODE ENABLED: Capping generations at 2000 samples total !!!"
    TEST_LIMIT_ARG="--test_limit 2000"
fi

# ==============================================================================
# PHASE 1: PREPARATION & EDITING (HUMAN PREP)
# ==============================================================================
echo "Starting PHASE 1: Human Editing..."

# 1.1 Consolidate Metadata for VTON matching
echo "Consolidating metadata..."
srun ${CONTAINER_ARG} --ntasks=1 python vton/edit/consolidate_metadata.py

# 1.2 Generate Edit Prompts (Human Instructions)
echo "Generating edit prompts via Qwen..."
srun ${CONTAINER_ARG} python vton/edit/generate_edit_prompts.py \
    --humans_root ${ROOT_DIR}/humans \
    --jsonl_dir ${JSONL_SRC} \
    --output_dir ${ROOT_DIR}/edited/prompts \
    --model_name ${MODEL_PATH_QWEN} \
    ${TEST_LIMIT_ARG}
gpu_cleanup

# 1.3 Generate Edited Images (Flux Generation)
echo "Generating edited human images via Flux..."
srun ${CONTAINER_ARG} python vton/edit/edit_generator.py \
    --root_dir ${ROOT_DIR} \
    --model_path ${MODEL_PATH_FLUX} \
    --batch_size 1 \
    ${TEST_LIMIT_ARG}
gpu_cleanup

# 1.4 Evaluate Edited Images (Quality Control)
echo "Evaluating edits (Correct/Incorrect)..."
srun ${CONTAINER_ARG} python vton/edit/edit_evaluator.py \
    --root_dir ${ROOT_DIR} \
    --prompt_file ${EDIT_EVAL_PROMPT_FILE} \
    --threshold 0.8 \
    --model_name ${MODEL_PATH_QWEN} \
    ${TEST_LIMIT_ARG}
gpu_cleanup

# 1.5 Clean and Rename (Consecutive Naming)
echo "Finalizing edited humans folder..."
srun ${CONTAINER_ARG} --ntasks=1 python vton/edit/rename.py

# ==============================================================================
# PHASE 2: VTON ITERATIVE LOOP (CLOTHING TRY-ON)
# ==============================================================================
echo "Starting PHASE 2: Virtual Try-On Iterative Loop (3 Iterations)..."

for ITER in {1..3}
do
    echo "--------------------------------------------"
    echo "ITERATION ${ITER} STARTING"
    echo "--------------------------------------------"

    # 2.1 VTON Generation
    srun ${CONTAINER_ARG} python vton/vton_generator.py \
        --root_dir ${ROOT_DIR} \
        --jsonl vton/edit/edit.jsonl \
        --model_path ${MODEL_PATH_VTON} \
        ${TEST_LIMIT_ARG}
    gpu_cleanup

    # 2.2 VTON Evaluation & Scoring
    srun ${CONTAINER_ARG} python vton/scorer.py \
        --root_dir ${ROOT_DIR} \
        --jsonl_dir vton/edit \
        --ultimate_dir ${ULTIMATE_DIR} \
        --incorrect_dir ${INCORRECT_DIR} \
        --iteration ${ITER} \
        --model_name ${MODEL_PATH_QWEN} \
        ${TEST_LIMIT_ARG}
    gpu_cleanup

    echo "Iteration ${ITER} complete."
done

echo "=============================================================================="
echo "PIPELINE COMPLETE. Final Results in: ${ULTIMATE_DIR}"
echo "=============================================================================="
