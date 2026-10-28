#!/bin/bash
#SBATCH --job-name=vton_phase2_loop
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --kill-on-bad-exit=1
#SBATCH --output=%x_%j.out

# --- PATH CONFIGURATION ---
WORKING_DIR=PATH_TO_WORKDIR
cd ${WORKING_DIR}

# --- CONFIGURATION ---
export ROOT_DIR=$(python3 -c "from vton.config import ROOT_DIR; print(ROOT_DIR)")
export ULTIMATE_DIR=$(python3 -c "from vton.config import ULTIMATE_DIR; print(ULTIMATE_DIR)")
export INCORRECT_DIR=$(python3 -c "from vton.config import INCORRECT_DIR; print(INCORRECT_DIR)")
export MODEL_PATH_QWEN=$(python3 -c "from vton.config import MODEL_PATH_QWEN; print(MODEL_PATH_QWEN)")
export MODEL_PATH_VTON=$(python3 -c "from vton.config import MODEL_PATH_VTON; print(MODEL_PATH_VTON)")

# --- CONTAINER CONFIGURATION ---
CONTAINER_ARG="--environment=qwen3vl-fa2"

gpu_cleanup() {
    echo "Cleaning up VRAM..."
    wait
    sleep 10
    srun ${CONTAINER_ARG} --ntasks=${SLURM_NTASKS} bash -c "pkill -u $USER -f python || true"
    srun ${CONTAINER_ARG} --ntasks=${SLURM_NTASKS} nvidia-smi > /dev/null
}

# --- TEST MODE ---
TEST_LIMIT_ARG=""
if [ "$1" == "test" ]; then
    TEST_LIMIT_ARG="--test_limit 2000"
fi

# ==============================================================================
# PHASE 2: VTON ITERATIVE LOOP
# ==============================================================================
echo "Starting PHASE 2: VTON Iterative Loop (3 Iterations)..."

for ITER in {1..3}
do
    echo "ITERATION ${ITER} STARTING"

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

echo "PHASE 2 COMPLETE. Final results in: ${ULTIMATE_DIR}"
