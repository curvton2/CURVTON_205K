#!/bin/bash
# ==============================================================================
# Non-curriculum (baseline) training on CURVTON
#
# Batches are drawn uniformly from the selected difficulty split for the whole
# run (--curriculum none). Use DIFFICULTY to pick the tiers and DATA_PERCENT to
# train on a fraction of them. The tier / percentage scripts below call this one:
#   train_easy_only.sh  train_medium_only.sh  train_hard_only.sh
#   train_easy_medium.sh  train_easy_hard.sh  train_medium_hard.sh
#   train_data_percent.sh
#
# Usage:
#   CURVTON_DATA_PATH=/path/to/dataset_ultimate bash scripts/train_non_curriculum.sh
#   DIFFICULTY=hard NPROC=8 CURVTON_DATA_PATH=... bash scripts/train_non_curriculum.sh --use_dream
# Any extra arguments are passed straight to train.py.
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # CURVTON_Experiments/

: "${CURVTON_DATA_PATH:?Set CURVTON_DATA_PATH to the CURVTON train root (contains easy/ medium/ hard/)}"
CURVTON_TEST_DATA_PATH="${CURVTON_TEST_DATA_PATH:-}"

DIFFICULTY="${DIFFICULTY:-all}"           # all | easy | medium | hard | easy_medium | easy_hard | medium_hard
DATA_PERCENT="${DATA_PERCENT:-100}"       # % of the selected training data to use (e.g. 10, 20, 50, 100)
NPROC="${NPROC:-4}"                       # GPUs on this node
TRAIN_MODE="${TRAIN_MODE:-full_unet}"     # full_unet | attention_only
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-30000}"
RUN_NAME="${RUN_NAME:-curvton_${TRAIN_MODE}_bs${BATCH_SIZE}_${DIFFICULTY}_${DATA_PERCENT}pct_none}"
DATA_FRACTION=$(awk -v p="$DATA_PERCENT" 'BEGIN { if (p <= 0 || p > 100) exit 1; printf "%g", p / 100 }') \
    || { echo "DATA_PERCENT must be in (0, 100] (got '$DATA_PERCENT')"; exit 1; }

ARGS=(
    --dataset curvton
    --curvton_data_path "$CURVTON_DATA_PATH"
    --difficulty "$DIFFICULTY"
    --data_fraction "$DATA_FRACTION"
    --curriculum none
    --train_mode "$TRAIN_MODE"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --max_steps "$MAX_STEPS"
    --run_name "$RUN_NAME"
)
[ -n "$CURVTON_TEST_DATA_PATH" ] && ARGS+=(--curvton_test_data_path "$CURVTON_TEST_DATA_PATH")

echo "Non-curriculum training: difficulty=$DIFFICULTY | data=${DATA_PERCENT}% | max_steps=$MAX_STEPS | GPUs=$NPROC"
torchrun --standalone --nproc_per_node="$NPROC" train.py "${ARGS[@]}" "$@"
