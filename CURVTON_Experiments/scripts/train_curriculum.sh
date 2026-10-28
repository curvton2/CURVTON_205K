#!/bin/bash
# ==============================================================================
# Curriculum training on CURVTON (Easy -> Medium -> Hard)
#
# Each batch is drawn from the easy / medium / hard splits with weights that
# change over 3 stages of STAGE_STEPS steps each (see curriculum_weights in
# utils.py):
#   hard / soft          : 100% easy -> 60/40 easy/medium -> 30/30/40 easy/medium/hard
#   reverse / soft_reverse: 100% hard -> 40/60 medium/hard -> 30/30/40
#   hard / reverse switch at stage boundaries; soft / soft_reverse blend linearly.
#
# Usage:
#   CURVTON_DATA_PATH=/path/to/dataset_ultimate bash scripts/train_curriculum.sh
#   CURRICULUM=hard NPROC=8 CURVTON_DATA_PATH=... bash scripts/train_curriculum.sh --use_dream
# Any extra arguments are passed straight to train.py.
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # CURVTON_Experiments/

: "${CURVTON_DATA_PATH:?Set CURVTON_DATA_PATH to the CURVTON train root (contains easy/ medium/ hard/)}"
CURVTON_TEST_DATA_PATH="${CURVTON_TEST_DATA_PATH:-}"

CURRICULUM="${CURRICULUM:-soft}"         # hard | soft | reverse | soft_reverse
NPROC="${NPROC:-4}"                       # GPUs on this node
TRAIN_MODE="${TRAIN_MODE:-full_unet}"     # full_unet | attention_only
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
MAX_STEPS="${MAX_STEPS:-30000}"
STAGE_STEPS="${STAGE_STEPS:-$((MAX_STEPS / 3))}"   # 3 stages span the whole run
HARD_PCT="${HARD_PCT:-}"                  # optional, reverse / soft_reverse only (e.g. 80)
RUN_NAME="${RUN_NAME:-curvton_${TRAIN_MODE}_bs${BATCH_SIZE}_${CURRICULUM}}"

case "$CURRICULUM" in
    hard|soft|reverse|soft_reverse) ;;
    *) echo "CURRICULUM must be hard, soft, reverse or soft_reverse (got '$CURRICULUM')"; exit 1 ;;
esac

ARGS=(
    --dataset curvton
    --curvton_data_path "$CURVTON_DATA_PATH"
    --difficulty all
    --curriculum "$CURRICULUM"
    --stage_steps "$STAGE_STEPS"
    --train_mode "$TRAIN_MODE"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --max_steps "$MAX_STEPS"
    --run_name "$RUN_NAME"
)
[ -n "$CURVTON_TEST_DATA_PATH" ] && ARGS+=(--curvton_test_data_path "$CURVTON_TEST_DATA_PATH")
[ -n "$HARD_PCT" ] && ARGS+=(--hard_pct "$HARD_PCT")

echo "Curriculum training: $CURRICULUM | stage_steps=$STAGE_STEPS | max_steps=$MAX_STEPS | GPUs=$NPROC"
torchrun --standalone --nproc_per_node="$NPROC" train.py "${ARGS[@]}" "$@"
