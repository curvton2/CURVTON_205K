#!/bin/bash
# ==============================================================================
# Non-curriculum training on a percentage of CURVTON (all three tiers).
# The subset is sampled at random from easy + medium + hard (paper ablation:
# 10%, 20%, 50%, 100%).
#
# Usage:
#   PERCENT=10 CURVTON_DATA_PATH=/path/to/dataset_ultimate bash scripts/train_data_percent.sh
# Same settings as train_non_curriculum.sh (NPROC, BATCH_SIZE, LR, MAX_STEPS, DIFFICULTY, ...);
# extra arguments are passed to train.py.
# ==============================================================================
set -euo pipefail
: "${PERCENT:?Set PERCENT to the share of training data to use, e.g. PERCENT=10}"
DATA_PERCENT="$PERCENT" exec bash "$(dirname "$0")/train_non_curriculum.sh" "$@"
