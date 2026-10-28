#!/bin/bash
# ==============================================================================
# Non-curriculum training on CURVTON: easy + hard tiers
# Same settings as train_non_curriculum.sh (NPROC, BATCH_SIZE, LR, MAX_STEPS,
# DATA_PERCENT, ...); extra arguments are passed to train.py.
#
# Usage:
#   CURVTON_DATA_PATH=/path/to/dataset_ultimate bash scripts/train_easy_hard.sh
# ==============================================================================
set -euo pipefail
DIFFICULTY=easy_hard exec bash "$(dirname "$0")/train_non_curriculum.sh" "$@"
