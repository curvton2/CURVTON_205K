#!/bin/bash
# ==============================================================================
# Non-curriculum training on CURVTON: easy tier only
# Same settings as train_non_curriculum.sh (NPROC, BATCH_SIZE, LR, MAX_STEPS,
# DATA_PERCENT, ...); extra arguments are passed to train.py.
#
# Usage:
#   CURVTON_DATA_PATH=/path/to/dataset_ultimate bash scripts/train_easy_only.sh
# ==============================================================================
set -euo pipefail
DIFFICULTY=easy exec bash "$(dirname "$0")/train_non_curriculum.sh" "$@"
