#!/bin/bash
# ==============================================================================
# Stratified evaluation by garment traditionality: traditional / non_traditional
#
# Builds the split with stratify_by_traditionality.py (skipped if
# TRADITIONALITY_ROOT already exists), then runs ../zeroshot/vton_metrics/evaluate_vton.py.
#
# Generated try-ons: $GENERATED_ROOT/<model>/<traditional|non_traditional>/<male|female>/tryon_image/
# (same file names as the real try-ons).
#
# Usage:
#   CURVTON_ROOT=/path/to/dataset_ultimate_test GENERATED_ROOT=/path/to/generated \
#   MODELS="catvton idmvton" bash Evaluation/VTON_Evaluation/stratified/eval_traditionality.sh
# Extra arguments are passed to evaluate_vton.py.
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

: "${CURVTON_ROOT:?Set CURVTON_ROOT to the CURVTON test root (contains easy/ medium/ hard/)}"
: "${GENERATED_ROOT:?Set GENERATED_ROOT to the folder holding <model>/<stratum>/ outputs}"
TRADITIONALITY_ROOT="${TRADITIONALITY_ROOT:-${CURVTON_ROOT%/}_traditionality}"
MODELS="${MODELS:-catvton}"
OUTPUT_DIR="${OUTPUT_DIR:-./stratified_eval_results/traditionality}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"
DEVICE="${DEVICE:-auto}"

if [ ! -d "$TRADITIONALITY_ROOT" ]; then
    python "$HERE/stratify_by_traditionality.py" --curvton-root "$CURVTON_ROOT" --output-root "$TRADITIONALITY_ROOT"
fi

# shellcheck disable=SC2086
python "$HERE/../zeroshot/vton_metrics/evaluate_vton.py" \
    --traditionality-root "$TRADITIONALITY_ROOT" \
    --generated-root "$GENERATED_ROOT" \
    --models $MODELS \
    --datasets traditionality \
    --sample-ratio "$SAMPLE_RATIO" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    "$@"
