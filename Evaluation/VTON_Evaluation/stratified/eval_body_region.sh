#!/bin/bash
# ==============================================================================
# Stratified evaluation by body region: upper_body / lower_body / dresses
#
# Builds the split with stratify_by_body_region.py (skipped if BODY_REGION_ROOT
# already exists), then runs ../zeroshot/vton_metrics/evaluate_vton.py.
#
# Generated try-ons: $GENERATED_ROOT/<model>/<upper_body|lower_body|dresses>/<male|female>/tryon_image/
# (same file names as the real try-ons).
#
# Usage:
#   CURVTON_ROOT=/path/to/dataset_ultimate_test GENERATED_ROOT=/path/to/generated \
#   MODELS="catvton idmvton" bash Evaluation/VTON_Evaluation/stratified/eval_body_region.sh
# Extra arguments are passed to evaluate_vton.py.
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

: "${CURVTON_ROOT:?Set CURVTON_ROOT to the CURVTON test root (contains easy/ medium/ hard/)}"
: "${GENERATED_ROOT:?Set GENERATED_ROOT to the folder holding <model>/<stratum>/ outputs}"
BODY_REGION_ROOT="${BODY_REGION_ROOT:-${CURVTON_ROOT%/}_body_region}"
MODELS="${MODELS:-catvton}"
OUTPUT_DIR="${OUTPUT_DIR:-./stratified_eval_results/body_region}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"
DEVICE="${DEVICE:-auto}"

if [ ! -d "$BODY_REGION_ROOT" ]; then
    python "$HERE/stratify_by_body_region.py" --curvton-root "$CURVTON_ROOT" --output-root "$BODY_REGION_ROOT"
fi

# shellcheck disable=SC2086
python "$HERE/../zeroshot/vton_metrics/evaluate_vton.py" \
    --category-root "$BODY_REGION_ROOT" \
    --generated-root "$GENERATED_ROOT" \
    --models $MODELS \
    --datasets category \
    --sample-ratio "$SAMPLE_RATIO" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    "$@"
