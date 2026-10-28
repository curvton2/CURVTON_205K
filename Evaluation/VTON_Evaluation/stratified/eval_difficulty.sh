#!/bin/bash
# ==============================================================================
# Stratified evaluation by difficulty: easy / medium / hard / overall
#
# The CURVTON test set is already split by difficulty, so no stratification
# step is needed. Metrics (SSIM, PSNR, LPIPS, FID, KID) come from
# ../zeroshot/vton_metrics/evaluate_vton.py.
#
# Generated try-ons: $GENERATED_ROOT/<model>/<easy|medium|hard>/<male|female>/tryon_image/
# (same file names as the real try-ons).
#
# Usage:
#   CURVTON_ROOT=/path/to/dataset_ultimate_test GENERATED_ROOT=/path/to/generated \
#   MODELS="catvton idmvton" bash Evaluation/VTON_Evaluation/stratified/eval_difficulty.sh
# Extra arguments are passed to evaluate_vton.py.
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

: "${CURVTON_ROOT:?Set CURVTON_ROOT to the CURVTON test root (contains easy/ medium/ hard/)}"
: "${GENERATED_ROOT:?Set GENERATED_ROOT to the folder holding <model>/<stratum>/ outputs}"
MODELS="${MODELS:-catvton}"
OUTPUT_DIR="${OUTPUT_DIR:-./stratified_eval_results/difficulty}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"
DEVICE="${DEVICE:-auto}"

# shellcheck disable=SC2086
python "$HERE/../zeroshot/vton_metrics/evaluate_vton.py" \
    --curvton-root "$CURVTON_ROOT" \
    --generated-root "$GENERATED_ROOT" \
    --models $MODELS \
    --datasets curvton \
    --sample-ratio "$SAMPLE_RATIO" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    "$@"
