#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --time 6:00:00
#SBATCH --output PATH_TO_LOG_DIR/%A_score.log
#SBATCH --error  PATH_TO_LOG_DIR/%A_score.err
#SBATCH --nodes 2                  # 2 nodes
#SBATCH --ntasks-per-node=4        # 4 tasks per node = 8 total GPU workers
#SBATCH --gpus-per-task=1          # 1 GPU per task
#SBATCH --cpus-per-task=8          # 8 CPU cores per task (matches num_workers=8)
#SBATCH --mem 400G

# ── HuggingFace cache ──────────────────────────────────────────────────────
export HF_HOME=PATH_TO_HF_HOME
export PYTHONUNBUFFERED=PYTHONUNBUFFERED_VALUE
export NCCL_IB_DISABLE=IB_DISABLE_FLAG           # keep InfiniBand for cross-node traffic

# ── SLURM / distributed env ───────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER
export WORLD_SIZE=NUM_PROCESSES                # 2 nodes × 4 GPUs

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR=PATH_TO_SCRIPT_DIR
EDIT_OUTPUTS=PATH_TO_EDIT_OUTPUTS
OUTPUT_DIR=PATH_TO_SCORE_OUTPUTS

mkdir -p "$OUTPUT_DIR"
mkdir -p PATH_TO_LOG_DIR

JOB_START=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "MASTER       : $MASTER_ADDR:$MASTER_PORT"
echo "WORLD_SIZE   : $WORLD_SIZE"
echo "START TIME   : $JOB_START"
echo "Edit outputs : $EDIT_OUTPUTS"
echo "Score output : $OUTPUT_DIR"
echo "============================================================"

# ── Launch 8 parallel GPU workers (4 per node) ────────────────────────────
# srun distributes tasks across the 2 nodes automatically.
# Each task sees its own single GPU via CUDA_VISIBLE_DEVICES.
srun -u \
    --environment=qwen3vl-fa2 \
    bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID

        echo \"[RANK \$RANK / LOCAL_RANK \$LOCAL_RANK] on \$(hostname) | GPU: \$CUDA_VISIBLE_DEVICES\"

        cd ${SCRIPT_DIR}
        python vlm_score.py \
            --edit_outputs_dir ${EDIT_OUTPUTS} \
            --output_dir       ${OUTPUT_DIR} \
            --model_name       'Qwen/Qwen3-VL-32B-Instruct' \
            --batch_size       32 \
            --num_workers      8 \
            --max_new_tokens   64 \
            --image_size       512
    "

EXIT_CODE=$?
JOB_END=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "START TIME  : $JOB_START"
echo "END TIME    : $JOB_END"
echo "Exit code   : $EXIT_CODE"

START_EPOCH=$(date -d "$JOB_START" +%s)
END_EPOCH=$(date -d "$JOB_END" +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "Wall time   : ${ELAPSED}s ($(( ELAPSED/60 ))m $(( ELAPSED%60 ))s)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "STATUS      : SUCCESS"
    echo ""
    echo "Per-rank score files:"
    ls -lh "${OUTPUT_DIR}"/scores_rank*.jsonl 2>/dev/null
    echo ""
    echo "Total records scored:"
    cat "${OUTPUT_DIR}"/scores_rank*.jsonl 2>/dev/null | wc -l
    echo ""
    echo "Complexity summary:"
    cat "${OUTPUT_DIR}/complexity_summary.json" 2>/dev/null || echo "(summary not yet written)"
else
    echo "STATUS      : FAILED — check logs"
fi
echo "============================================================"
