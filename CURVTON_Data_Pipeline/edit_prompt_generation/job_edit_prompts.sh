#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --time 4:00:00
#SBATCH --output PATH_TO_LOG_DIR/%A_edit.log
#SBATCH --error  PATH_TO_LOG_DIR/%A_edit.err
#SBATCH --nodes 2                  # 4 nodes × 4 GPUs = 16 GPUs total
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 400G

# ── Environment ──────────────────────────────────────────────
export HF_HOME=PATH_TO_HF_HOME
export PYTHONUNBUFFERED=PYTHONUNBUFFERED_VALUE
export NCCL_IB_DISABLE=IB_DISABLE_FLAG

# ── Distributed setup ────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER            # different port to avoid clash with beans job
export WORLD_SIZE=NUM_PROCESSES

JOB_START=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Nodes     : $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "MASTER    : $MASTER_ADDR:$MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "START TIME: $JOB_START"
echo "============================================================"

mkdir -p PATH_TO_LOG_DIR
mkdir -p PATH_TO_OUTPUT_DIR

SCRIPT_DIR=PATH_TO_SCRIPT_DIR


# ── Launch 16 parallel GPU workers ───────────────────────────
srun -u \
    --environment=qwen3vl-fa2 \
    bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        echo \"[RANK \$RANK | GPU \$LOCAL_RANK] on \$(hostname)\"

        python PATH_TO_QWEN_EDIT_PROMPTS \
            --json_dir   PATH_TO_JSON_DIR \
            --image_dir  PATH_TO_IMAGE_DIR \
            --output_dir PATH_TO_OUTPUT_DIR \
            --max_samples    3000 \
            --batch_size     24 \
            --max_new_tokens 512 \
            --model_name     'Qwen/Qwen3-VL-32B-Instruct'
    "

EXIT_CODE=$?
JOB_END=$(date '+%Y-%m-%d %H:%M:%S')
START_EPOCH=$(date -d "$JOB_START" +%s)
END_EPOCH=$(date -d "$JOB_END" +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))

echo "============================================================"
echo "START TIME : $JOB_START"
echo "END TIME   : $JOB_END"
echo "Wall time  : ${ELAPSED}s ($(( ELAPSED/60 ))m $(( ELAPSED%60 ))s)"
echo "Exit code  : $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "STATUS     : SUCCESS"
    TXT_COUNT=$(ls PATH_TO_OUTPUT_DIR/*.txt 2>/dev/null | wc -l)
    echo "TXT files  : $TXT_COUNT"
    echo "JSONL files:"
    ls -lh PATH_TO_OUTPUT_DIR/edit_prompts_rank*.jsonl 2>/dev/null
else
    echo "STATUS     : FAILED — check logs"
fi
echo "============================================================"
