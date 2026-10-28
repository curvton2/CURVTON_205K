#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --time 4:00:00
#SBATCH --output PATH_TO_LOG_DIR/%A.log
#SBATCH --error  PATH_TO_LOG_DIR/%A.err
#SBATCH --nodes 4                 # 4 nodes
#SBATCH --ntasks-per-node=4       # 4 tasks per node = 16 total GPU workers
#SBATCH --gpus-per-task=1         # 1 GPU per task
#SBATCH --cpus-per-task=8
#SBATCH --mem 400G

# ── HuggingFace cache ─────────────────────────────────────────
export HF_HOME=PATH_TO_HF_HOME
export PYTHONUNBUFFERED=PYTHONUNBUFFERED_VALUE
export NCCL_IB_DISABLE=IB_DISABLE_FLAG          # keep InfiniBand for cross-node NCCL

# ── Master node = first node in SLURM allocation ──────────────
# Must be reachable by all 8 tasks across both nodes
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER
export WORLD_SIZE=NUM_PROCESSES              # 4 nodes × 4 GPUs

# ── Timestamps ────────────────────────────────────────────────
JOB_START=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Nodes     : $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "MASTER    : $MASTER_ADDR:$MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "START TIME: $JOB_START"
echo "WorkDir   : PATH_TO_WORKDIR"
echo "============================================================"

mkdir -p PATH_TO_LOG_DIR

# ── Launch 8 parallel GPU workers across 2 nodes ──────────────
# srun distributes 4 tasks to node-0 and 4 tasks to node-1.
# Each task gets its own GPU (remapped to cuda:0 via CUDA_VISIBLE_DEVICES).
# NCCL uses InfiniBand for cross-node communication.
srun -u \
    --environment=qwen3vl-fa2 \
    bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID

        echo \"[RANK \$RANK / LOCAL_RANK \$LOCAL_RANK] on \$(hostname) | GPU: \$CUDA_VISIBLE_DEVICES\"

        cd PATH_TO_WORKDIR
        python qwen_hf_beans.py \
            --batch_size     48 \
            --max_new_tokens 512 \
            --max_samples    768 \
            --model_name     'Qwen/Qwen3-VL-32B-Instruct' \
            --output_dir     PATH_TO_OUTPUT_DIR
    "

EXIT_CODE=$?
JOB_END=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo "START TIME : $JOB_START"
echo "END TIME   : $JOB_END"
echo "Exit code  : $EXIT_CODE"

# Compute total wall time in seconds
START_EPOCH=$(date -d "$JOB_START" +%s)
END_EPOCH=$(date -d "$JOB_END" +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "Wall time  : ${ELAPSED}s ($(( ELAPSED/60 ))m $(( ELAPSED%60 ))s)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "STATUS     : SUCCESS"
    echo "Output files:"
    ls -lh PATH_TO_OUTPUT_DIR/beans_predictions_rank*.jsonl 2>/dev/null
    TOTAL=$(cat PATH_TO_OUTPUT_DIR/beans_predictions_rank*.jsonl 2>/dev/null | wc -l)
    echo "Total records: $TOTAL"
else
    echo "STATUS     : FAILED — check logs"
fi
echo "============================================================"
