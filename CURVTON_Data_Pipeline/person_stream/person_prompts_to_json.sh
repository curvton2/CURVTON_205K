#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --time 12:00:00
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 400G
#SBATCH --output PATH_TO_LOG_DIR/%A_humans_json.log
#SBATCH --error  PATH_TO_LOG_DIR/%A_humans_json.err

# -- Environment --
export HF_HOME=PATH_TO_HF_HOME
export PYTHONUNBUFFERED=PYTHONUNBUFFERED_VALUE
export NCCL_IB_DISABLE=IB_DISABLE_FLAG

# -- Distributed setup --
# Find the master node to coordinate the 8 GPUs across 2 nodes
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=MASTER_PORT_NUMBER
export WORLD_SIZE=NUM_PROCESSES

echo "============================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Nodes     : $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "MASTER    : $MASTER_ADDR:$MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "START TIME: $(date)"
echo "============================================================"

# Ensure directories exist before launching workers
mkdir -p PATH_TO_LOG_DIR
mkdir -p PATH_TO_TARGET_DIR

# Launch 8 parallel tasks (1 per GPU)
# drop_last=False is handled by the DataLoader in the Python script
srun -u \
    --environment=qwen3vl-fa2 \
    bash -c "
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        
        python PATH_TO_QWEN_EDIT_PROMPTS_JSON \
            --root_dir   PATH_TO_ROOT_DIR \
            --target_dir PATH_TO_TARGET_DIR \
            --batch_size 20 \
            --model_name 'Qwen/Qwen3-VL-32B-Instruct'
    "

echo "============================================================"
echo "END TIME   : $(date)"
echo "============================================================"
