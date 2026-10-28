#!/bin/bash
echo "[Node $SLURM_NODEID] Starting torchrun on $(hostname)"

# CD into correct directory
cd PATH_TO_WORKDIR

# Confirm FA2 and Torch before running
python -c "import flash_attn; print('[Node $SLURM_NODEID] flash_attn:', flash_attn.__version__)"
python -c "import torch; print('[Node $SLURM_NODEID] torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())"
python -c "from diffusers import Flux2KleinPipeline; print('[Node $SLURM_NODEID] Flux2KleinPipeline OK')"

# Torchrun to distribute across 4 local GPUs (1 node)
# --nnodes=1 defines a single node explicitly
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  generate_cloth.py
