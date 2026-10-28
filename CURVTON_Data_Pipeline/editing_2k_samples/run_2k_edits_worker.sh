#!/bin/bash
echo "[Node $SLURM_NODEID] Starting torchrun on $(hostname)"

# Confirm FA2 is live on this node before starting torchrun
python -c "import flash_attn; print('[Node $SLURM_NODEID] flash_attn:', flash_attn.__version__)"
python -c "import torch; print('[Node $SLURM_NODEID] torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())"
python -c "from diffusers import Flux2KleinPipeline; print('[Node $SLURM_NODEID] Flux2KleinPipeline OK')"

torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  generate_2k_edits.py