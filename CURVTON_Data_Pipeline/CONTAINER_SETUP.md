Step 1: Create the Dockerfile

On the login node:

mkdir -p $SCRATCH/fa2-build
nano $SCRATCH/fa2-build/Dockerfile


Paste the following:

FROM BASE_CONTAINER_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel

# Core ML packages
RUN pip install --root-user-action=ignore \
    accelerate \
    transformers \
    datasets \
    tokenizers \
    safetensors \
    huggingface_hub \
    sentencepiece \
    peft \
    trl

# Inference utilities
RUN pip install --root-user-action=ignore \
    qwen-vl-utils \
    Pillow \
    numpy \
    requests

# Flash Attention 2 (prebuilt accelerator wheel)
RUN pip install --root-user-action=ignore --no-deps \
    PATH_TO_FLASH_ATTN_WHEEL

# Sanity checks
RUN python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
RUN python -c "import torch; print('torch:', torch.__version__)"

Step 2: Build on a Compute Node

Container builds must not be done on the login node.

srun -A SLURM_ACCOUNT --pty bash


Then:

cd $SCRATCH/fa2-build
podman build -t qwen3vl-fa2:latest .


Convert the image to SquashFS for Enroot:

mkdir -p $SCRATCH/ce-images

enroot import -x mount \
  -o $SCRATCH/ce-images/qwen3vl-fa2.sqsh \
  podman://qwen3vl-fa2:latest

exit

Step 3: Create EDF Environment File

Create the EDF config:

nano $HOME/.edf/qwen3vl-fa2.toml

image = "PATH_TO_CONTAINER_IMAGE"

mounts = [
    "PATH_TO_SHARED_MOUNT_1",
    "PATH_TO_SHARED_MOUNT_2",
]

workdir = "PATH_TO_WORKDIR"

Step 4: Interactive Test
srun --environment=qwen3vl-fa2 --pty -A SLURM_ACCOUNT bash


Inside the container:

python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "from transformers import Qwen3VLForConditionalGeneration; print('transformers OK')"


Exit when finished:

exit

Step 5: Slurm Job Script

Create the job file:

nano PATH_TO_RUN_SCRIPT

#!/bin/bash --login
#SBATCH --account SLURM_ACCOUNT
#SBATCH --partition normal
#SBATCH --time 4:00:00
#SBATCH --output PATH_TO_LOG_DIR/%A.log
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem 400G

export HF_HOME=PATH_TO_HF_HOME

srun -u --environment=qwen3vl-fa2 bash -c "
    echo '=== Environment Check ==='
    python -c 'import flash_attn; print(\"flash_attn:\", flash_attn.__version__)'
    python -c 'import torch; print(\"torch:\", torch.__version__, \"| GPUs:\", torch.cuda.device_count())'
    echo '========================='
    cd PATH_TO_WORKDIR
    python qwen_hf_beans.py
"


Submit the job:

sbatch PATH_TO_RUN_SCRIPT


Monitor logs:

tail -f PATH_TO_LOG_DIR/<jobid>.log

Using FlashAttention-2 in Code

Inside qwen_hf_beans.py, this now works correctly:

attn_implementation = "flash_attention_2"


No fallback, no silent disabling.

Risk Note (Important)

FlashAttention wheel is built for CUDA 12.6

Cluster host uses CUDA 13.x

No conflict occurs because:

The container bundles its own CUDA runtime

Execution is fully container-isolated

This is the same mechanism used by official vendor containers.