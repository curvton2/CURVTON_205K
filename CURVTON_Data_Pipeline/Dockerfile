FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential curl git wget tmux \
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

# Flash Attention 2 — prebuilt aarch64 wheel from abacus.ai
# Built for GH200/Hopper, works with pytorch 24.x/25.x + CUDA 12.x
RUN pip install --root-user-action=ignore --no-deps \
    "https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/flash_attn-2.7.2.post1-cp312-cp312-linux_aarch64.whl"

# Verify
RUN python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
RUN python -c "import torch; print('torch:', torch.__version__)"