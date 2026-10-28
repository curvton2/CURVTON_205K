# Setup Guide

This document contains the step-by-step procedure to set up the environment and run the project.

For project overview and architecture, see:
- [README.md](./README.md)

## 1. Prerequisites

- Python 3.10+ recommended
- `pip` available
- Optional: NVIDIA GPU + CUDA for accelerated training/inference

Check Python:

```bash
python --version
```

## 2. Create Virtual Environment

From repository root:

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Optional GPU Setup

Install CUDA-enabled PyTorch that matches your system:
- https://pytorch.org/get-started/locally/

Validate GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 5. Validate the Entry Points

```bash
python CURVTON_Experiments/train.py --help
python CURVTON_Experiments/inference.py --help
python Evaluation/VTON_Evaluation/trained/unified_eval.py --help
```

## 6. Run Core Workflows

### 6.1 Training

```bash
# Curriculum (CURRICULUM=hard|soft|reverse|soft_reverse, default soft)
CURVTON_DATA_PATH=/path/to/dataset_ultimate bash CURVTON_Experiments/scripts/train_curriculum.sh

# Non-curriculum baseline (DIFFICULTY=all|easy|medium|hard, default all)
CURVTON_DATA_PATH=/path/to/dataset_ultimate bash CURVTON_Experiments/scripts/train_non_curriculum.sh
```

### 6.2 Inference

```bash
python CURVTON_Experiments/inference.py --checkpoint /path/to/ckpt.pt --person person.png --cloth cloth.png --output result.png
```

### 6.3 Evaluation

```bash
python Evaluation/VTON_Evaluation/trained/unified_eval.py --target base -- --help
```

## 7. Run Grouped Evaluation Experiments

### 7.1 Zero-shot

```bash
python Evaluation/VTON_Evaluation/zeroshot/run_zeroshot_eval.py --help
```

### 7.2 Trained checkpoint metrics

```bash
python Evaluation/VTON_Evaluation/trained/run_trained_eval.py --help
```

## 8. M1-M9 Pretrained Metrics

Examples:

```bash
python Evaluation/Dataset_Complexity_Metrics/test_metric.py --metric m1 --help
python Evaluation/Dataset_Complexity_Metrics/compute_paper_metrics.py --help
```

## 9. Deactivate Environment

```bash
deactivate
```

