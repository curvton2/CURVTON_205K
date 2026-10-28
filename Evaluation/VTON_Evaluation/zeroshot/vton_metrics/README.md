# VTON Evaluation Metrics

A comprehensive evaluation script for **Virtual Try-On (VTON)** models computing SSIM, PSNR, LPIPS, FID, and KID metrics.

## Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **SSIM** | Structural Similarity Index | ↑ Higher |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | ↑ Higher |
| **LPIPS** | Learned Perceptual Image Patch Similarity | ↓ Lower |
| **FID** | Fréchet Inception Distance | ↓ Lower |
| **KID** | Kernel Inception Distance | ↓ Lower |

## Installation

```bash
# from the repository root (single requirements file)
pip install -r requirements.txt
```

## Directory Structure

Generated images should follow this layout:

```
generated_root/
└── <model_name>/
    └── <split_or_category>/
        ├── image1.png
        ├── image2.png
        └── ...
```

## Usage

### CURVTON Dataset (easy / medium / hard / overall)

```bash
python evaluate_vton.py \
    --curvton-root /path/to/curvton_test \
    --generated-root /path/to/generated \
    --models catvton idmvton flux \
    --datasets curvton \
    --sample-ratio 1.0 \
    --overall-per-split 350 \
    --output-dir ./results \
    --seed 42 --device cuda \
    --batch-size 32 --num-workers 8 \
    --log-level INFO
```

### Category Dataset (upper_body / lower_body / dresses)

```bash
python evaluate_vton.py \
    --category-root /path/to/category_dataset \
    --generated-root /path/to/generated \
    --models idmvton \
    --datasets category \
    --sample-ratio 1.0 \
    --output-dir ./results \
    --seed 42 --device cuda
```

### Traditionality Dataset (traditional / non_traditional)

```bash
python evaluate_vton.py \
    --traditionality-root /path/to/traditionality_dataset \
    --generated-root /path/to/generated \
    --models ootd \
    --datasets traditionality \
    --sample-ratio 0.28 \
    --output-dir ./results \
    --seed 42 --device cuda
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--curvton-root` | Path to CURVTON test dataset | — |
| `--category-root` | Path to category-stratified dataset | — |
| `--traditionality-root` | Path to traditionality dataset | — |
| `--generated-root` | Root folder of generated images | — |
| `--models` | Model name(s) to evaluate | — |
| `--datasets` | Dataset(s): `curvton`, `category`, `traditionality` | — |
| `--sample-ratio` | Fraction of pairs to evaluate (0–1] | 0.2 |
| `--overall-per-split` | Samples per split for overall metric | 350 |
| `--output-dir` | Directory to save JSON/CSV results | — |
| `--device` | `cuda` or `cpu` | cuda |
| `--batch-size` | Batch size for LPIPS/FID/KID | 32 |
| `--num-workers` | DataLoader workers | 8 |
| `--image-size` | Resize images to this size | 512 |
| `--seed` | Random seed for reproducibility | 42 |
| `--strict` | Fail if no matched pairs found | False |
| `--log-level` | Logging verbosity | WARNING |

## Output

Results are saved as `vton_metrics.json` and `vton_metrics.csv`:

```json
{
  "idmvton": {
    "curvton": {
      "easy":   {"ssim": 0.817, "lpips": 0.138, "fid": 32.4, "kid": 0.0087, "psnr": 19.24, "matched_images": 1004},
      "medium": {"ssim": 0.803, "lpips": 0.143, "fid": 34.6, "kid": 0.0058, "psnr": 19.03, "matched_images": 1002},
      "hard":   {"ssim": 0.768, "lpips": 0.165, "fid": 38.6, "kid": 0.0032, "psnr": 18.96, "matched_images": 1002},
      "overall":{"ssim": 0.797, "lpips": 0.148, "fid": 34.1, "kid": 0.0050, "psnr": 19.08, "sampled_images": 1050}
    }
  }
}
```

## Dataset Formats Supported

The script auto-detects real image directories using common names (`image/`, `images/`, `tryon_image/`, etc.) and supports gender-stratified layouts (`female/`, `male/` subdirectories).

## Citation

If you use this evaluation code, please cite the datasets and models accordingly.
