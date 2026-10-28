# Dataset Complexity Metrics

Pretrained metrics (M1–M9) that measure how varied and difficult a virtual try-on dataset is, plus the plotting scripts used for the CURVTON Easy / Medium / Hard analysis and a VLM-based complexity scorer.

---

## 📁 Project Structure

The metrics reported in the paper sit at the top of this folder; everything else is in `extras/`.

```
Evaluation/Dataset_Complexity_Metrics/
├── compute_metrics.py              # The 13 diversity numbers reported in the paper (per dataset / CURVTON tier)
├── pose_diversity.py               # Pose diversity σ²_pose exactly as defined in the paper
├── pretrained_metrics/
│   ├── metrics/                    # Metric backends behind the paper numbers
│   │   ├── m1_pose.py              #   (keypoint extractor used by pose_diversity.py)
│   │   ├── m2_occlusion.py
│   │   ├── m3_background.py
│   │   ├── m4_illumination.py
│   │   ├── m5_body_shape.py
│   │   ├── m7_garment_texture.py
│   │   └── m9_camera_angle.py
│   ├── cache_setup.py              # Model/HF cache locations
│   └── dataloader.py
├── plot_scripts/                   # Paper figures (Easy / Medium / Hard)
├── curvton_cache_autogen.py        # Builds the NPZ feature caches plot_scripts/ read
├── extras/                         # Outputs NOT reported in the paper (see extras/README.md)
├── configs/                        # YAML dataset configs (paths to fill in)
├── dataloaders/curvton_dataloader.py  # CURVTON loader (easy / medium / hard, male / female)
├── datasets/                       # Benchmark dataset loaders and registry (VITON-HD, DressCode, StreetTryOn, LAION, …)
├── test_metric.py                  # Run one metric (--metric m1 … m9) on every dataset in a config
├── metric_test_common.py           # Shared CLI and loop for test_metric.py
└── config.py                       # Default dataset roots (used by extras/compute_all_metrics.py)
```

---

## 🚀 Installation

From the repository root:

```bash
pip install -r requirements.txt   # single requirements file at the repository root
```

Run the commands below from inside `Evaluation/Dataset_Complexity_Metrics/`.

---

## 📊 Metrics reported in the paper

`compute_metrics.py` writes exactly the columns of the paper's "Task-free dataset diversity comparison" table:

| Dimension | Paper column | Output key | Computed by |
|---|---|---|---|
| Pose | σ²_pose | `pose` | `pose_diversity.py`: mean over 8 joints of the bend-angle variance (deg²) |
| Occlusion | Occl. Cmplx. / Person Vis. | `occl_cmplx` / `person_vis` | M2 |
| Background | BGE / BSE / BOD | `bge` / `bse` / `bod` | M3 |
| Illumination | IGM / LVM | `igm` / `lvm` | M4 |
| Body shape | σ²_shape | `shape` | M5 |
| Garment | Gmt Var. / Gmt Div. | `gmt_var` / `gmt_div` | M7 |
| Camera | σ̂_φ / σ̂_θ | `camera_azimuth` / `camera_elevation` | M9 |

```bash
# CURVTON, per difficulty tier (easy / medium / hard / all)
python compute_metrics.py --curvton_root /path/to/dataset_ultimate_test --output results/curvton.json

# A benchmark dataset (vitonhd, dresscode, street_tryon, ... — see datasets/loaders.py)
python compute_metrics.py --dataset vitonhd --root /path/to/vitonhd --output results/vitonhd.json

# Pose diversity on its own
python pose_diversity.py --curvton_root /path/to/dataset_ultimate_test
```

### Testing a single metric

```bash
python test_metric.py --metric m1 --config configs/pretrained_metrics_datasets.yaml --device cuda
python test_metric.py --metric m9 --curvton_root /path/to/dataset_ultimate
```

---

## ➕ Extras (not reported in the paper)

`extras/` holds M6 appearance, M8 VAE latent, the unified complexity index, and the full-report scripts that print every secondary output of M1–M9. See [`extras/README.md`](extras/README.md).

---

## 🎨 Paper Figures (`plot_scripts/`)

Each script compares the Easy / Medium / Hard splits on a 5 % sample. The scripts read per-sample NPZ caches from `--cache-dir` (default `./eda_cache/curvton`). If a cache is missing, pass `--curvton-root` and it is computed automatically by `curvton_cache_autogen.py`:

```bash
python plot_scripts/pose_diversity_easy_medium_hard.py \
    --curvton-root /path/to/dataset_ultimate \
    --out-dir figures/pose

python plot_scripts/occlusion_heatmap_easy_medium_hard.py \
    --curvton-root /path/to/dataset_ultimate \
    --occ-maps-dir ./eda_cache/curvton_occ \
    --out-dir figures/occlusion
```

| Script | Cache keys | Metric |
|---|---|---|
| `pose_diversity_easy_medium_hard.py` | `pose_vecs`, `angles` | M1 |
| `occlusion_heatmap_easy_medium_hard.py` | `occ_maps` | M2 |
| `background_complexity_eda.py` | `bg_entropy`, `bg_obj_count` | M3 |
| `illumination_pca_overlay_easy_medium_hard.py` | `lum_maps` | M4 |
| `body_shape_variation_easy_medium_hard.py` | `betas` | M5 |
| `camera_angle_overlay_easy_medium_hard.py`, `camera_angle_scatter_easy_medium_hard.py` | `azimuths`, `elevations` | M9 |

Extra options for cache generation: `--autogen-device`, `--autogen-batch-size`, `--autogen-num-workers`, `--autogen-seed`.

---

## 📦 CURVTON Layout

```
dataset_ultimate/
├── easy/
│   ├── female/
│   │   ├── cloth_image/
│   │   ├── initial_person_image/
│   │   └── tryon_image/
│   └── male/
├── medium/
└── hard/
```

---

## 📋 Configuration Files

| Config | Used by |
|---|---|
| `pretrained_metrics_datasets.yaml` | `test_metric.py` (default) and `extras/compute_all_metrics.py` |
| `benchmark_test.yaml` | `extras/compute_all_metrics.py` on the standard benchmarks (VITON-HD, DressCode, LAION, StreetTryOn) |

Dataset roots in these files point to the original cluster and need to be changed to your own locations.

---

## 📄 License

This project is for research purposes.
