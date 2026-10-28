# Extras — dataset-complexity outputs not reported in the paper

The paper's diversity table is produced by `../compute_metrics.py`. This folder keeps everything else the metric code can compute, for further analysis.

| File | What it computes |
|---|---|
| `m6_appearance.py` | **M6 Appearance**: face / identity embedding diversity (`appearance_*`, `n_faces`) |
| `m8_vae_latent.py` | **M8 VAE latent**: diversity of Stable Diffusion VAE latents (`vae_*`) |
| `unified_index.py` | A single combined complexity index across all metrics |
| `compute_all_metrics.py` | Full report on benchmark datasets: every output of M1–M9, the unified index and per-category scores |
| `compute_curvton_all_metrics.py` | Full report on CURVTON tiers and sample ratios (multi-GPU) |

Secondary outputs of the paper metrics that appear only in the full reports:

| Metric | Outputs not in the paper |
|---|---|
| M1 Pose | `pose_diversity`, `pose_diversity_angle`, `pose_diversity_end_effector`, `pose_diversity_logdet_*`, `pose_diversity_uncertainty`, `pose_artic_*`, `pose_uncertainty_mean` |
| M2 Occlusion | per-occluder ratios (`occlusion_body_parts`, `_carried_objects`, `_accessories`, `_environment`, `_other_people`, `_other`), `occlusion_mean`, `occlusion_var`, `person_occlusion_total`, `person_coverage`, `self_occlusion` |
| M3 Background | `bg_entropy_var`, `bg_semantic_entropy_mean` / `_var`, `bg_semantic_unique_mean` / `_var`, `bg_complexity_semantic`, `bg_overall_complexity`, `bg_difficulty_label` |
| M4 Illumination | `illumination_complexity`, `illumination_richness_score`, `luminance_mean_global` |
| M5 Body shape | `shape_variance_total`, `shape_diversity_logdet_normalized` / `_raw` |
| M7 Garment | `garment_diversity_logdet_normalized` / `_raw`, `garment_effective_rank`, `garment_ensemble_score_0_1` |
| M9 Camera | `azimuth_mean`, `azimuth_weighted_mean`, `elevation_mean`, `azimuth_entropy`, `frontal_ratio`, `side_ratio`, `back_ratio`, `camera_diversity_score`, `estimation_confidence` |

Run from `Evaluation/Dataset_Complexity_Metrics/`:

```bash
python extras/compute_all_metrics.py --config configs/pretrained_metrics_datasets.yaml --output_dir results/all_metrics
python extras/compute_curvton_all_metrics.py --base_path /path/to/dataset_ultimate --out_dir results/curvton_all

# Test M6 / M8 on their own
python test_metric.py --metric m6
python test_metric.py --metric m8
```
