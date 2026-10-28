"""
extras/compute_all_metrics.py
==================================================
Main entry-point for the Pretrained-Model Dataset Complexity Pipeline.

Computes 7 dataset-complexity metrics for each of the 10 try-on datasets:

  ┌────┬──────────────────────────────┬────────────────────────────────────┐
  │ M  │ Metric Family                │ Key outputs                        │
  ├────┼──────────────────────────────┼────────────────────────────────────┤
  │ 1  │ Pose Diversity               │ D_pose (log-det cov)               │
  │    │ Pose Articulation Complexity │ C_artic (Σ joint-angle var)        │
  │ 2  │ Occlusion Complexity         │ C_occ = E[O] + Var(O)              │
  │ 3  │ Background Complexity        │ H_bg (entropy), C_obj (density)    │
  │ 4  │ Illumination Complexity      │ C_light = E[Var(G_i)]              │
  │ 5  │ Body Shape Diversity         │ D_shape (log-det β cov)            │
  │ 6  │ Appearance Diversity         │ D_face (mean cosine distance)      │
  │ 7  │ Garment Texture Diversity    │ D_garment (log-det CLIP cov)       │
  │UCI │ Unified Complexity Index     │ Weighted z-score sum               │
  └────┴──────────────────────────────┴────────────────────────────────────┘

Each dataloader provides batches of:
    { "person": Tensor(B,3,H,W), "cloth": Tensor(B,3,H,W), ... }

Metric modules only see the tensors they need:
  M1, M2, M3, M4, M5, M6  → person_imgs
  M7                        → cloth_imgs

Usage
------
  # Evaluate all 10 datasets via YAML config:
  python extras/compute_all_metrics.py --config configs/pretrained_metrics_datasets.yaml

  # Evaluate a single dataset:
  python extras/compute_all_metrics.py \
      --dataset viton --root /data/VITON \
      --batch_size 16 --device cuda

  # Dry-run smoke-test (synthetic tensors, no real dataset):
  python extras/compute_all_metrics.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Add workspace root to path ───────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))   # workspace root

# Thin shim → delegates entirely to datasets/loaders.py (the only source of truth)
from pretrained_metrics.dataloader import get_dataloader, ALL_DATASETS
from pretrained_metrics.metrics.m1_pose           import PoseMetrics
from pretrained_metrics.metrics.m2_occlusion      import OcclusionMetrics
from pretrained_metrics.metrics.m3_background     import BackgroundMetrics
from pretrained_metrics.metrics.m4_illumination   import IlluminationMetrics
from pretrained_metrics.metrics.m5_body_shape     import BodyShapeMetrics
from extras.m6_appearance     import AppearanceMetrics
from pretrained_metrics.metrics.m7_garment_texture import GarmentTextureMetrics
from extras.m8_vae_latent     import VAELatentMetric
from pretrained_metrics.metrics.m9_camera_angle   import CameraAngleMetrics
from extras.unified_index     import (
    UnifiedComplexityIndex,
    METRIC_KEYS,
    VITON_HD_BASELINES,
    VITON_HD_STDS,
)

# Configuration management
try:
    import config
except ImportError:
    config = None


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation for a single dataset
# ─────────────────────────────────────────────────────────────────────────────

def _all_nan():
    return float("nan")


def _sigmoid_local(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _category_scores_from_result(result: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """
    Build per-category raw/z/normalized scores for the unified metric families.
    Normalization is z-score vs VITON-HD baseline, then sigmoid(z) -> [0,1].
    """
    raw_scores: Dict[str, float] = {}
    z_scores: Dict[str, float] = {}
    norm_scores: Dict[str, float] = {}

    for mk, _ in METRIC_KEYS:
        v = result.get(mk, float("nan"))
        try:
            raw = float(v)
        except Exception:
            raw = float("nan")
        raw_scores[mk] = raw

        mu = float(VITON_HD_BASELINES.get(mk, 0.0))
        sig = float(max(VITON_HD_STDS.get(mk, 1.0), 1e-6))
        if isinstance(raw, float) and not math.isnan(raw):
            z = (raw - mu) / sig
            z_scores[mk] = z
            norm_scores[mk] = _sigmoid_local(z)
        else:
            z_scores[mk] = float("nan")
            norm_scores[mk] = float("nan")

    valid = [v for v in norm_scores.values() if not (isinstance(v, float) and math.isnan(v))]
    overall = float(sum(valid) / len(valid)) if valid else float("nan")

    return {
        "raw": raw_scores,
        "z": z_scores,
        "normalized_0_1": norm_scores,
        "overall_mean_0_1": overall,
    }


def evaluate_one_dataset(
    dataset_name: str,
    root: str,
    cfg: dict,
) -> Dict:
    """
    Run all pretrained-metric modules on a single dataset.
    Returns a flat dict of metric values.
    """
    device     = cfg.get("device", "cpu")
    batch_size = cfg.get("batch_size", 16)
    if batch_size < 16:
        print(f"  [Config] batch_size={batch_size} < 16; using 16.")
        batch_size = 16
    num_workers= cfg.get("num_workers", 4)
    img_size   = tuple(cfg.get("img_size", [512, 384]))
    split      = cfg.get("split", "test")
    use_overlay  = cfg.get("use_overlay", False)

    # Selective metric flags
    run_pose    = cfg.get("run_pose", True)
    run_occ     = cfg.get("run_occ", True)
    run_bg      = cfg.get("run_bg", True)
    run_illum   = cfg.get("run_illum", True)
    run_shape   = cfg.get("run_shape", True)
    run_appear  = cfg.get("run_appear", True)
    run_garment = cfg.get("run_garment", True)
    run_vae     = cfg.get("run_vae", True)
    run_camera  = cfg.get("run_camera", True)
    m6_face_detector_backend = cfg.get("m6_face_detector_backend", "auto")
    m6_retinaface_repo_dir = cfg.get("m6_retinaface_repo_dir", None)
    m6_retinaface_weights = cfg.get("m6_retinaface_weights", None)
    m6_retinaface_backbone = cfg.get("m6_retinaface_backbone", "mobilenetv1_0.25")
    m6_retinaface_device = cfg.get("m6_retinaface_device", "cpu")

    # Resolve root if not provided
    if not root and config:
        root = config.get_root(dataset_name)

    # Use dedicated overlay version if flag is set
    if use_overlay and not dataset_name.endswith("_overlay"):
        if dataset_name.lower() in ["dresscode", "vitonhd", "laion"]:
            dataset_name = f"{dataset_name.lower()}_overlay"

    print(f"\n{'='*65}")
    print(f"  Dataset : {dataset_name.upper()}")
    print(f"  Root    : {root}")
    print(f"  Batch   : {batch_size}  |  Device : {device}  |  Split : {split}")
    print(f"{'='*65}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    is_dresscode = "dresscode" in dataset_name.lower()
    dresscode_cat = cfg.get("dresscode_category", "upper_body") if is_dresscode else None
    loaders: List = []
    try:
        if is_dresscode and str(dresscode_cat).lower() == "all":
            for cat in ["upper_body", "lower_body", "dresses"]:
                try:
                    ld = get_dataloader(
                        dataset_name, root,
                        split=split,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        img_size=img_size,
                        category=cat,
                    )
                    loaders.append((cat, ld))
                except Exception as e:
                    print(f"  [WARN] DressCode category '{cat}' skipped: {e}")
            if not loaders:
                print("  [SKIP] No DressCode categories could be loaded.")
                return {}
            n_samples = sum(len(ld.dataset) for _, ld in loaders)
        else:
            ld = get_dataloader(
                dataset_name, root,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                img_size=img_size,
                **({} if not is_dresscode else {"category": dresscode_cat}),
            )
            loaders = [(str(dresscode_cat) if is_dresscode else "default", ld)]
            n_samples = len(ld.dataset)
    except (FileNotFoundError, RuntimeError, Exception) as e:
        print(f"  [SKIP] {e}")
        import traceback
        traceback.print_exc()
        return {}

    print(f"  Samples : {n_samples}")

    # ── Metric objects ────────────────────────────────────────────────────────
    def _loaded(msg: str):
        print(f"  [Model] {msg} loaded")

    m1 = PoseMetrics(device=device)           if run_pose    else None
    if m1: _loaded("M1 Pose")
    m2 = OcclusionMetrics(device=device)      if run_occ     else None
    if m2: _loaded("M2 Occlusion")
    m3 = BackgroundMetrics(device=device)     if run_bg      else None
    if m3: _loaded("M3 Background")
    m4 = IlluminationMetrics()                if run_illum   else None
    if m4: _loaded("M4 Illumination")
    m5 = BodyShapeMetrics(device=device)      if run_shape   else None
    if m5: _loaded("M5 BodyShape")
    m6 = AppearanceMetrics(
        device=device,
        face_detector_backend=m6_face_detector_backend,
        retinaface_repo_dir=m6_retinaface_repo_dir,
        retinaface_weights=m6_retinaface_weights,
        retinaface_backbone=m6_retinaface_backbone,
        retinaface_device=m6_retinaface_device,
    ) if run_appear else None
    if m6: _loaded("M6 Appearance")
    m7 = GarmentTextureMetrics(device=device) if run_garment else None
    if m7: _loaded("M7 GarmentTexture")
    m8 = VAELatentMetric(device=device)       if run_vae     else None
    if m8: _loaded("M8 VAE")
    m9 = CameraAngleMetrics(device=device)    if run_camera  else None
    if m9: _loaded("M9 CameraAngle")

    # ── Batch loop ────────────────────────────────────────────────────────────
    t0 = time.time()
    for cat_name, loader in loaders:
        desc_name = dataset_name if not is_dresscode else f"{dataset_name}:{cat_name}"
        for batch in tqdm(loader, desc=f"  {desc_name}", unit="batch"):
            person = batch["person"].float()   # (B,3,H,W)
            cloth  = batch["cloth"].float()    # (B,3,H,W)

            if m1: m1.update(person)
            if m2: m2.update(person)
            if m3: m3.update(person)
            if m4: m4.update(person)
            if m5: m5.update(person)
            if m6: m6.update(person)
            if m7: m7.update(cloth)
            if m8: m8.update(person)
            if m9: m9.update(person)

    elapsed = time.time() - t0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    r1 = m1.compute() if m1 else {}
    r2 = m2.compute() if m2 else {}
    r3 = m3.compute() if m3 else {}
    r4 = m4.compute() if m4 else {}
    r5 = m5.compute() if m5 else {}
    r6 = m6.compute() if m6 else {}
    r7 = m7.compute() if m7 else {}
    r8 = m8.compute() if m8 else {}
    r9 = m9.compute() if m9 else {}

    dresscode_cat = cfg.get("dresscode_category") if is_dresscode else None
    result = {
        "dataset":   dataset_name,
        "split":     split,
        **({"dresscode_category": dresscode_cat} if dresscode_cat else {}),
        "n_samples": n_samples,
        "elapsed_s": round(elapsed, 2),
        **r1, **r2, **r3, **r4, **r5, **r6, **r7, **r8, **r9,
    }
    # Derived variance terms (when only std is exposed by backend).
    if "appearance_diversity_std" in result:
        sd = result.get("appearance_diversity_std")
        if isinstance(sd, (int, float)) and not math.isnan(float(sd)):
            result["appearance_diversity_var"] = float(sd) * float(sd)
    if "azimuth_std" in result:
        sd = result.get("azimuth_std")
        if isinstance(sd, (int, float)) and not math.isnan(float(sd)):
            result["azimuth_var"] = float(sd) * float(sd)
    if "elevation_std" in result:
        sd = result.get("elevation_std")
        if isinstance(sd, (int, float)) and not math.isnan(float(sd)):
            result["elevation_var"] = float(sd) * float(sd)
    cat_scores = _category_scores_from_result(result)
    result["category_metrics_raw"] = cat_scores["raw"]
    result["category_metrics_zscore"] = cat_scores["z"]
    result["category_metrics_normalized_0_1"] = cat_scores["normalized_0_1"]
    result["category_complexity_mean_0_1"] = cat_scores["overall_mean_0_1"]

    _print_result_box(result)
    print("  Metrics JSON:")
    print(json.dumps(result, indent=2, default=str))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_KEYS = [
    # M1
    ("pose_diversity",              "Pose Diversity (log-det)"),
    ("pose_diversity_logdet_raw",   "Pose Diversity logdet raw"),
    ("pose_diversity_logdet_normalized", "Pose Diversity logdet norm"),
    ("pose_artic_complexity",       "Pose Artic. Complexity"),
    ("pose_artic_mean_per_image",   "Pose Artic. Mean/img"),
    # M2
    ("occlusion_mean",              "Occlusion Mean"),
    ("occlusion_var",               "Occlusion Var"),
    ("occlusion_complexity",        "Occlusion Complexity"),
    ("occlusion_body_parts",        "Occ. Body Parts Mean"),
    ("occlusion_carried_objects",   "Occ. Carried Obj Mean"),
    ("occlusion_accessories",       "Occ. Accessories Mean"),
    ("occlusion_environment",       "Occ. Environment Mean"),
    ("occlusion_other_people",      "Occ. Other People Mean"),
    ("occlusion_other",             "Occ. Other Mean"),
    # M3
    ("bg_entropy_mean",             "BG Entropy Mean"),
    ("bg_entropy_var",              "BG Entropy Var"),
    ("bg_object_density_mean",      "BG Object Density"),
    ("bg_semantic_entropy_mean",    "BG Semantic Entropy Mean"),
    ("bg_semantic_entropy_var",     "BG Semantic Entropy Var"),
    ("bg_semantic_unique_mean",     "BG Semantic Unique Mean"),
    ("bg_semantic_unique_var",      "BG Semantic Unique Var"),
    # M4
    ("luminance_mean_global",       "Luminance Mean"),
    ("luminance_var_global",        "Luminance Var"),
    ("illumination_gradient_mean",  "Illum. Gradient Mean"),
    ("illumination_richness_score", "Illum. Richness Score"),
    ("illumination_complexity",     "Illum. Complexity"),
    # M5
    ("shape_diversity_logdet",      "Shape Diversity (log-det)"),
    ("shape_diversity_logdet_raw",  "Shape Diversity logdet raw"),
    ("shape_diversity_logdet_normalized", "Shape Diversity logdet norm"),
    ("shape_variance_total",        "Shape Variance Total"),
    # M6
    ("appearance_diversity_mean",   "Appearance Diversity Mean"),
    ("appearance_diversity_std",    "Appearance Diversity Std"),
    ("appearance_diversity_var",    "Appearance Diversity Var"),
    ("appearance_diversity_logdet_raw", "Appearance logdet raw"),
    ("appearance_diversity_logdet_normalized", "Appearance logdet norm"),
    # M7
    ("garment_diversity_logdet",    "Garment Diversity (log-det)"),
    ("garment_diversity_logdet_raw", "Garment Diversity logdet raw"),
    ("garment_diversity_logdet_normalized", "Garment Diversity logdet norm"),
    ("garment_variance_total",      "Garment Variance Total"),
    # M8
    ("vae_diversity_logdet",        "VAE Diversity (log-det)"),
    ("vae_diversity_logdet_raw",    "VAE Diversity logdet raw"),
    ("vae_diversity_logdet_normalized", "VAE Diversity logdet norm"),
    ("vae_variance_total",          "VAE Variance Total"),
    # M9
    ("azimuth_std",                 "Camera Azimuth Std"),
    ("azimuth_var",                 "Camera Azimuth Var"),
    ("elevation_mean",              "Camera Elevation Mean"),
    ("elevation_std",               "Camera Elevation Std"),
    ("elevation_var",               "Camera Elevation Var"),
    ("camera_diversity_score",      "Camera Diversity Score"),
]


def _fmt(v):
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.4f}"


def _print_result_box(r: dict):
    W = 62
    name = r["dataset"].upper()
    if "dresscode_category" in r:
        name = f"{name} [{r['dresscode_category']}]"
    split = r.get("split")
    if split:
        name = f"{name} | split={split}"
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │{'Results — ' + name:^{W}}│")
    print(f"  ├{'─'*W}┤")
    for key, label in DISPLAY_KEYS:
        if key in r:
            print(f"  │  {label:<38} {_fmt(r[key]):>20}  │")
            norm = r.get("category_metrics_normalized_0_1", {}).get(key, float("nan"))
            if not (isinstance(norm, float) and math.isnan(norm)):
                print(f"  │    {'normalized_0_1':<36} {_fmt(norm):>20}  │")
    if "category_complexity_mean_0_1" in r:
        print(f"  ├{'─'*W}┤")
        print(f"  │  {'Category Complexity Mean (0-1)':<38} {_fmt(r['category_complexity_mean_0_1']):>20}  │")
    print(f"  └{'─'*W}┘")


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run smoke-test (no real dataset needed)
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(device: str = "cpu"):
    """Validates the pipeline using random image tensors."""
    print("\n" + "=" * 60)
    print("  DRY RUN — Synthetic Tensors (No Dataset Required)")
    print("=" * 60)

    B, H, W = 4, 256, 192
    person = torch.rand(B, 3, H, W)
    cloth  = torch.rand(B, 3, H, W)

    metrics = [
        ("Pose",       PoseMetrics(device=device)),
        ("Occlusion",  OcclusionMetrics(device=device)),
        ("Background", BackgroundMetrics(device=device)),
        ("Illumination", IlluminationMetrics()),
        ("BodyShape",  BodyShapeMetrics(device=device)),
        ("Appearance", AppearanceMetrics(device=device)),
        ("GarmentTex", GarmentTextureMetrics(device=device)),
        ("VAE",        VAELatentMetric(device=device)),
        ("Camera",     CameraAngleMetrics(device=device)),
    ]

    N_ITER = 3
    for name, m in metrics:
        print(f"\n  ── {name} ──")
        for _ in range(N_ITER):
            if hasattr(m, "update"):
                if name == "GarmentTex":
                    m.update(cloth)
                else:
                    m.update(person)
        res = m.compute()
        for k, v in res.items():
            print(f"    {k:<40} {_fmt(v) if isinstance(v, float) else v}")

    print("\n  ✓ Dry run complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_key(name: str, cfg: dict) -> str:
    """Unique string key for a dataset entry (includes dresscode category)."""
    cat = cfg.get("dresscode_category", "")
    return f"{name}__{cat}" if cat else name


def _load_checkpoint(output_dir: str) -> dict:
    """
    Load existing checkpoint from output_dir/checkpoint.json.
    Returns a dict mapping checkpoint_key → result dict.
    """
    path = Path(output_dir) / "checkpoint.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        # data is {key: result_dict}
        print(f"  [Resume] Loaded {len(data)} completed dataset(s) from {path}")
        return data
    except Exception as e:
        print(f"  [Resume] Could not read checkpoint ({e}). Starting fresh.")
        return {}


def _write_checkpoint(checkpoint: dict, output_dir: str):
    """Persist the checkpoint dict to disk (overwrite)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "checkpoint.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def _save(all_results: List[dict], output_dir: str, uci_scores: List[dict],
          uci: "UnifiedComplexityIndex" = None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── Comprehensive JSON (raw + normalized + normalization params) ────────
    if uci is not None:
        comprehensive = uci.get_comprehensive_results(uci_scores)
        comprehensive["timestamp"] = ts
        comp_path = out / f"pretrained_metrics_comprehensive_{ts}.json"
        with open(comp_path, "w") as f:
            json.dump(comprehensive, f, indent=2, default=str)
        print(f"\n  Comprehensive results (raw + normalized) saved → {comp_path}")

    # ── Legacy JSON (backward compatible) ──────────────────────────────────
    json_path = out / f"pretrained_metrics_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({"metrics": all_results, "unified_scores": uci_scores}, f, indent=2, default=str)
    print(f"  Raw metrics + unified scores saved → {json_path}")

    # ── CSV with both raw and normalized columns ───────────────────────────
    try:
        import pandas as pd
        rows = []
        for i, r in enumerate(all_results):
            row = {k: v for k, v in r.items()}
            if i < len(uci_scores):
                row["unified_score"] = uci_scores[i].get("unified_score", float("nan"))
                # Add normalized scores and z-scores as separate columns
                for mk, norm_v in uci_scores[i].get("normalised_metrics", {}).items():
                    row[f"{mk}_normalized"] = norm_v
                for mk, z_v in uci_scores[i].get("z_scores", {}).items():
                    row[f"{mk}_zscore"] = z_v
            rows.append(row)
        df = pd.DataFrame(rows)
        csv = out / f"pretrained_metrics_{ts}.csv"
        df.to_csv(csv, index=False)
        print(f"  CSV (raw + normalized columns) saved → {csv}")
    except ImportError:
        pass

    # ── Per-dataset JSONs (in addition to consolidated files) ───────────────
    def _tok(v: object, default: str = "na") -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            s = default
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        return s.strip("._-") or default

    per_ds_dir = out / "per_dataset_json"
    per_ds_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for i, r in enumerate(all_results):
        ds = _tok(r.get("dataset"), "dataset")
        sp = _tok(r.get("split"), "split")
        cat = _tok(r.get("dresscode_category"), "")
        suffix = f"_{cat}" if cat else ""
        fn = f"{ds}_{sp}{suffix}_{ts}.json"
        path = per_ds_dir / fn

        payload = {
            "dataset": r.get("dataset"),
            "split": r.get("split"),
            "dresscode_category": r.get("dresscode_category"),
            "timestamp": ts,
            "raw_metrics": r,
        }
        if i < len(uci_scores):
            payload["uci"] = uci_scores[i]

        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        written += 1

    if written:
        print(f"  Per-dataset JSONs saved → {per_ds_dir} ({written} files)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Pretrained-Model Dataset Complexity Pipeline (7 metrics, 10 datasets)"
    )
    p.add_argument("--config",    type=str, default=None,
                   help="YAML config (multi-dataset mode).")
    p.add_argument("--dataset",   type=str, default=None,
                   help=f"Single dataset name. One of: {ALL_DATASETS}")
    p.add_argument("--root",      type=str, default=None,
                   help="Dataset root directory.")
    p.add_argument("--dry_run",   action="store_true",
                   help="Smoke-test with random tensors (no dataset needed).")
    p.add_argument("--output_dir",type=str, default=".")
    p.add_argument("--batch_size",type=int, default=16)
    p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--img_size",  type=int, nargs=2, default=[512, 384], metavar=("H","W"))
    p.add_argument("--split",     type=str, default="test")
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dresscode_category", type=str, default="upper_body")
    p.add_argument("--use_overlay", action="store_true", help="Use dedicated dataloaders from datasets/overlay_loaders.py")
    p.add_argument("--no_resume",  action="store_true",
                   help="Ignore existing checkpoint and re-evaluate all datasets.")
    # Selective metrics
    p.add_argument("--no_pose",    action="store_true")
    p.add_argument("--no_occ",     action="store_true")
    p.add_argument("--no_bg",      action="store_true")
    p.add_argument("--no_illum",   action="store_true")
    p.add_argument("--no_shape",   action="store_true")
    p.add_argument("--no_appear",  action="store_true")
    p.add_argument("--no_garment", action="store_true")
    p.add_argument("--no_vae",     action="store_true")
    p.add_argument("--no_camera",  action="store_true")
    p.add_argument("--force_appear", action="store_true",
                   help="Force-enable appearance metric for all datasets/config entries.")
    p.add_argument("--m6_face_detector_backend", type=str, default="auto",
                   choices=["auto", "retinaface_pytorch", "insightface", "haar"],
                   help="Face detector backend for M6 appearance metric.")
    p.add_argument("--m6_retinaface_repo_dir", type=str, default=None,
                   help="Path to cloned yakhyo/retinaface-pytorch repo.")
    p.add_argument("--m6_retinaface_weights", type=str, default=None,
                   help="Path to RetinaFace .pth weights (e.g., retinaface_mv1_0.25.pth).")
    p.add_argument("--m6_retinaface_backbone", type=str, default="mobilenetv1_0.25",
                   help="RetinaFace backbone name used by yakhyo repo.")
    p.add_argument("--m6_retinaface_device", type=str, default="cpu",
                   help="Device for RetinaFace detector (cpu or cuda).")
    return p.parse_args()


def main():
    args = _parse()

    if args.dry_run:
        dry_run(args.device)
        return

    base_cfg = dict(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        split=args.split,
        dresscode_category=args.dresscode_category,
        use_overlay=args.use_overlay,
        run_pose=    not args.no_pose,
        run_occ=     not args.no_occ,
        run_bg=      not args.no_bg,
        run_illum=   not args.no_illum,
        run_shape=   not args.no_shape,
        run_appear=  not args.no_appear,
        run_garment= not args.no_garment,
        run_vae=     not args.no_vae,
        run_camera=  not args.no_camera,
        m6_face_detector_backend=args.m6_face_detector_backend,
        m6_retinaface_repo_dir=args.m6_retinaface_repo_dir,
        m6_retinaface_weights=args.m6_retinaface_weights,
        m6_retinaface_backbone=args.m6_retinaface_backbone,
        m6_retinaface_device=args.m6_retinaface_device,
    )
    if args.force_appear:
        base_cfg["run_appear"] = True

    output_dir = args.output_dir

    # ── Checkpoint (resume) ───────────────────────────────────────────────────
    resume = not args.no_resume
    checkpoint: dict = _load_checkpoint(output_dir) if resume else {}
    all_results: List[dict] = list(checkpoint.values())   # seed with already-done

    # ── Multi-dataset YAML ────────────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        defaults = {**base_cfg, **raw.get("defaults", {})}
        for entry in raw.get("datasets", []):
            cfg  = {**defaults, **entry}
            if args.force_appear:
                cfg["run_appear"] = True
            name = cfg.pop("name")
            root = cfg.pop("root")
            cfg.pop("pred_dir", None)

            key = _checkpoint_key(name, cfg)
            if key in checkpoint:
                print(f"\n  [Skip] {name} (already in checkpoint — use --no_resume to recompute)")
                continue

            res = evaluate_one_dataset(name, root, cfg)
            if res:
                all_results.append(res)
                checkpoint[key] = res
                _write_checkpoint(checkpoint, output_dir)   # save after every dataset
                print(f"  [Checkpoint] Saved progress ({len(checkpoint)} dataset(s) done).")

    # ── Single dataset ────────────────────────────────────────────────────────
    elif args.dataset and args.root:
        key = _checkpoint_key(args.dataset, base_cfg)
        if key in checkpoint and resume:
            print(f"\n  [Skip] {args.dataset} already in checkpoint. Use --no_resume to recompute.")
        else:
            res = evaluate_one_dataset(args.dataset, args.root, base_cfg)
            if res:
                all_results.append(res)
                checkpoint[key] = res
                _write_checkpoint(checkpoint, output_dir)

    else:
        print("[INFO] No action specified. Use --dry_run, --dataset+--root, or --config.")
        print(f"[INFO] Available datasets: {ALL_DATASETS}")
        return

    if not all_results:
        print("\n  No results to save.")
        return

    # ── Unified Complexity Index ───────────────────────────────────────────────
    uci = UnifiedComplexityIndex()
    for r in all_results:
        uci.add_dataset(r["dataset"], r)
    uci_scores = uci.compute_scores()
    uci.print_report(uci_scores)

    # ── Final consolidated save ───────────────────────────────────────────────
    _save(all_results, output_dir, uci_scores, uci)


if __name__ == "__main__":
    main()
