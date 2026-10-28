"""
extras/unified_index.py
=========================
Final Unified Dataset Complexity & Diversity Index
---------------------------------------------------
Takes raw output from the 9 metrics and synthesizes a comprehensive 
dataset-level report containing:
  - Domain Complexity (0-1)
  - Domain Diversity (0-1)
  - Domain Hybrid Score (Complexity * Diversity)
  - Overall Dataset Complexity & Diversity (Weighted Averages)

Normalization:
  z = (Raw - baseline_mu) / baseline_sigma
  score = sigmoid(z / Temperature)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Ontology mapping: Domain -> (Complexity Key, Diversity Key)
# ─────────────────────────────────────────────────────────────────────────────
METRIC_ONTOLOGY = {
    "1_Pose": {
        "name": "Pose",
        "c_key": "pose_artic_complexity",
        "d_key": "pose_diversity",
        "w": 1.0
    },
    "2_Occlusion": {
        "name": "Occlusion",
        "c_key": "person_occlusion_total",
        "d_key": "occlusion_var",   # Variance acts as a proxy for diversity here
        "w": 1.0
    },
    "3_Background": {
        "name": "Background",
        "c_key": "bg_overall_complexity",
        "d_key": "bg_semantic_entropy_global",
        "w": 1.0
    },
    "4_Illumination": {
        "name": "Illumination",
        "c_key": "illumination_gradient_mean",
        "d_key": "luminance_var_global", 
        "w": 1.0
    },
    "5_BodyShape": {
        "name": "Body Shape",
        "c_key": "shape_complexity_abs",       # Deformation magnitude proxy
        "d_key": "shape_diversity_logdet",
        "w": 1.0
    },
    "6_FaceAppearance": {
        "name": "Face Appearance",
        "c_key": "appearance_diversity_mean",  # Cosine dist mean = inherent difficulty
        "d_key": "appearance_diversity_std",   # Cosine dist std = diversity
        "w": 1.0
    },
    "7_GarmentTexture": {
        "name": "Garment Texture",
        "c_key": "garment_variance_total",     # Pattern complexity
        "d_key": "garment_diversity_logdet_normalized",
        "w": 1.0
    },
    "8_VAELatent": {
        "name": "VAE Latent",
        "c_key": "vae_complexity_abs",         # Overall signal energy
        "d_key": "vae_diversity_logdet_normalized",
        "w": 1.0
    },
    "9_CameraAngle": {
        "name": "Camera Angle",
        "c_key": "camera_complexity_abs",                # Or elevation_mean if present
        "d_key": "camera_diversity_score",
        "w": 1.0
    }
}

# Standard VITON-HD-like baselines for accurate Z-scoring
BASELINES = {
    "pose_artic_complexity":       (0.3, 0.15),
    "pose_diversity":              (-15.0, 5.0),
    "person_occlusion_total":      (0.15, 0.10),
    "occlusion_var":               (0.05, 0.05),
    "bg_overall_complexity":       (0.5, 0.2),
    "bg_semantic_entropy_global":  (2.5, 1.0),
    "illumination_complexity":     (0.05, 0.05),
    "luminance_var_global":        (0.02, 0.02),
    "shape_variance_total":        (5.0, 2.0),
    "shape_diversity_logdet":      (-30.0, 10.0),
    "appearance_diversity_mean":   (0.4, 0.2),
    "appearance_diversity_std":    (0.1, 0.05),
    "garment_variance_total":      (20.0, 10.0),
    "garment_diversity_logdet_normalized": (-5.0, 2.0),
    "vae_variance_total":          (1000.0, 500.0),
    "vae_diversity_logdet_normalized": (-10.0, 5.0),
    "azimuth_std":                 (5.0, 5.0),
    "camera_diversity_score":      (0.5, 0.25)
}

# Backward-compatible exports expected by compute_all_metrics.py.
# These constants were historically provided by this module.
VITON_HD_BASELINES = {k: float(mu) for k, (mu, _sig) in BASELINES.items()}
VITON_HD_STDS = {k: float(sig) for k, (_mu, sig) in BASELINES.items()}
METRIC_KEYS = [(k, k.replace("_", " ").title()) for k in BASELINES.keys()]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)

def _isnan(v) -> bool:
    try:
        return v != v or v is None
    except Exception:
        return True

def _f(v) -> str:
    if _isnan(v):
        return "N/A"
    return f"{float(v):.4f}"

class UnifiedComplexityIndex:
    def __init__(self, target_temp: float = 2.5):
        self._target = max(target_temp, 0.5)
        self._records = []

    def add_dataset(self, name: str, metrics: Dict[str, float]):
        self._records.append({"dataset": name, **metrics})

    def compute_scores(self) -> List[Dict]:
        all_keys = set()
        for dom, struct in METRIC_ONTOLOGY.items():
            all_keys.add(struct["c_key"])
            all_keys.add(struct["d_key"])

        # Determine dynamic temperatures based on max Z-scores
        max_abs_z = {k: 0.0 for k in all_keys}
        for rec in self._records:
            for k in all_keys:
                val = rec.get(k, float('nan'))
                if not _isnan(val):
                    mu, sig = BASELINES.get(k, (0.0, 1.0))
                    sig = max(sig, 1e-6)
                    z = (val - mu) / sig
                    max_abs_z[k] = max(max_abs_z[k], abs(z))

        tau = {}
        for k in all_keys:
            maz = max_abs_z[k]
            tau[k] = max(maz / self._target, 1.0) if maz >= 1e-9 else 1.0

        out = []
        for rec in self._records:
            entry = {
                "dataset": rec["dataset"],
                "domains": {},
                "raw_metrics": {k: v for k, v in rec.items() if k != "dataset"}
            }

            total_c_weight = 0.0
            total_d_weight = 0.0
            sum_c = 0.0
            sum_d = 0.0
            sum_h = 0.0

            for dom, struct in sorted(METRIC_ONTOLOGY.items()):
                ck = struct["c_key"]
                dk = struct["d_key"]
                w = struct["w"]

                # Process Complexity
                raw_c = rec.get(ck, float('nan'))
                c_score = float('nan')
                if not _isnan(raw_c):
                    mu, sig = BASELINES.get(ck, (0.0, 1.0))
                    z_c = (raw_c - mu) / max(sig, 1e-6)
                    c_score = _sigmoid(z_c / tau[ck])

                # Process Diversity
                raw_d = rec.get(dk, float('nan'))
                d_score = float('nan')
                if not _isnan(raw_d):
                    mu, sig = BASELINES.get(dk, (0.0, 1.0))
                    z_d = (raw_d - mu) / max(sig, 1e-6)
                    d_score = _sigmoid(z_d / tau[dk])

                # Hybrid (C * D)
                h_score = float('nan')
                if not _isnan(c_score) and not _isnan(d_score):
                    h_score = c_score * d_score

                entry["domains"][struct["name"]] = {
                    "complexity": c_score,
                    "diversity": d_score,
                    "hybrid": h_score,
                    "raw_c": raw_c,
                    "raw_d": raw_d
                }

                # Accumulate for overall dataset scores
                if not _isnan(c_score):
                    sum_c += c_score * w
                    total_c_weight += w
                if not _isnan(d_score):
                    sum_d += d_score * w
                    total_d_weight += w

            entry["overall_complexity"] = (sum_c / total_c_weight) if total_c_weight > 0 else float('nan')
            entry["overall_diversity"] = (sum_d / total_d_weight) if total_d_weight > 0 else float('nan')
            
            # Overall Hybrid = mean(complexities) * mean(diversities)
            # OR mean(hybrids). We use mean(hybrids) for a truer reflection of per-domain joint performance.
            if total_c_weight > 0 and total_d_weight > 0:
                entry["overall_hybrid"] = entry["overall_complexity"] * entry["overall_diversity"]
            else:
                entry["overall_hybrid"] = float('nan')

            out.append(entry)
        return out

    def print_report(self, scores: List[Dict]):
        W = 110
        print("\\n" + "═" * W)
        print(f"  {'COMPREHENSIVE DATASET EVALUATION (Complexity & Diversity)':^{W-4}}")
        print("═" * W)

        for d in scores:
            print(f"\\n  ► DATASET: {d['dataset'].upper()}")
            print(f"    {'Domain':<25} | {'Complexity (0-1)':>18} | {'Diversity (0-1)':>18} | {'Hybrid (C×D)':>18}")
            print(f"    {'─'*25}─┼─{'─'*18}─┼─{'─'*18}─┼─{'─'*18}")

            for dom_name, ds in d["domains"].items():
                c_str = f"{ds['complexity']:.4f}" if not math.isnan(ds['complexity']) else "N/A"
                d_str = f"{ds['diversity']:.4f}" if not math.isnan(ds['diversity']) else "N/A"
                h_str = f"{ds['hybrid']:.4f}" if not math.isnan(ds['hybrid']) else "N/A"
                print(f"    {dom_name:<25} | {c_str:>18} | {d_str:>18} | {h_str:>18}")

            print(f"    {'═'*89}")
            oa_c = f"{d['overall_complexity']:.4f}" if not math.isnan(d['overall_complexity']) else "N/A"
            oa_d = f"{d['overall_diversity']:.4f}" if not math.isnan(d['overall_diversity']) else "N/A"
            oa_h = f"{d['overall_hybrid']:.4f}" if not math.isnan(d['overall_hybrid']) else "N/A"
            
            print(f"    {'OVERALL DATASET SCORE':<25} | {oa_c:>18} | {oa_d:>18} | {oa_h:>18}")
        
        print("\\n" + "═" * W)
