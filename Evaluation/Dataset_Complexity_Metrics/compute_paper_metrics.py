#!/usr/bin/env python3
"""
compute_paper_metrics.py — The task-free dataset-diversity metrics reported in the paper
(main paper "Task-free dataset diversity comparison"; definitions in the Supplementary).

Only the 13 numbers in that table are computed and written:

    Column            Paper symbol      Source
    pose              sigma^2_pose      pose_diversity.py (mean joint bend-angle variance, deg^2)
    occl_cmplx        Occl. Cmplx.      M2 occlusion_complexity
    person_vis        Person Vis.       M2 person_visibility
    bge               BGE (C)           M3 bg_entropy_mean
    bse               BSE (D)           M3 bg_semantic_entropy_global
    bod               BOD (D)           M3 bg_object_density_mean
    igm               IGM               M4 illumination_gradient_mean
    lvm               LVM (D)           M4 luminance_var_global
    shape             sigma^2_shape     M5 shape_diversity_logdet
    gmt_var           Gmt Var.          M7 garment_variance_total
    gmt_div           Gmt Div.          M7 garment_diversity_logdet
    camera_azimuth    sigma_phi         M9 azimuth_std
    camera_elevation  sigma_theta       M9 elevation_std

Everything else the metric classes can produce (M6 appearance, M8 VAE latent, the
unified complexity index and all secondary outputs) lives in extras/.

Usage
-----
# CURVTON, per difficulty tier
python compute_paper_metrics.py --curvton_root /path/to/dataset_ultimate_test --output results/curvton.json

# A benchmark dataset supported by datasets/loaders.py
python compute_paper_metrics.py --dataset vitonhd --root /path/to/vitonhd --output results/vitonhd.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_diversity import PoseDiversity  # noqa: E402
from pretrained_metrics.metrics.m2_occlusion import OcclusionMetrics  # noqa: E402
from pretrained_metrics.metrics.m3_background import BackgroundMetrics  # noqa: E402
from pretrained_metrics.metrics.m4_illumination import IlluminationMetrics  # noqa: E402
from pretrained_metrics.metrics.m5_body_shape import BodyShapeMetrics  # noqa: E402
from pretrained_metrics.metrics.m7_garment_texture import GarmentTextureMetrics  # noqa: E402
from pretrained_metrics.metrics.m9_camera_angle import CameraAngleMetrics  # noqa: E402

# paper column -> (metric group, output key)
PAPER_COLUMNS = {
    "occl_cmplx":       ("occlusion",    "occlusion_complexity"),
    "person_vis":       ("occlusion",    "person_visibility"),
    "bge":              ("background",   "bg_entropy_mean"),
    "bse":              ("background",   "bg_semantic_entropy_global"),
    "bod":              ("background",   "bg_object_density_mean"),
    "igm":              ("illumination", "illumination_gradient_mean"),
    "lvm":              ("illumination", "luminance_var_global"),
    "shape":            ("body_shape",   "shape_diversity_logdet"),
    "gmt_var":          ("garment",      "garment_variance_total"),
    "gmt_div":          ("garment",      "garment_diversity_logdet"),
    "camera_azimuth":   ("camera",       "azimuth_std"),
    "camera_elevation": ("camera",       "elevation_std"),
}


def evaluate(batches: Iterable[Tuple[torch.Tensor, torch.Tensor]], device: str, desc: str) -> Dict[str, float]:
    """batches yields (person, cloth) tensors (B, 3, H, W) in [0, 1]."""
    pose = PoseDiversity(device=device)
    person_metrics = {
        "occlusion": OcclusionMetrics(device=device),
        "background": BackgroundMetrics(device=device),
        "illumination": IlluminationMetrics(),
        "body_shape": BodyShapeMetrics(device=device),
        "camera": CameraAngleMetrics(device=device),
    }
    garment = GarmentTextureMetrics(device=device)

    n = 0
    for person, cloth in tqdm(batches, desc=desc, unit="batch"):
        pose.update(person)
        for m in person_metrics.values():
            m.update(person)
        garment.update(cloth)
        n += person.shape[0]

    outputs = {name: m.compute() for name, m in person_metrics.items()}
    outputs["garment"] = garment.compute()

    row = {"n_images": n, "pose": pose.compute()["pose_variance_deg2"]}
    for column, (group, key) in PAPER_COLUMNS.items():
        row[column] = float(outputs[group].get(key, float("nan")))
    return row


# ── Data sources ──────────────────────────────────────────────────────────────

def curvton_batches(root: str, difficulty: str, sample_ratio: float, batch_size: int,
                    num_workers: int, seed: int):
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from dataloaders.curvton_dataloader import CURVTONDataloader

    samples = CURVTONDataloader(base_path=root, difficulty=difficulty, sample_ratio=sample_ratio,
                                seed=seed, return_paths=True).samples
    tf = T.Compose([T.Resize((512, 384)), T.ToTensor()])

    class _Pairs(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            s = samples[i]
            return (tf(Image.open(s["person_path"]).convert("RGB")),
                    tf(Image.open(s["cloth_path"]).convert("RGB")))

    yield from DataLoader(_Pairs(), batch_size=batch_size, num_workers=num_workers)


def dataset_batches(name: str, root: str, split: str, batch_size: int, num_workers: int):
    from pretrained_metrics.dataloader import get_dataloader

    for batch in get_dataloader(name, root, split=split, batch_size=batch_size, num_workers=num_workers):
        yield batch["person"].float(), batch["cloth"].float()


def main() -> int:
    p = argparse.ArgumentParser(description="Dataset-diversity metrics reported in the CURVTON-205K paper.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--curvton_root", help="CURVTON root with easy/ medium/ hard/")
    src.add_argument("--dataset", help="Dataset name from datasets/loaders.py (e.g. vitonhd, dresscode)")
    p.add_argument("--root", help="Dataset root (with --dataset)")
    p.add_argument("--split", default="test")
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard", "all"],
                   help="CURVTON tiers to report (with --curvton_root)")
    p.add_argument("--sample_ratio", type=float, default=1.0, help="Share of CURVTON samples to use")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results/paper_metrics.json", help="JSON file for the results")
    args = p.parse_args()

    results: Dict[str, Dict] = {}
    if args.curvton_root:
        for diff in args.difficulties:
            batches = curvton_batches(args.curvton_root, diff, args.sample_ratio, args.batch_size,
                                      args.num_workers, args.seed)
            results[f"curvton_{diff}"] = evaluate(batches, args.device, f"curvton/{diff}")
    else:
        if not args.root:
            p.error("--root is required with --dataset")
        batches = dataset_batches(args.dataset, args.root, args.split, args.batch_size, args.num_workers)
        results[args.dataset] = evaluate(batches, args.device, args.dataset)

    columns = ["pose"] + list(PAPER_COLUMNS)
    print("\n" + "dataset".ljust(20) + "".join(c[:10].rjust(11) for c in columns))
    for name, row in results.items():
        print(name.ljust(20) + "".join(f"{row[c]:11.3f}" for c in columns))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
