#!/usr/bin/env python3
"""
pose_diversity.py — Pose diversity sigma^2_pose, exactly as defined in the paper
(Supplementary, "Diversity Metric Definition", Pose).

    1. MMPose ViTPose extracts 17 COCO keypoints K_i for every image I_i.
    2. For each of J = 8 limb triplets (a, b, c) — left/right elbow, knee,
       shoulder and torso-hip — the bend angle at the vertex joint b is

           theta_ij = arccos( (K_a - K_b) . (K_c - K_b) / (|K_a - K_b| |K_c - K_b|) )   [degrees]

    3. sigma^2_pose = (1 / J) * sum_j Var_i(theta_ij)                                    [deg^2]

    A joint is skipped for an image when one of its three keypoints is not detected;
    the variance of each joint is taken over the images where it was measured.

Usage
-----
# CURVTON test set, per difficulty tier
python pose_diversity.py --curvton_root /path/to/dataset_ultimate_test

# Any dataset supported by datasets/loaders.py (VITON-HD, DressCode, StreetTryOn, ...)
python pose_diversity.py --dataset vitonhd --root /path/to/vitonhd --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pretrained_metrics.metrics.m1_pose import LIMB_TRIPLETS, TRIPLET_IDX, _KeypointExtractor  # noqa: E402


class PoseDiversity:
    """Accumulates per-image joint bend angles; compute() returns sigma^2_pose in deg^2."""

    def __init__(self, device: str = "cpu"):
        self.extractor = _KeypointExtractor(device=device)
        self._angles: List[np.ndarray] = []   # (J,) per image, NaN where a joint is missing

    @torch.no_grad()
    def update(self, person_imgs: torch.Tensor) -> None:
        """person_imgs : (B, 3, H, W) float32 in [0, 1]."""
        kps = self.extractor(person_imgs)                    # (B, 17, 2)
        valid = np.isfinite(kps).all(axis=2) & (np.linalg.norm(kps, axis=2) > 1e-6)
        for k, v in zip(kps, valid):
            self._angles.append(np.array([self._bend_angle(k, v, a, b, c) for a, b, c in TRIPLET_IDX]))

    @staticmethod
    def _bend_angle(k: np.ndarray, valid: np.ndarray, a: int, b: int, c: int) -> float:
        if not (valid[a] and valid[b] and valid[c]):
            return np.nan
        u, w = k[a] - k[b], k[c] - k[b]
        denom = np.linalg.norm(u) * np.linalg.norm(w)
        if denom < 1e-8:
            return np.nan
        return float(np.degrees(np.arccos(np.clip(np.dot(u, w) / denom, -1.0, 1.0))))

    def compute(self) -> Dict[str, float]:
        if not self._angles:
            return {"pose_variance_deg2": float("nan"), "n_images": 0}
        A = np.stack(self._angles)                            # (N, J)
        per_joint = [float(np.var(col[np.isfinite(col)], ddof=0)) if np.isfinite(col).sum() > 1 else np.nan
                     for col in A.T]
        return {
            "pose_variance_deg2": float(np.nanmean(per_joint)),
            "per_joint_variance_deg2": {
                f"{a}-{b}-{c}": v for (a, b, c), v in zip(LIMB_TRIPLETS, per_joint)
            },
            "n_images": int(A.shape[0]),
        }

    def reset(self) -> None:
        self._angles.clear()


# ── Data sources ──────────────────────────────────────────────────────────────

def curvton_batches(root: str, difficulty: str, sample_ratio: float, batch_size: int,
                    num_workers: int, seed: int) -> Iterable[torch.Tensor]:
    """Person images of one CURVTON tier ("all" = easy + medium + hard), as [0, 1] tensors."""
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from dataloaders.curvton_dataloader import CURVTONDataloader

    samples = CURVTONDataloader(base_path=root, difficulty=difficulty, sample_ratio=sample_ratio,
                                seed=seed, return_paths=True).samples
    tf = T.Compose([T.Resize((512, 384)), T.ToTensor()])

    class _Persons(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return tf(Image.open(samples[i]["person_path"]).convert("RGB"))

    yield from DataLoader(_Persons(), batch_size=batch_size, num_workers=num_workers)


def dataset_batches(name: str, root: str, split: str, batch_size: int, num_workers: int,
                    **kwargs) -> Iterable[torch.Tensor]:
    from pretrained_metrics.dataloader import get_dataloader

    for batch in get_dataloader(name, root, split=split, batch_size=batch_size,
                                num_workers=num_workers, **kwargs):
        yield batch["person"].float()


def run(batches: Iterable[torch.Tensor], device: str, desc: str) -> Dict[str, float]:
    metric = PoseDiversity(device=device)
    for person in tqdm(batches, desc=desc, unit="batch"):
        metric.update(person)
    return metric.compute()


def main() -> int:
    p = argparse.ArgumentParser(description="Pose diversity sigma^2_pose (deg^2) as defined in the paper.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--curvton_root", help="CURVTON root with easy/ medium/ hard/")
    src.add_argument("--dataset", help="Dataset name from datasets/loaders.py (e.g. vitonhd, dresscode)")
    p.add_argument("--root", help="Dataset root (with --dataset)")
    p.add_argument("--split", default="test")
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard", "all"],
                   help="CURVTON tiers to report (with --curvton_root)")
    p.add_argument("--sample_ratio", type=float, default=1.0, help="Share of CURVTON images to use")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", help="Optional JSON file for the results")
    args = p.parse_args()

    results: Dict[str, Dict] = {}
    if args.curvton_root:
        for diff in args.difficulties:
            batches = curvton_batches(args.curvton_root, diff, args.sample_ratio, args.batch_size,
                                      args.num_workers, args.seed)
            results[f"curvton_{diff}"] = run(batches, args.device, f"curvton/{diff}")
    else:
        if not args.root:
            p.error("--root is required with --dataset")
        batches = dataset_batches(args.dataset, args.root, args.split, args.batch_size, args.num_workers)
        results[args.dataset] = run(batches, args.device, args.dataset)

    for name, r in results.items():
        print(f"{name:<20} sigma^2_pose = {r['pose_variance_deg2']:.2f} deg^2   (n = {r['n_images']})")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
