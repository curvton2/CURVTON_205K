"""
curvton_cache_autogen.py
========================
Builds the per-sample NPZ feature caches that the scripts in plot_scripts/
read, so the plots can be produced straight from a CURVTON dataset root.

Cache layout (pct = round(sample_ratio * 100)):
    <cache_dir>/curvton_{difficulty}_{pct}pct.npz
        pose_vecs          (N, 34)   normalised COCO-17 keypoints   (M1)
        angles             (N, T)    joint bend angles, radians     (M1)
        bg_entropy         (N,)      background texture entropy     (M3)
        bg_obj_count       (N,)      background object count        (M3)
        lum_maps           (N, H, W) downsampled luminance maps     (M4)
        betas              (N, 10)   SMPL shape parameters          (M5)
        azimuths           (N,)      camera azimuth, degrees        (M9)
        elevations         (N,)      camera elevation, degrees      (M9)
        camera_confidence  (N,)      camera estimate confidence     (M9)
    <occ_maps_dir>/curvton_{difficulty}_{pct}pct_occ_maps.npz
        occ_maps           (N, H, W) occluder masks                 (M2)

Only the feature groups named in `required_keys` are computed; keys already
present in an existing cache file are kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

DIFFICULTIES = ("easy", "medium", "hard")
MAP_SIZE = (64, 48)  # (H, W) for lum_maps / occ_maps; keeps the 512x384 aspect

# feature group -> keys it produces
_GROUPS: Dict[str, Sequence[str]] = {
    "pose": ("pose_vecs", "angles"),
    "occlusion": ("occ_maps",),
    "background": ("bg_entropy", "bg_obj_count"),
    "illumination": ("lum_maps",),
    "body_shape": ("betas",),
    "camera": ("azimuths", "elevations", "camera_confidence"),
}
_KEY_TO_GROUP = {k: g for g, keys in _GROUPS.items() for k in keys}


# ── Paths ─────────────────────────────────────────────────────────────────────

def _pct(sample_ratio: float) -> int:
    return int(round(sample_ratio * 100))


def cache_path(cache_dir: Path, difficulty: str, sample_ratio: float) -> Path:
    return Path(cache_dir) / f"curvton_{difficulty}_{_pct(sample_ratio)}pct.npz"


def occ_maps_path(occ_maps_dir: Path, difficulty: str, sample_ratio: float) -> Path:
    return Path(occ_maps_dir) / f"curvton_{difficulty}_{_pct(sample_ratio)}pct_occ_maps.npz"


def _npz_keys(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        with np.load(path, allow_pickle=True) as data:
            return set(data.files)
    except Exception:
        return set()


# ── Public API used by plot_scripts/ ──────────────────────────────────────────

def add_autogen_args(parser: argparse.ArgumentParser) -> None:
    """Add the CLI options needed to (re)generate missing caches."""
    g = parser.add_argument_group("cache auto-generation")
    g.add_argument("--curvton-root", type=Path, default=None,
                   help="CURVTON dataset root (dataset_ultimate). If set, missing "
                        "caches are computed from it.")
    g.add_argument("--occ-maps-dir", type=Path, default=None,
                   help="Directory for *_occ_maps.npz occlusion caches.")
    g.add_argument("--autogen-device", type=str, default=None,
                   help="Device for feature extraction (default: cuda if available).")
    g.add_argument("--autogen-batch-size", type=int, default=16)
    g.add_argument("--autogen-num-workers", type=int, default=4)
    g.add_argument("--autogen-seed", type=int, default=42,
                   help="Seed for selecting the sampled subset of CURVTON.")


def caches_requiring_generation(
    cache_dir: Path,
    sample_ratio: float,
    difficulties: Sequence[str],
    required_keys: Sequence[str],
) -> List[str]:
    """Return the difficulties whose cache is missing or lacks a required key."""
    missing = []
    for diff in difficulties:
        keys = _npz_keys(cache_path(cache_dir, diff, sample_ratio))
        if not set(required_keys) <= keys:
            missing.append(diff)
    return missing


def ensure_curvton_caches(
    difficulties: Sequence[str],
    args: argparse.Namespace,
    sample_ratio: float,
    required_keys: Sequence[str],
) -> None:
    """Compute the requested feature groups for each difficulty and write the caches."""
    if not difficulties:
        return
    curvton_root = getattr(args, "curvton_root", None)
    if curvton_root is None:
        raise FileNotFoundError(
            f"Caches missing for {list(difficulties)} (keys: {list(required_keys)}). "
            "Pass --curvton-root <dataset_ultimate> to generate them."
        )

    unknown = [k for k in required_keys if k not in _KEY_TO_GROUP]
    if unknown:
        raise KeyError(f"No feature extractor for cache keys: {unknown}")
    groups = sorted({_KEY_TO_GROUP[k] for k in required_keys})

    import torch
    device = args.autogen_device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    occ_dir = Path(args.occ_maps_dir) if getattr(args, "occ_maps_dir", None) else cache_dir

    for diff in difficulties:
        print(f"[cache-autogen] {diff}: computing {groups} on {device} "
              f"({_pct(sample_ratio)}% of {curvton_root})", flush=True)
        features = _extract_features(
            curvton_root, diff, sample_ratio, groups, device,
            batch_size=args.autogen_batch_size,
            num_workers=args.autogen_num_workers,
            seed=args.autogen_seed,
        )
        occ = {k: v for k, v in features.items() if k == "occ_maps"}
        rest = {k: v for k, v in features.items() if k != "occ_maps"}
        if rest:
            _merge_save(cache_path(cache_dir, diff, sample_ratio), rest)
        if occ:
            _merge_save(occ_maps_path(occ_dir, diff, sample_ratio), occ)


# ── Internals ─────────────────────────────────────────────────────────────────

def _merge_save(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: Dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            merged = {k: data[k] for k in data.files}
    merged.update(arrays)
    np.savez_compressed(path, **merged)
    print(f"[cache-autogen] wrote {path} ({', '.join(sorted(arrays))})", flush=True)


def _build_loader(curvton_root, difficulty, sample_ratio, batch_size, num_workers, seed):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from dataloaders.curvton_dataloader import CURVTONDataloader

    transform = T.Compose([T.Resize((512, 384)), T.ToTensor()])
    samples = CURVTONDataloader(
        base_path=str(curvton_root), difficulty=difficulty,
        sample_ratio=sample_ratio, seed=seed, return_paths=True,
    ).samples

    class _PersonImages(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            try:
                return transform(Image.open(samples[idx]["person_path"]).convert("RGB"))
            except Exception:
                return torch.zeros(3, 512, 384)

    return DataLoader(_PersonImages(), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)


def _downsample(maps, size=MAP_SIZE) -> np.ndarray:
    """(B, H, W) tensor/array -> (B, h, w) float32 via area averaging."""
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(np.asarray(maps) if not torch.is_tensor(maps) else maps).float()
    return F.interpolate(t.unsqueeze(1).cpu(), size=size, mode="area").squeeze(1).numpy()


def _extract_features(curvton_root, difficulty, sample_ratio, groups, device,
                      batch_size, num_workers, seed) -> Dict[str, np.ndarray]:
    import torch
    from tqdm import tqdm

    metrics = {}
    if "pose" in groups:
        from pretrained_metrics.metrics.m1_pose import PoseMetrics
        metrics["pose"] = PoseMetrics(device=device)
    if "occlusion" in groups:
        from pretrained_metrics.metrics.m2_occlusion import OcclusionMetrics
        metrics["occlusion"] = OcclusionMetrics(device=device)
    if "background" in groups:
        from pretrained_metrics.metrics.m3_background import BackgroundMetrics
        metrics["background"] = BackgroundMetrics(device=device)
    if "illumination" in groups:
        from pretrained_metrics.metrics.m4_illumination import _rgb_to_luminance
    if "body_shape" in groups:
        from pretrained_metrics.metrics.m5_body_shape import BodyShapeMetrics
        metrics["body_shape"] = BodyShapeMetrics(device=device)
    if "camera" in groups:
        from pretrained_metrics.metrics.m9_camera_angle import CameraAngleMetrics
        metrics["camera"] = CameraAngleMetrics(device=device)

    occ_maps: List[np.ndarray] = []
    lum_maps: List[np.ndarray] = []
    loader = _build_loader(curvton_root, difficulty, sample_ratio, batch_size, num_workers, seed)

    with torch.no_grad():
        for imgs in tqdm(loader, desc=f"[cache-autogen] {difficulty}"):
            for name in ("pose", "background", "body_shape", "camera"):
                if name in metrics:
                    metrics[name].update(imgs)
            if "occlusion" in metrics:
                masks = metrics["occlusion"]._seg.segment(imgs)
                ref = masks["garment"]
                occluder = sum(
                    masks.get(k, torch.zeros_like(ref)).float()
                    for k in ("carried_obj", "accessories", "environment", "other_people", "other")
                ).clamp(0, 1)
                occ_maps.append(_downsample(occluder))
            if "illumination" in groups:
                _, L = _rgb_to_luminance(imgs.cpu())
                lum_maps.append(_downsample(np.stack(L)))

    out: Dict[str, np.ndarray] = {}
    if "pose" in metrics:
        m = metrics["pose"]
        # PoseMetrics only stores a pose vector for images with >=1 valid keypoint
        # (uncertainty < 1); keep the angle rows for the same images so they align.
        keep = np.asarray(m._per_image_uncertainty) < 1.0
        angles = np.stack(m._angle_vecs).astype(np.float32)[keep]
        col_mean = np.nan_to_num(np.nanmean(angles, axis=0)) if angles.size else 0.0
        angles = np.where(np.isnan(angles), col_mean, angles)
        out["pose_vecs"] = np.stack(m._pose_vecs).astype(np.float32) if m._pose_vecs \
            else np.zeros((0, 34), np.float32)
        out["angles"] = angles
    if "occlusion" in metrics:
        out["occ_maps"] = np.concatenate(occ_maps).astype(np.float32)
    if "background" in metrics:
        m = metrics["background"]
        out["bg_entropy"] = np.asarray(m._entropies, dtype=np.float32)
        out["bg_obj_count"] = np.asarray(m._obj_counts, dtype=np.float32)
    if "illumination" in groups:
        out["lum_maps"] = np.concatenate(lum_maps).astype(np.float32)
    if "body_shape" in metrics:
        out["betas"] = np.stack(metrics["body_shape"]._betas).astype(np.float32)
    if "camera" in metrics:
        m = metrics["camera"]
        out["azimuths"] = np.asarray(m._azimuths, dtype=np.float32)
        out["elevations"] = np.asarray(m._elevations, dtype=np.float32)
        out["camera_confidence"] = np.asarray(m._confidences, dtype=np.float32)
    return out
