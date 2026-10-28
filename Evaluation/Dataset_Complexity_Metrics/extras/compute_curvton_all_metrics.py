"""
extras/compute_curvton_all_metrics.py
===============================================
Compute pretrained metrics for CURVTON dataset

Supports:
- Individual metrics for Easy, Medium, Hard splits
- Combined metrics (weighted average)
- Multiple sample ratios (10%, 20%, ..., 100%)
- Comparison with benchmark datasets (VITON-HD, DressCode)
- Multi-GPU distributed processing (torchrun)
- Per-metric checkpoint saving with crash recovery

Usage:
    # Single GPU
    python extras/compute_curvton_all_metrics.py \
        --base_path ../dataset_v3/dataset_ultimate \
        --out_dir metrics_output/curvton \
        --sample_ratio 1.0

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 extras/compute_curvton_all_metrics.py \
        --base_path /path/to/dataset_ultimate \
        --out_dir metrics_output/curvton

    # Multiple ratios
    torchrun --nproc_per_node=4 extras/compute_curvton_all_metrics.py \
        --base_path /path/to/dataset_ultimate \
        --multi_ratio
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloaders.curvton_dataloader import CURVTONDataloader

# Import class-based metric modules
from pretrained_metrics.metrics.m1_pose import PoseMetrics
from pretrained_metrics.metrics.m2_occlusion import OcclusionMetrics
from pretrained_metrics.metrics.m3_background import BackgroundMetrics
from pretrained_metrics.metrics.m4_illumination import IlluminationMetrics
from pretrained_metrics.metrics.m5_body_shape import BodyShapeMetrics
from extras.m6_appearance import AppearanceMetrics
from pretrained_metrics.metrics.m7_garment_texture import GarmentTextureMetrics
from extras.unified_index import UnifiedComplexityIndex


# ── Distributed helpers ───────────────────────────────────────────────────────

def _setup_distributed():
    """Initialize torch.distributed if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return rank, world, device


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _print_rank0(msg, rank=0):
    if rank == 0:
        print(msg)


# ── Dataset wrapper: PIL→tensor in DataLoader workers ─────────────────────────

_TRANSFORM = T.Compose([T.Resize((512, 384)), T.ToTensor()])


class CURVTONTensorDataset(Dataset):
    """Wraps CURVTONDataloader samples.  PIL decode + resize + toTensor
    happen inside __getitem__ so DataLoader workers handle I/O in parallel."""

    def __init__(self, curvton_loader: CURVTONDataloader):
        self.samples = curvton_loader.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            person = _TRANSFORM(Image.open(s["person_path"]).convert("RGB"))
            cloth = _TRANSFORM(Image.open(s["cloth_path"]).convert("RGB"))
        except Exception:
            person = torch.zeros(3, 512, 384)
            cloth = torch.zeros(3, 512, 384)
        return person, cloth


# ── State keys for gathering across ranks ─────────────────────────────────────

_METRIC_STATE_KEYS = {
    PoseMetrics:          ["_pose_vecs", "_all_angles", "_per_image_artic"],
    OcclusionMetrics:     ["_ratios_body", "_ratios_carried", "_ratios_accessory",
                           "_ratios_env", "_ratios_people", "_ratios_other",
                           "_ratios_total"],
    BackgroundMetrics:    ["_entropies", "_obj_counts"],
    IlluminationMetrics:  ["_mean_L", "_grad_var"],
    BodyShapeMetrics:     ["_betas"],
    AppearanceMetrics:    ["_embeddings"],
    GarmentTextureMetrics: ["_embeddings"],
}


def _gather_metric_state(metric_obj, metric_cls):
    """All-gather internal Python lists from all ranks onto rank 0."""
    if not _is_distributed():
        return

    keys = _METRIC_STATE_KEYS.get(metric_cls, [])
    for key in keys:
        local_data = getattr(metric_obj, key)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_data)
        if dist.get_rank() == 0:
            if isinstance(local_data, dict):
                merged = {}
                for d in gathered:
                    for k, v in d.items():
                        merged.setdefault(k, []).extend(v)
                setattr(metric_obj, key, merged)
            else:
                merged = []
                for lst in gathered:
                    merged.extend(lst)
                setattr(metric_obj, key, merged)


# ── Core metric runner ────────────────────────────────────────────────────────

def _run_single_metric(label, metric_cls, metric_kwargs, dataloader,
                       use_cloth, device, rank, verbose):
    """Instantiate one metric, iterate DataLoader, gather across ranks,
    compute on rank 0, then free GPU memory."""
    if verbose:
        _print_rank0(f"\n    [{label}] Loading model ...", rank)

    metric_obj = metric_cls(**metric_kwargs)

    for person_batch, cloth_batch in dataloader:
        batch = cloth_batch if use_cloth else person_batch
        metric_obj.update(batch)

    # Gather state from all GPUs before computing
    _gather_metric_state(metric_obj, metric_cls)

    n_samples = 0
    sub = {}
    if rank == 0:
        keys = _METRIC_STATE_KEYS.get(metric_cls, [])
        if keys:
            first_attr = getattr(metric_obj, keys[0])
            if isinstance(first_attr, dict):
                n_samples = max((len(v) for v in first_attr.values()), default=0)
            else:
                n_samples = len(first_attr)

        sub = metric_obj.compute()
        if verbose:
            print(f"    [{label}] Done ({n_samples} samples). Keys: {list(sub.keys())}")

    del metric_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sub, n_samples


_ALL_METRIC_SPECS = [
    ("pose",            PoseMetrics,          True,  False),
    ("occlusion",       OcclusionMetrics,     True,  False),
    ("background",      BackgroundMetrics,    True,  False),
    ("illumination",    IlluminationMetrics,  False, False),
    ("body_shape",      BodyShapeMetrics,     True,  False),
    ("appearance",      AppearanceMetrics,    True,  False),
    ("garment_texture", GarmentTextureMetrics, True, True),
]
# label, class, needs_device, use_cloth


def compute_metrics_for_split(
    curvton_loader: CURVTONDataloader,
    split_name: str,
    device: str = "cuda",
    batch_size: int = 40,
    num_workers: int = 16,
    rank: int = 0,
    world_size: int = 1,
    out_dir: str = "metrics_output/curvton",
    verbose: bool = True,
    only_metrics: List[str] | None = None,
    existing_results: Dict[str, float] | None = None,
    appearance_kwargs: Dict[str, object] | None = None,
) -> Dict[str, float]:
    """Compute metrics for one split, saving each metric incrementally.

    Parameters
    ----------
    only_metrics : list of str, optional
        If given, only run these metric labels (e.g. ["appearance"]).
        Otherwise run all 7 metrics.
    existing_results : dict, optional
        Previously computed results for this split to merge into.
    """

    _print_rank0(
        f"\n  Computing metrics for {split_name} "
        f"({len(curvton_loader)} samples)...",
        rank,
    )

    dataset = CURVTONTensorDataset(curvton_loader)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                           shuffle=False)
        if _is_distributed() else None
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    # Build metric_specs from global list, optionally filtered
    metric_specs = []
    for label, cls, needs_device, use_cloth in _ALL_METRIC_SPECS:
        if only_metrics is not None and label not in only_metrics:
            continue
        kwargs = {"device": device} if needs_device else {}
        if label == "appearance" and appearance_kwargs:
            kwargs.update(appearance_kwargs)
        metric_specs.append((label, cls, kwargs, use_cloth))

    results: Dict[str, float] = dict(existing_results or {})
    n_processed = results.get("n_samples", 0)
    out_path = Path(out_dir)

    for label, cls, kwargs, use_cloth in metric_specs:
        # Check for existing checkpoint (crash recovery)
        ckpt_file = out_path / f"curvton_metrics_{dataset_name}_{split_name}_{label}_ckpt.json"
        if rank == 0 and ckpt_file.exists():
            try:
                with open(ckpt_file, "r") as f:
                    cached = json.load(f)
                results.update(cached["metrics"])
                n_processed = max(n_processed, cached.get("n_samples", 0))
                if verbose:
                    print(f"    [{label}] Recovered from checkpoint.")
                continue
            except Exception:
                pass

        try:
            sub, n = _run_single_metric(
                label, cls, kwargs, loader,
                use_cloth=use_cloth,
                device=device,
                rank=rank,
                verbose=verbose,
            )
            if rank == 0:
                results.update(sub)
                n_processed = max(n_processed, n)
                # Save per-metric checkpoint
                out_path.mkdir(parents=True, exist_ok=True)
                with open(ckpt_file, "w") as f:
                    json.dump({"metrics": sub, "n_samples": n},
                              f, indent=2, default=str)
        except Exception as e:
            if verbose:
                _print_rank0(f"    Warning: {label} metric failed: {e}", rank)

    results["n_samples"] = n_processed

    # Save per-split results
    if rank == 0:
        split_file = out_path / f"curvton_metrics_{dataset_name}_{split_name}_split.json"
        with open(split_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"    Saved split checkpoint: {split_file}")

    return results


def compute_curvton_metrics(
    base_path: str,
    out_dir: str = "metrics_output/curvton",
    sample_ratio: float = 1.0,
    difficulties: List[str] = ["easy", "medium", "hard"],
    batch_size: int = 40,
    num_workers: int = 16,
    seed: int = 42,
    m6_face_detector_backend: str = "auto",
    m6_retinaface_repo_dir: str | None = None,
    m6_retinaface_weights: str | None = None,
    m6_retinaface_backbone: str = "mobilenetv1_0.25",
    m6_retinaface_device: str = "cpu",
) -> Dict[str, Dict]:
    """Compute pretrained metrics for CURVTON dataset with multi-GPU support."""

    rank, world_size, device = _setup_distributed()
    out_path = Path(out_dir)
    if rank == 0:
        out_path.mkdir(parents=True, exist_ok=True)

    _print_rank0("=" * 70, rank)
    _print_rank0("CURVTON Pretrained Metrics Computation", rank)
    _print_rank0("=" * 70, rank)
    _print_rank0(f"  Base path:    {base_path}", rank)
    _print_rank0(f"  Output:       {out_dir}", rank)
    _print_rank0(f"  Sample ratio: {sample_ratio:.0%}", rank)
    _print_rank0(f"  Difficulties: {difficulties}", rank)
    _print_rank0(f"  Device:       {device}  (world_size={world_size})", rank)
    _print_rank0(f"  Batch/GPU:    {batch_size}  Workers/GPU: {num_workers}", rank)
    _print_rank0("=" * 70, rank)

    all_results = {}
    sample_counts = {}
    _loaders: Dict[str, CURVTONDataloader] = {}
    appearance_kwargs = {
        "face_detector_backend": m6_face_detector_backend,
        "retinaface_repo_dir": m6_retinaface_repo_dir,
        "retinaface_weights": m6_retinaface_weights,
        "retinaface_backbone": m6_retinaface_backbone,
        "retinaface_device": m6_retinaface_device,
    }

    # Build loaders once for reuse in both phases
    for diff in difficulties:
        loader = CURVTONDataloader(
            base_path=base_path,
            difficulty=diff,
            sample_ratio=sample_ratio,
            seed=seed,
            return_paths=True,
        )
        _loaders[diff] = loader
        sample_counts[diff] = len(loader)

    # ── Phase 1: fast GPU metrics (M1-M5, M7) for ALL splits ─────────────
    _print_rank0("\n" + "━" * 70, rank)
    _print_rank0("Phase 1: Computing pose, occlusion, background, "
                 "illumination, body_shape, garment_texture", rank)
    _print_rank0("━" * 70, rank)

    fast_metrics = ["pose", "occlusion", "background", "illumination",
                    "body_shape", "garment_texture"]

    for diff in difficulties:
        _print_rank0(f"\n[{diff.upper()}] Phase 1", rank)
        metrics = compute_metrics_for_split(
            _loaders[diff], diff,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            out_dir=out_dir,
            only_metrics=fast_metrics,
        )
        all_results[diff] = metrics

    # ── Phase 2: appearance / ArcFace (M6) for ALL splits ─────────────────
    _print_rank0("\n" + "━" * 70, rank)
    _print_rank0("Phase 2: Computing appearance (ArcFace) — deferred metric",
                 rank)
    _print_rank0("━" * 70, rank)

    for diff in difficulties:
        _print_rank0(f"\n[{diff.upper()}] Phase 2 (appearance)", rank)
        metrics = compute_metrics_for_split(
            _loaders[diff], diff,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            out_dir=out_dir,
            only_metrics=["appearance"],
            existing_results=all_results[diff],
            appearance_kwargs=appearance_kwargs,
        )
        all_results[diff] = metrics

    # ── Phase 3: Unified Complexity Index (rank 0 only) ─────────────────
    #
    #   Step 1 → Collect all raw metrics from ALL splits (easy/medium/hard)
    #   Step 2 → Compute per-metric adaptive temperature τ_k from the
    #            z-score ranges observed across splits, so that no metric
    #            saturates the sigmoid and loses discriminative information
    #   Step 3 → Apply sigmoid(z_k / τ_k) and combine into final score
    #
    if rank == 0:
        _print_rank0("\n" + "━" * 70, rank)
        _print_rank0("Phase 3: Unified Complexity Index", rank)
        _print_rank0("━" * 70, rank)

        # Step 1: Feed all split results into the unified index
        _print_rank0("\n  Step 1: Collecting raw metrics from all splits...", rank)
        uci = UnifiedComplexityIndex()
        for diff in difficulties:
            uci.add_dataset(f"CURVTON-{diff}", all_results[diff])
            _print_rank0(f"    Added CURVTON-{diff.upper()} "
                         f"({all_results[diff].get('n_samples', '?')} samples)",
                         rank)

        # Step 2 + 3: compute_scores() internally does:
        #   Pass 1 → z-scores for all (split, metric), find max|z| per metric
        #   Pass 2 → set τ_k = max|z_k| / target, apply sigmoid(z/τ_k)
        _print_rank0("\n  Step 2: Computing adaptive temperatures from "
                     "cross-split z-score ranges...", rank)
        uci_scores = uci.compute_scores()

        # Print the computed temperatures
        if uci_scores:
            temps = uci_scores[0].get("temperatures", {})
            if temps:
                _print_rank0(f"\n  {'Metric':<35} {'τ (temperature)':>16}", rank)
                _print_rank0(f"  {'─'*35} {'─'*16}", rank)
                from extras.unified_index import METRIC_KEYS as _MK
                for mk, label in _MK:
                    t = temps.get(mk, 0.0)
                    _print_rank0(f"  {label:<35} {t:>16.4f}", rank)
                _print_rank0("", rank)

        _print_rank0("  Step 3: Sigmoid scoring + weighted combination done.\n",
                     rank)

        # Store unified scores back into per-split results
        for entry in uci_scores:
            diff_key = entry["dataset"].split("-")[-1].lower()
            if diff_key in all_results:
                all_results[diff_key]["unified_score"] = entry["unified_score"]

        output = {
            "dataset": "CURVTON",
            "base_path": base_path,
            "sample_ratio": sample_ratio,
            "timestamp": datetime.now().isoformat(),
            "sample_counts": sample_counts,
            "total_samples": sum(sample_counts.values()),
            "world_size": world_size,
            "batch_size_per_gpu": batch_size,
            "metrics": all_results,
            "unified_index_report": uci_scores,
            "adaptive_temperatures": uci_scores[0].get("temperatures", {})
                                     if uci_scores else {},
        }

        ratio_pct = int(sample_ratio * 100)
        output_file = out_path / f"curvton_metrics_{dataset_name}_{split_name}_{ratio_pct}pct.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved metrics to {output_file}")

        # ── Comprehensive JSON (raw + normalized + normalization params) ───
        comprehensive = uci.get_comprehensive_results(uci_scores)
        comprehensive["timestamp"] = datetime.now().isoformat()
        comprehensive["dataset_info"] = {
            "name": "CURVTON",
            "base_path": base_path,
            "sample_ratio": sample_ratio,
            "sample_counts": sample_counts,
            "total_samples": sum(sample_counts.values()),
        }
        comp_file = out_path / f"curvton_metrics_comprehensive_{dataset_name}_{split_name}_{ratio_pct}pct.json"
        with open(comp_file, "w") as f:
            json.dump(comprehensive, f, indent=2, default=str)
        print(f"  Comprehensive results (raw + normalized) saved → {comp_file}")

        uci.print_report(uci_scores)

        # Clean up checkpoint files
        for ckpt in out_path.glob("_ckpt_*.json"):
            ckpt.unlink()
        for split_ckpt in out_path.glob("_split_*.json"):
            split_ckpt.unlink()

    if _is_distributed():
        dist.barrier()

    return all_results


def compute_multi_ratio_metrics(
    base_path: str,
    out_dir: str = "metrics_output/curvton",
    ratios: List[float] = None,
    batch_size: int = 40,
    num_workers: int = 16,
    m6_face_detector_backend: str = "auto",
    m6_retinaface_repo_dir: str | None = None,
    m6_retinaface_weights: str | None = None,
    m6_retinaface_backbone: str = "mobilenetv1_0.25",
    m6_retinaface_device: str = "cpu",
) -> Dict[str, Dict]:
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 1.0]
    """Compute metrics for multiple sample ratios."""
    rank = int(os.environ.get("RANK", 0))
    all_ratio_results = {}

    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        _print_rank0(f"\n{'#' * 70}", rank)
        _print_rank0(f"# Computing metrics for {ratio_pct}% of CURVTON dataset",
                     rank)
        _print_rank0(f"{'#' * 70}", rank)

        results = compute_curvton_metrics(
            base_path=base_path,
            out_dir=out_dir,
            sample_ratio=ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            m6_face_detector_backend=m6_face_detector_backend,
            m6_retinaface_repo_dir=m6_retinaface_repo_dir,
            m6_retinaface_weights=m6_retinaface_weights,
            m6_retinaface_backbone=m6_retinaface_backbone,
            m6_retinaface_device=m6_retinaface_device,
        )
        all_ratio_results[f"{ratio_pct}%"] = results

    if rank == 0:
        summary_file = Path(out_dir) / f"curvton_metrics_{dataset_name}_all_ratios.json"
        with open(summary_file, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "ratios": [f"{int(r*100)}%" for r in ratios],
                "results": all_ratio_results,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)
        print(f"\n  Saved multi-ratio summary to {summary_file}")

    return all_ratio_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CURVTON pretrained metrics")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Path to CURVTON dataset")
    parser.add_argument("--out_dir", type=str, default="metrics_output/curvton",
                        help="Output directory for metrics")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Fraction of dataset to use (0.1 to 1.0)")
    parser.add_argument("--multi_ratio", action="store_true",
                        help="Compute metrics for multiple sample ratios")
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 1.0],
                        help="Specific ratios to process if --multi_ratio is set")
    parser.add_argument("--difficulties", type=str, nargs="+",
                        default=["easy", "medium", "hard"],
                        help="Difficulty levels to process")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="DataLoader workers per GPU")
    parser.add_argument("--m6_face_detector_backend", type=str, default="auto",
                        choices=["auto", "retinaface_pytorch", "insightface", "haar"],
                        help="Face detector backend for M6 appearance metric.")
    parser.add_argument("--m6_retinaface_repo_dir", type=str, default=None,
                        help="Path to cloned yakhyo/retinaface-pytorch repo.")
    parser.add_argument("--m6_retinaface_weights", type=str, default=None,
                        help="Path to RetinaFace .pth weights file.")
    parser.add_argument("--m6_retinaface_backbone", type=str, default="mobilenetv1_0.25",
                        help="RetinaFace backbone/network name.")
    parser.add_argument("--m6_retinaface_device", type=str, default="cpu",
                        help="Device for RetinaFace detector (cpu or cuda).")

    args = parser.parse_args()

    if args.multi_ratio:
        compute_multi_ratio_metrics(
            base_path=args.base_path,
            out_dir=args.out_dir,
            ratios=args.ratios,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            m6_face_detector_backend=args.m6_face_detector_backend,
            m6_retinaface_repo_dir=args.m6_retinaface_repo_dir,
            m6_retinaface_weights=args.m6_retinaface_weights,
            m6_retinaface_backbone=args.m6_retinaface_backbone,
            m6_retinaface_device=args.m6_retinaface_device,
        )
    else:
        compute_curvton_metrics(
            base_path=args.base_path,
            out_dir=args.out_dir,
            sample_ratio=args.sample_ratio,
            difficulties=args.difficulties,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            m6_face_detector_backend=args.m6_face_detector_backend,
            m6_retinaface_repo_dir=args.m6_retinaface_repo_dir,
            m6_retinaface_weights=args.m6_retinaface_weights,
            m6_retinaface_backbone=args.m6_retinaface_backbone,
            m6_retinaface_device=args.m6_retinaface_device,
        )
