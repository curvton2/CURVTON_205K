"""
setup_and_run.py
================
Loads the model ONCE, then runs inference directly across images by sampling
prompts from camera_control.json. No CSVs are generated or used.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from batch_infer import (
    load_pipeline, CameraEditDataset, camera_edit_collate, run_inference, LatencyTracker
)
from sample_prompts import DEFAULT_JSON, load_library, sample as sample_prompts


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def collect_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

def generate_samples(studio: Path, diff: str, n: int, seed: int, lib: dict) -> List[dict]:
    img_dir = studio / diff
    if not img_dir.exists():
        print(f"  [WARN] {img_dir} does not exist — skipping {diff}.")
        return []

    images = collect_images(img_dir)
    if not images:
        print(f"  [WARN] No images found in {img_dir} — skipping {diff}.")
        return []

    print(f"  {diff.upper()}  ({len(images)} images)")

    use_replace = n > 100
    prompts = sample_prompts(
        lib,
        difficulty=diff,
        n=n,
        replace=use_replace,
        seed=seed,
    )

    all_rows = []
    # For each prompt, applying to all images
    for idx, prompt in enumerate(prompts):
        for img_idx, img in enumerate(images):
            all_rows.append({
                "image_path": str(img),
                "transform": "",
                "prompt": prompt,
                "seed": seed + idx * 10_000 + img_idx,
                "run_idx": idx 
            })
    return all_rows


def run_all(
    pipe,
    samples_by_diff: Dict[str, List[dict]],
    results_dir: Path,
    steps: int,
    num_workers: int,
    force_size: Tuple[int, int],
    compile_flag: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for diff in ["easy", "medium", "hard"]:
        samples = samples_by_diff.get(diff, [])
        if not samples:
            continue

        print(f"\n{'─'*62}")
        print(f"  {diff.upper()}  —  {len(samples)} total operations")
        print(f"{'─'*62}")

        # Group by run_idx (iteration per prompt)
        run_indices = sorted(list(set(s["run_idx"] for s in samples)))
        total_runs = len(run_indices)

        for run_idx in run_indices:
            run_samples = [s for s in samples if s["run_idx"] == run_idx]
            run_name = f"{diff}_{run_idx:03d}"
            out_dir = results_dir / diff / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  [{run_idx+1}/{total_runs}]  {run_name} ({len(run_samples)} images)")
            print(f"  Output → {out_dir}")

            dataset = CameraEditDataset(raw_samples=run_samples, force_size=force_size)
            
            loader_kwargs = dict(
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=camera_edit_collate,
                pin_memory=(device == "cuda"),
                drop_last=False,
            )
            if num_workers > 0:
                loader_kwargs["prefetch_factor"]    = 2
                loader_kwargs["persistent_workers"] = True

            dataloader = DataLoader(dataset, **loader_kwargs)

            tracker: LatencyTracker = run_inference(
                pipe=pipe,
                dataloader=dataloader,
                output_dir=out_dir,
                num_inference_steps=steps,
                true_guidance_scale=1.0,
                device=device,
                compiled=compile_flag,
                force_size=force_size,
            )

            report = tracker.summary()
            report_path = out_dir / "latency.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            n_out = len(list(out_dir.glob("*.png")))
            print(f"  ✓  {n_out} images  |  avg {report['avg_gpu_per_sample_s']:.1f}s/img")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--studio", default=".", help="Studio root containing easy/ medium/ hard/ images and camera_control.json.")
    p.add_argument("--n", type=int, default=10, help="Number of prompts to sample per difficulty.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force_size", type=int, nargs=2, default=[1024, 1024], metavar=("W", "H"))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fa2", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    studio = Path(args.studio).resolve()
    res_dir = studio / "results"

    json_path = studio / "camera_control.json"
    if not json_path.exists():
        json_path = DEFAULT_JSON  # CURVTON_assets/dictionaries/camera_control.json
    if not json_path.exists():
        print(f"[ERROR] camera_control.json not found in {studio} or {DEFAULT_JSON}.")
        sys.exit(1)
        
    lib = load_library(json_path)

    print("=" * 62)
    print("  CONFIGURATION")
    print("=" * 62)
    print(f"  Studio root    : {studio}")
    print(f"  Prompt library : {json_path}")
    print(f"  Results        : {res_dir}")
    print(f"  Prompts/level  : {args.n}")
    print(f"  Steps          : {args.steps}")
    print(f"  Force size     : {args.force_size[0]}×{args.force_size[1]}")
    print("=" * 62)

    samples_by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        samples_by_diff[diff] = generate_samples(studio, diff, args.n, args.seed, lib)
        
    total_imgs = sum(len(v) for v in samples_by_diff.values())
    if total_imgs == 0:
        print("\n[ERROR] No images found to process. Please ensure you have easy/, medium/, or hard/ directories with images.")
        sys.exit(1)

    print("\n── STEP 1: Loading model (No CSVs involved) ─────────────────")
    # For newer batch_infer without 'use_fa2' kwargs
    pipe = load_pipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        compile_transformer=args.compile,
    )

    print(f"\n── STEP 2: Running inference ({total_imgs} total augmented images) ─")
    run_all(
        pipe=pipe,
        samples_by_diff=samples_by_diff,
        results_dir=res_dir,
        steps=args.steps,
        num_workers=args.num_workers,
        force_size=tuple(args.force_size) if args.force_size else None,
        compile_flag=args.compile,
    )
    
    print("\n" + "=" * 62)
    print("  ALL DONE")
    print("=" * 62)

if __name__ == "__main__":
    main()