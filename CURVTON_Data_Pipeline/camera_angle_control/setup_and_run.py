"""
setup_and_run.py
================
Loads the model ONCE, then runs inference across all CSVs in the same
process.  No subprocess calls, no redundant model reloads.

Timeline:
  ┌─────────────────────────────────────────────────────────┐
  │  OLD (subprocess per CSV)                               │
  │  load model → infer easy_000 → exit                     │
  │  load model → infer easy_001 → exit                     │
  │  ...  ×30  (30 model loads, ~3 min wasted each time)    │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  NEW (single process)                                   │
  │  load model ONCE                                        │
  │    → infer easy_000                                     │
  │    → infer easy_001                                     │
  │    ...  ×N  (model stays in VRAM the whole time)        │
  └─────────────────────────────────────────────────────────┘

Prompts are sampled from camera_control.json via sample_prompts.py.
Put both files in the same studio root as this script.

Steps:
  1. Delete csv_output/ and regenerate it — prompts sampled from JSON library
  2. Load the pipeline once (transformer + LoRA fused + optional compile/FA2)
  3. Loop over all CSVs — for each one:
       a. Build a CameraEditDataset + DataLoader from the CSV
       b. Call run_inference() directly (no subprocess)
       c. Save output images to results/<diff>/<csv_stem>/
       d. Save per-CSV latency report to results/<diff>/<csv_stem>/latency.json

USAGE
-----
    python setup_and_run.py                     # full run, 10 CSVs per difficulty
    python setup_and_run.py --n 100             # 100 CSVs per difficulty
    python setup_and_run.py --csv_only          # regenerate CSVs only, skip inference
    python setup_and_run.py --skip_csv          # skip CSV regen, run inference only
    python setup_and_run.py --compile           # torch.compile (faster after warmup)
    python setup_and_run.py --fa2               # FlashAttention-2 on text encoder
    python setup_and_run.py --compile --fa2     # both optimisations
    python setup_and_run.py --studio PATH_TO_STUDIO_ROOT      # custom studio root

FILES REQUIRED IN STUDIO ROOT
------------------------------
    batch_infer.py        — inference engine
    sample_prompts.py     — prompt sampling utility
    camera_control.json   — prompt library (100 prompts per difficulty)
"""

import argparse
import csv
import json
import inspect
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


# =============================================================================
# Studio root — adjust if your studio path differs
# =============================================================================
DEFAULT_STUDIO = "PATH_TO_STUDIO_ROOT"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =============================================================================
# Step 1 — Regenerate csv_output/ using prompts from camera_control.json
# =============================================================================

def collect_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def regenerate_csvs(
    studio:   Path,
    csv_root: Path,
    n:        int,
    seed:     int,
) -> Dict[str, List[Path]]:
    """
    Delete csv_root/ entirely and rebuild from scratch.

    For each difficulty:
    - Images are read from  studio/<diff>/IMAGE_1.jpg ... IMAGE_N.jpg
      - Prompts are sampled from camera_control.json (via sample_prompts.py)
      - n CSV files are written to csv_root/<diff>/<diff>_000.csv ...
      - Each CSV has one row per image, all rows share the same prompt
      - Seeds are deterministic: seed + csv_idx * 10_000 + img_idx

    If n <= 100  → sample without replacement (all prompts distinct)
    If n >  100  → sample with replacement   (some prompts repeat, but
                   different seeds produce different diffusion outputs)
    """
    # ── Load prompt library ───────────────────────────────────────────────────
    if str(studio) not in sys.path:
        sys.path.insert(0, str(studio))

    try:
        from sample_prompts import DEFAULT_JSON, load_library, sample as sample_prompts  # type: ignore
    except ImportError:
        raise ImportError(
            "Cannot import sample_prompts. "
            "Make sure sample_prompts.py is in the studio root: "
            f"{studio}"
        )

    json_path = studio / "camera_control.json"
    if not json_path.exists():
        json_path = DEFAULT_JSON  # CURVTON_assets/dictionaries/camera_control.json
    if not json_path.exists():
        raise FileNotFoundError(f"camera_control.json not found at {json_path}.")

    lib = load_library(json_path)

    # ── Wipe and recreate csv_root/ ───────────────────────────────────────────
    if csv_root.exists():
        print(f"  Deleting existing {csv_root} …")
        shutil.rmtree(csv_root)
    csv_root.mkdir(parents=True)
    print(f"  Created fresh {csv_root}\n")

    all_csvs: Dict[str, List[Path]] = {}

    for diff in ["easy", "medium", "hard"]:

        # ── Collect images ────────────────────────────────────────────────────
        img_dir = studio / diff
        if not img_dir.exists():
            print(f"  [WARN] {img_dir} does not exist — skipping {diff}.")
            continue

        images = collect_images(img_dir)
        if not images:
            print(f"  [WARN] No images found in {img_dir} — skipping {diff}.")
            continue

        print(f"  {diff.upper()}  ({len(images)} images)")

        # ── Sample n prompts from JSON library ────────────────────────────────
        # replace=True when n > 100 (pool size) — different seeds still give
        # different diffusion outputs even if the prompt repeats
        use_replace = n > 100
        prompts = sample_prompts(
            lib,
            difficulty=diff,
            n=n,
            replace=use_replace,
            seed=seed,
        )

        # ── Write CSVs ────────────────────────────────────────────────────────
        out_dir = csv_root / diff
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        print(f"  Writing {n} CSVs → {out_dir}/")
        for idx, prompt in enumerate(prompts):
            csv_path = out_dir / f"{diff}_{idx:03d}.csv"
            rows = [
                {
                    "image_path": str(img),       # absolute path
                    "transform":  "",             # empty — prompt carries everything
                    "prompt":     prompt,
                    "seed":       seed + idx * 10_000 + img_idx,
                }
                for img_idx, img in enumerate(images)
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["image_path", "transform", "prompt", "seed"]
                )
                w.writeheader()
                w.writerows(rows)

            short = prompt[:65] + ("…" if len(prompt) > 65 else "")
            print(f"    {csv_path.name}  {len(rows)} rows  |  {short}")
            written.append(csv_path)

        all_csvs[diff] = written
        print()

    return all_csvs


# =============================================================================
# Step 2 — Load model ONCE
# =============================================================================

def load_model(compile_flag: bool, fa2_flag: bool, studio: Path):
    """
    Import load_pipeline from batch_infer.py and call it once.
    The pipeline object stays in VRAM for the entire run — all CSVs share it.

    Uses inspect.signature to work with any version of batch_infer.py:
      - Older versions without use_fa2  : --fa2 is silently ignored
      - Older versions without compile  : --compile is silently ignored
      - Latest version                  : both flags respected
    """
    if str(studio) not in sys.path:
        sys.path.insert(0, str(studio))

    from batch_infer import load_pipeline  # type: ignore

    sig    = inspect.signature(load_pipeline)
    params = sig.parameters

    kwargs: Dict = dict(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )

    if "compile_transformer" in params:
        kwargs["compile_transformer"] = compile_flag
    elif compile_flag:
        print("  [WARN] This batch_infer.py does not support --compile. "
              "Update to the latest version.")

    if "use_fa2" in params:
        kwargs["use_fa2"] = fa2_flag
    elif fa2_flag:
        print("  [WARN] This batch_infer.py does not support --fa2. "
              "Update to the latest version.")

    pipe = load_pipeline(**kwargs)
    return pipe


# =============================================================================
# Step 3 — Run inference across all CSVs reusing the loaded model
# =============================================================================

def run_all(
    pipe,
    all_csvs:     Dict[str, List[Path]],
    results_dir:  Path,
    studio:       Path,
    steps:        int,
    num_workers:  int,
    force_size:   Tuple[int, int],
    compile_flag: bool,
) -> None:
    """
    Loop over every CSV and call run_inference() directly in this process.
    The model never leaves VRAM — zero reload cost between CSVs.
    """
    if str(studio) not in sys.path:
        sys.path.insert(0, str(studio))

    from batch_infer import (          # type: ignore
        CameraEditDataset,
        camera_edit_collate,
        run_inference,
        LatencyTracker,
    )

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    total_csvs = sum(len(v) for v in all_csvs.values())
    done       = 0

    for diff in ["easy", "medium", "hard"]:
        csvs = all_csvs.get(diff, [])
        if not csvs:
            continue

        print(f"\n{'─'*62}")
        print(f"  {diff.upper()}  —  {len(csvs)} CSVs")
        print(f"{'─'*62}")

        for csv_path in csvs:
            done += 1
            out_dir = results_dir / diff / csv_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{done}/{total_csvs}]  {csv_path.name}")
            print(f"  Output → {out_dir}")

            # ── Dataset ───────────────────────────────────────────────────────
            try:
                dataset = CameraEditDataset(
                    csv_path=str(csv_path),
                    force_size=force_size,
                )
            except Exception as e:
                print(f"  [ERROR] Dataset load failed: {e} — skipping.")
                continue

            # ── DataLoader ────────────────────────────────────────────────────
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

            # ── Inference (model already in VRAM — no reload) ─────────────────
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

            # ── Per-CSV latency report ─────────────────────────────────────────
            report     = tracker.summary()
            report_path = out_dir / "latency.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            n_out = len(list(out_dir.glob("*.png")))
            print(f"  ✓  {n_out} images  |  "
                  f"avg {report['avg_gpu_per_sample_s']:.1f}s/img  |  "
                  f"total GPU {report['total_gpu_time_s']:.1f}s")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate CSVs from JSON prompt library → load model once → run all inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--studio",
                   default=DEFAULT_STUDIO,
                   help="Studio root: must contain easy/ medium/ hard/ image folders, "
                        "batch_infer.py, sample_prompts.py, camera_control.json.")
    p.add_argument("--n",
                   type=int, default=10,
                   help="Number of CSVs (= number of distinct prompts) per difficulty. "
                        "Up to 100 → all prompts distinct. "
                        "Over 100 → prompts repeat but seeds differ.")
    p.add_argument("--seed",        type=int, default=42,
                   help="Base random seed (controls prompt sampling + per-row seeds).")
    p.add_argument("--steps",       type=int, default=4,
                   help="Denoising steps (4=fast, 20=quality).")
    p.add_argument("--num_workers", type=int, default=4,
                   help="CPU DataLoader workers for parallel image loading.")
    p.add_argument("--force_size",  type=int, nargs=2, default=[1024, 1024],
                   metavar=("W", "H"),
                   help="Fixed output resolution. Required for stable torch.compile.")
    p.add_argument("--compile",  action="store_true",
                   help="torch.compile the diffusion transformer. "
                        "~2 min one-time warmup, then 15–30%% faster per sample.")
    p.add_argument("--fa2",      action="store_true",
                   help="FlashAttention-2 on text encoder. "
                        "Requires: pip install flash-attn --no-build-isolation")
    p.add_argument("--csv_only", action="store_true",
                   help="Regenerate CSVs only — do not run inference.")
    p.add_argument("--skip_csv", action="store_true",
                   help="Skip CSV regeneration — run inference on existing csv_output/.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    studio = Path(args.studio)

    csv_dir = studio / "csv_output"
    res_dir = studio / "results"

    print("=" * 62)
    print("  CONFIGURATION")
    print("=" * 62)
    print(f"  Studio root    : {studio}")
    print(f"  Image dirs     : {studio}/easy|medium|hard/")
    print(f"  Prompt library : {studio}/camera_control.json")
    print(f"  CSV output     : {csv_dir}")
    print(f"  Results        : {res_dir}")
    print(f"  CSVs / level   : {args.n}  ({args.n * 3} total)")
    print(f"  Steps          : {args.steps}")
    print(f"  Force size     : {args.force_size[0]}×{args.force_size[1]}")
    print(f"  torch.compile  : {args.compile}")
    print(f"  FlashAttn-2    : {args.fa2}")
    print("=" * 62)

    # ── Step 1: CSV generation ────────────────────────────────────────────────
    if not args.skip_csv:
        print("\n── STEP 1: Regenerating csv_output/ from camera_control.json ─")
        all_csvs = regenerate_csvs(
            studio=studio,
            csv_root=csv_dir,
            n=args.n,
            seed=args.seed,
        )
    else:
        print("\n── STEP 1: Skipped — loading existing CSVs ───────────────────")
        all_csvs: Dict[str, List[Path]] = {}
        for diff in ["easy", "medium", "hard"]:
            d    = csv_dir / diff
            csvs = sorted(d.glob(f"{diff}_*.csv")) if d.exists() else []
            all_csvs[diff] = csvs
            print(f"  {diff}: {len(csvs)} CSVs found")

    total = sum(len(v) for v in all_csvs.values())
    print(f"\n  Total CSVs ready : {total}  "
          f"(easy={len(all_csvs.get('easy', []))}, "
          f"medium={len(all_csvs.get('medium', []))}, "
          f"hard={len(all_csvs.get('hard', []))})")

    if args.csv_only:
        print("\n  --csv_only: stopping before inference.")
        print(f"\n  Inspect with:")
        print(f"    head -3 {csv_dir}/easy/easy_000.csv")
        print(f"    head -3 {csv_dir}/medium/medium_000.csv")
        print(f"    head -3 {csv_dir}/hard/hard_000.csv")
        return

    if total == 0:
        print("\n  [ERROR] No CSVs to process. Run without --skip_csv first.")
        sys.exit(1)

    # ── Step 2: Load model once ───────────────────────────────────────────────
    print("\n── STEP 2: Loading model (once for all CSVs) ─────────────────")
    print("  Model stays in VRAM until the script exits.")
    pipe = load_model(
        compile_flag=args.compile,
        fa2_flag=args.fa2,
        studio=studio,
    )

    # ── Step 3: Run inference ─────────────────────────────────────────────────
    print(f"\n── STEP 3: Running inference ({total} CSVs, model loaded once) ─")

    run_all(
        pipe=pipe,
        all_csvs=all_csvs,
        results_dir=res_dir,
        studio=studio,
        steps=args.steps,
        num_workers=args.num_workers,
        force_size=tuple(args.force_size),
        compile_flag=args.compile,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ALL DONE")
    print("=" * 62)
    for diff in ["easy", "medium", "hard"]:
        d = res_dir / diff
        if not d.exists():
            continue
        subdirs    = sorted(d.iterdir())
        total_imgs = sum(len(list(s.glob("*.png"))) for s in subdirs)
        print(f"\n  {diff.upper()}  —  {len(subdirs)} result folders  |  {total_imgs} images total")
        for s in subdirs[:3]:
            n_imgs = len(list(s.glob("*.png")))
            print(f"    {s.name}/  ({n_imgs} images)")
        if len(subdirs) > 3:
            print(f"    … +{len(subdirs) - 3} more folders")
    print()


if __name__ == "__main__":
    main()