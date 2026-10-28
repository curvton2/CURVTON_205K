"""
Post Try-On Camera Edit
=======================
Randomly selects 15% of completed try-on images and their corresponding initial person images 
from Medium and Hard difficulties. 
Applies the exact same random camera control prompt to both images simultaneously.
Uses Qwen Image Edit batch inference.
"""

import argparse
import os
import sys
import random
import json
import faulthandler
from pathlib import Path

# Need to load modules from vton root
VTON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(VTON_ROOT)

try:
    from batch_infer import (
        load_pipeline, CameraEditDataset, camera_edit_collate, run_inference, LatencyTracker
    )
    from sample_prompts import load_library, sample_flat
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

import torch
from torch.utils.data import DataLoader

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root dataset directory")
    parser.add_argument("--camera_control_json", type=str, required=True, help="Path to camera_control.json")
    parser.add_argument("--dataset_end_dir", type=str, required=True, help="Path to ultimate dataset")
    parser.add_argument("--percentage", type=float, default=0.15, help="Percentage to sample (0.15 = 15%)")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    
    RANK = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    
    device_id = 0 if torch.cuda.device_count() == 1 else LOCAL_RANK
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    
    ultimate_dir = args.dataset_end_dir
    lib = load_library(Path(args.camera_control_json))
    
    pool_easy = sample_flat(lib, "easy")
    pool_medium = sample_flat(lib, "medium")
    pool_hard = sample_flat(lib, "hard")
    
    prompt_pools = {
        "medium": pool_easy + pool_medium,
        "hard": pool_medium + pool_hard
    }
    
    all_pairs = []
    
    for diff in ["medium", "hard"]:
        for gender in ["male", "female"]:
            tryon_dir = os.path.join(ultimate_dir, diff, gender, "tryon_image")
            person_dir = os.path.join(ultimate_dir, diff, gender, "initial_person_image")
            
            if not os.path.exists(tryon_dir) or not os.path.exists(person_dir):
                continue
                
            tryon_files = sorted([f for f in os.listdir(tryon_dir) if f.endswith(".png")])
            person_files = set(f for f in os.listdir(person_dir) if f.endswith(".png"))
            
            # Select 15% globally for this combo
            # Seed the random generator deterministically so all nodes build the same list before sharding
            rng = random.Random(f"{diff}_{gender}_12345")
            num_to_select = max(1, int(len(tryon_files) * args.percentage))
            if len(tryon_files) == 0:
                continue
            selected_files = rng.sample(tryon_files, num_to_select)
            
            for f in selected_files:
                parts = f.replace(".png", "").split("_")
                if len(parts) >= 3:
                     person_base = f"{parts[0]}_{parts[1]}_{parts[2]}" 
                     expected_person_file = f"{person_base}.png"
                else:
                     continue
                     
                if expected_person_file not in person_files:
                    continue
                    
                # Sample a prompt string for this pair
                prompt = rng.choice(prompt_pools[diff])
                seed_for_edit = rng.randint(0, 2**31 - 1)
                
                tryon_full = os.path.join(tryon_dir, f)
                person_full = os.path.join(person_dir, expected_person_file)
                
                all_pairs.append({
                    "tryon_path": tryon_full,
                    "person_path": person_full,
                    "prompt": prompt,
                    "seed": seed_for_edit,
                    "diff": diff,
                    "gender": gender,
                })
                
    # Shard samples across GPUs
    my_pairs = [p for i, p in enumerate(all_pairs) if i % WORLD_SIZE == RANK]
    
    if RANK == 0:
        print(f"Post Try-On Camera Edit: Selected {len(all_pairs)} pairs globally (15% of Medium/Hard).")
        
    if not my_pairs:
        print(f"[RANK {RANK}] No pairs assigned.")
        return
        
    raw_samples = []
    for p in my_pairs:
        # Initial Person Image sample
        raw_samples.append({
            "image_path": p["person_path"],
            "transform": "",
            "prompt": p["prompt"],
            "seed": p["seed"]
        })
        # Try-On Image sample
        raw_samples.append({
            "image_path": p["tryon_path"],
            "transform": "",
            "prompt": p["prompt"],
            "seed": p["seed"]
        })
        
    # Save all edited results directly to a new folder in ultimate to keep them organized
    output_dir = Path(ultimate_dir) / "post_tryon_camera_edited"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = CameraEditDataset(raw_samples=raw_samples, force_size=[1024, 1024])
    
    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=camera_edit_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(dataset, **loader_kwargs)
    
    print(f"[RANK {RANK}] Loading model...")
    pipe = load_pipeline(
        device=device,
        dtype=torch.bfloat16,
        compile_transformer=args.compile,
    )
    
    print(f"[RANK {RANK}] Processing {len(dataset)} camera edits...")
    
    tracker: LatencyTracker = run_inference(
        pipe=pipe,
        dataloader=dataloader,
        output_dir=output_dir,
        num_inference_steps=args.steps,
        true_guidance_scale=1.0,  # Ensure true_cfg is 1.0 per spec
        device=device,
        compiled=args.compile,
        force_size=(1024, 1024),
    )

    import shutil
    print(f"[RANK {RANK}] Replacing original images with camera-edited versions...")
    for rec in tracker.records:
        orig = Path(rec["image_path"])
        out_path = output_dir / f"{orig.stem}_out{int(rec['sample_idx']):04d}.png"
        if out_path.exists():
            shutil.move(str(out_path), str(orig))

    if RANK == 0:
        print("\n--- Post-Edit Camera Adjustments Complete ---")
        report_path = output_dir / "latency.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(tracker.summary(), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    main()
