"""
Flux.2-Klein 9B  Edit Image Generator
======================================
Reads the .txt prompts produced by generate_edit_prompts.py, pairs each
with the source human image, and generates the edited image via FLUX.2-klein-9B.

Parallel inference across N GPUs using srun (RANK / LOCAL_RANK from
SLURM env).  Each GPU runs fully independently — no torch.distributed.

Uses the proven model-loading and inference patterns from
generate_2k_edits.py, combined with the pipeline's own dataset logic
(flux_dataloader.py, directory structure, naming conventions).
"""

import argparse
import faulthandler
import gc
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback as tb
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from PIL import Image

# Unbuffered output — critical for SLURM log visibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────
# Rank info
# ─────────────────────────────────────────────────────────────
RANK       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",  "0")))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",  "1")))
IS_MAIN    = (RANK == 0)

def log(msg: str):
    print(f"[GPU {LOCAL_RANK} | RANK {RANK}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from diffusers import Flux2KleinPipeline
except ImportError as e:
    print(f"[RANK {RANK}] CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

# Add parent dir to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import HUMANS_DIR

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
IMAGE_HEIGHT     = 1024
IMAGE_WIDTH      = 1024
NUM_STEPS        = 4
GUIDANCE_SCALE   = 1.0      # Required for FLUX.2-klein — do NOT change


# ─────────────────────────────────────────────────────────────
# Dataset — reads .txt prompts and pairs with source images
# ─────────────────────────────────────────────────────────────
class FluxEditDataset(Dataset):
    """
    Pairs generated .txt prompts with source human images.
    Skips outputs that already exist (resumability).
    Structure:
    - prompts: {root_dir}/edited/prompts/{difficulty}/{gender}/{name}.txt
    - humans:  {HUMANS_DIR}/{gender}/{human_id}.png
    - output:  {root_dir}/edited/{difficulty}/{gender}/{name}.png
    """
    def __init__(self, root_dir, rank=0, world_size=1, batch_size=1):
        self.root_dir = root_dir
        all_samples = []

        prompts_root = os.path.join(root_dir, "edited", "prompts")
        humans_root = HUMANS_DIR

        if not os.path.exists(prompts_root):
            log(f"WARNING: Prompts directory not found: {prompts_root}")
            self.my_samples = []
            return

        # Minimum valid PNG size (catch partial writes from crashes)
        MIN_PNG_BYTES = 10_000

        for diff in ["easy"]:
            diff_path = os.path.join(prompts_root, diff)
            if not os.path.exists(diff_path):
                continue

            for gender in ["male", "female"]:
                gender_path = os.path.join(diff_path, gender)
                if not os.path.exists(gender_path):
                    continue

                output_dir = os.path.join(root_dir, "edited", diff, gender)
                os.makedirs(output_dir, exist_ok=True)

                files = sorted([f for f in os.listdir(gender_path) if f.endswith(".txt")])
                skipped = 0

                for f in files:
                    prompt_path = os.path.join(gender_path, f)
                    base_name = f.replace(".txt", "")
                    parts = base_name.split("_")

                    if len(parts) < 3:
                        continue

                    human_id = f"{parts[0]}_{parts[1]}"
                    source_img_path = os.path.join(humans_root, gender, f"{human_id}.png")

                    if not os.path.exists(source_img_path):
                        continue

                    output_img_path = os.path.join(output_dir, f"{base_name}.png")

                    # NOTE: Removed individual file skipping here.
                    # We append all samples to preserve deterministic batching,
                    # and check for completeness batch-wise below.

                    all_samples.append({
                        "prompt_path": prompt_path,
                        "source_img_path": source_img_path,
                        "output_img_path": output_img_path,
                        "base_name": base_name,
                        "difficulty": diff,
                        "gender": gender,
                    })

        # Shard across ranks from the COMPLETE list
        my_full_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]

        # Explicit override: no resume logic at all
        self.my_samples = my_full_samples
        self.skipped_batches = 0
        
        log(f"Dataset: Total={len(all_samples)} | Shard(Rank {rank})={len(my_full_samples)} items to process.")

    def __len__(self):
        return len(self.my_samples)

    def __getitem__(self, idx):
        s = self.my_samples[idx]
        source_image = Image.open(s["source_img_path"]).convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS
        )
        with open(s["prompt_path"], 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        return {
            "image": source_image,
            "prompt": prompt,
            "output_path": s["output_img_path"],
            "base_name": s["base_name"],
        }


# ─────────────────────────────────────────────────────────────
# Collate — keep PIL images as list (diffusers expects this)
# ─────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "prompts":    [b["prompt"]      for b in batch],
        "images":     [b["image"]       for b in batch],
        "out_paths":  [Path(b["output_path"]) for b in batch],
        "base_names": [b["base_name"]   for b in batch],
    }


# ─────────────────────────────────────────────────────────────
# Background image writer (proven pattern from generate_2k_edits.py)
# ─────────────────────────────────────────────────────────────
class BackgroundWriter:
    def __init__(self, num_workers=4):
        self.queue = queue.Queue(maxsize=32)
        self.workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            img, path = item
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path, format="PNG")
            except Exception as e:
                log(f"[BackgroundWriter] Error saving {path}: {e}")
            finally:
                self.queue.task_done()

    def submit(self, img: Image.Image, path: Path):
        self.queue.put((img, path))

    def join(self):
        self.queue.join()

    def stop(self):
        self.join()
        for _ in range(len(self.workers)):
            self.queue.put(None)
        for t in self.workers:
            t.join()


# ─────────────────────────────────────────────────────────────
# Pipeline loader with retry (handles transient CUDA errors)
# ─────────────────────────────────────────────────────────────
import gc

def load_pipeline(model_path: str, device: str, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        log(f"Loading FLUX.2-klein-9B (attempt {attempt}/{max_retries}) "
            f"from {model_path} on {device} ...")
        t0 = time.time()

        try:
            try:
                pipe = Flux2KleinPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                log("Flash Attention 2 enabled")
            except Exception as e:
                log(f"FA2 failed ({e}), falling back to default attention")
                pipe = Flux2KleinPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                )

            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)



            log(f"Model ready in {time.time() - t0:.1f}s")
            return pipe

        except torch.cuda.OutOfMemoryError:
            log("FATAL: CUDA OOM during model load — not retryable.")
            raise

        except Exception as e:
            log(f"Pipeline load attempt {attempt} failed: {e}")
            if attempt < max_retries:
                log(f"Cleaning up and retrying in 30s ...")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(30)
            else:
                log("All retry attempts exhausted.")
                raise


def run_inference(pipe, prompts, input_imgs, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompts,
        image=input_imgs,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        generator=generator,
    )
    return result.images


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # Enable faulthandler: prints Python traceback on SIGSEGV before crash
    faulthandler.enable(all_threads=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root dataset directory")
    parser.add_argument("--model_path", type=str,
                        default="black-forest-labs/FLUX.2-klein-9B",
                        help="Path to Flux model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test_limit", type=int, default=-1,
                        help="Total samples to process for testing")
    args = parser.parse_args()

    import socket
    JOB_START = datetime.now()

    log(f"host={socket.gethostname()} | RANK={RANK} LOCAL_RANK={LOCAL_RANK} "
        f"WORLD_SIZE={WORLD_SIZE} | start={JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")

    if IS_MAIN:
        print(f"\n{'='*60}", flush=True)
        print(f"FLUX.2-Klein 9B Image Generator  |  {WORLD_SIZE} workers", flush=True)
        print(f"Batch size : {args.batch_size} | Steps: {NUM_STEPS} | "
              f"Guidance: {GUIDANCE_SCALE} | Size: {IMAGE_WIDTH}×{IMAGE_HEIGHT}", flush=True)
        print(f"Output dir : {args.root_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)

    # Calculate local limit for testing
    local_limit = -1
    if args.test_limit > 0:
        local_limit = args.test_limit // WORLD_SIZE
        if RANK < (args.test_limit % WORLD_SIZE):
            local_limit += 1

    # ── Device Setup ─────────────────────────────────────────
    device_id = 0 if torch.cuda.device_count() == 1 else LOCAL_RANK
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    # ── Dataset ──────────────────────────────────────────────
    log("Loading dataset ...")
    dataset = FluxEditDataset(args.root_dir, rank=RANK, world_size=WORLD_SIZE, batch_size=args.batch_size)

    # Truncate dataset to local_limit so DataLoader has exact count
    if local_limit > 0 and len(dataset) > local_limit:
        dataset.my_samples = dataset.my_samples[:local_limit]
        log(f"Test mode: truncated dataset to {local_limit} samples")

    log(f"Dataset: {len(dataset)} images pending | "
        f"batch_size={args.batch_size} | "
        f"batches={-(-len(dataset) // args.batch_size)} (ceil)")

    if len(dataset) == 0:
        log("Nothing to do — all outputs exist or no prompts found.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False,
    )
    log(f"DataLoader ready: {len(dataloader)} batches.")

    # ── Load model ───────────────────────────────────────────
    pipe = load_pipeline(args.model_path, device)
    writer = BackgroundWriter(num_workers=4)

    # ── Warmup (first inference is always slower due to CUDA compilation) ──
    log("Warming up ...")
    t_warmup = time.time()
    first_item = dataset[0]
    _ = run_inference(
        pipe,
        prompts=[first_item["prompt"]],
        input_imgs=[first_item["image"]],
        device=device,
        seed=42 + RANK,
    )
    torch.cuda.synchronize(torch.device(device))
    log(f"Warmup done in {time.time() - t_warmup:.1f}s")

    log("Starting generation — each rank independent, no barrier.")

    # ── Main inference loop ──────────────────────────────────
    t_start        = time.time()
    local_generated = 0
    errors          = 0

    for batch_idx, batch in enumerate(dataloader):
        if local_limit > 0 and local_generated >= local_limit:
            log(f"Reached local test limit of {local_limit}. Stopping.")
            break

        prompts    = batch["prompts"]
        input_imgs = batch["images"]
        out_paths  = batch["out_paths"]

        try:
            t0 = time.time()
            result_images = run_inference(
                pipe,
                prompts=prompts,
                input_imgs=input_imgs,
                device=device,
                seed=hash(prompts[0]) % (2 ** 31),
            )
            torch.cuda.synchronize(torch.device(device))
            inf_time = time.time() - t0

            for img, out_path in zip(result_images, out_paths):
                writer.submit(img, out_path)

            local_generated += len(prompts)

            true_idx = batch_idx + 1 + dataset.skipped_batches
            true_total = len(dataloader) + dataset.skipped_batches

            log(f"Batch {true_idx}/{true_total} | "
                f"{len(prompts)} imgs in {inf_time:.1f}s "
                f"({len(prompts)/inf_time:.2f} img/s) | "
                f"done={local_generated}/{len(dataset)}")

        except Exception as e:
            errors += len(prompts)
            log(f"ERROR in batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    # ── Finalize ─────────────────────────────────────────────
    torch.cuda.synchronize(torch.device(device))
    log("Waiting for background IO ...")
    writer.stop()

    t_total    = time.time() - t_start
    throughput = local_generated / t_total if t_total > 0 else 0
    JOB_END    = datetime.now()

    log(f"\n{'─'*50}")
    log(f"SUMMARY  (Rank {RANK} / GPU {LOCAL_RANK})")
    log(f"{'─'*50}")
    log(f"  Images generated       : {local_generated}")
    log(f"  Errors                 : {errors}")
    log(f"  Job start              : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Job end                : {JOB_END.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Wall-clock time        : {t_total:.2f}s ({t_total/60:.1f} min)")
    log(f"  Throughput             : {throughput:.3f} images/sec")
    if local_generated > 0:
        log(f"  Avg time per image     : {t_total/local_generated:.3f}s")
    log(f"{'─'*50}\n")

    torch.cuda.empty_cache()
    log("Flux Generation Stage Complete.")


if __name__ == "__main__":
    # Register SIGSEGV handler: log and exit gracefully (exit 0)
    # so --kill-on-bad-exit=0 keeps other GPUs alive
    def _segfault_handler(signum, frame):
        print(f"[RANK {RANK}] SIGSEGV caught! Exiting gracefully.", flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # Hard exit without cleanup — process is corrupted

    signal.signal(signal.SIGSEGV, _segfault_handler)

    try:
        main()
    except Exception:
        log("FATAL EXCEPTION:")
        tb.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
