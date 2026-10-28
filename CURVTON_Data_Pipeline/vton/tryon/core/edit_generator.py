"""
Flux.2-Klein 9B Try-On Image Generator
=======================================
Reads initial person images and cloth images from dataset_ultimate.
Generates try-on images. If retry_prompt/{name}.json exists for the pair (written by
rename.py after the evaluator rejected the previous attempt), its 'improved_prompt' is
used. Otherwise the prompt is built from DRESS_INFO, falling back to:
'Make this person try on this cloth.'

Images input: 
1. initial person image
2. cloth image
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

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

RANK       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",  "0")))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",  "1")))
IS_MAIN    = (RANK == 0)

def log(msg: str):
    print(f"[GPU {LOCAL_RANK} | RANK {RANK}] {msg}", flush=True)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from diffusers import Flux2KleinPipeline
except ImportError as e:
    print(f"[RANK {RANK}] CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

try:
    sys.path.append(os.path.dirname(__file__))
    from dress_info import DRESS_INFO
except ImportError:
    log("Warning: dress_info.py not found, garment prompts will use fallback.")
    DRESS_INFO = []

IMAGE_HEIGHT     = 1024
IMAGE_WIDTH      = 1024
NUM_STEPS        = 4
GUIDANCE_SCALE   = 1.0


class TryonFluxDataset(Dataset):
    def __init__(self, root_dir, dataset_v3_dir, rank=0, world_size=1, batch_size=1, category_limit=-1, iteration=1):
        self.root_dir = root_dir
        self.dataset_v3_dir = dataset_v3_dir
        self.category_limit = category_limit
        self.iteration = iteration
        
        all_samples = []

        for diff in ["easy", "medium", "hard"]:
            for gender in ["male", "female"]:
                # Read inputs from dataset_v3/dataset_ultimate
                cloth_dir = os.path.join(self.dataset_v3_dir, "dataset_ultimate", diff, gender, "cloth_image")
                person_dir = os.path.join(self.dataset_v3_dir, "dataset_ultimate", diff, gender, "initial_person_image")
                
                # Save tryon images directly alongside person and cloth images
                tryon_dir = os.path.join(self.dataset_v3_dir, "dataset_ultimate", diff, gender, "tryon_image")
                os.makedirs(tryon_dir, exist_ok=True)
                retry_dir = os.path.join(self.dataset_v3_dir, "dataset_ultimate", diff, gender, "retry_prompt")
                
                if not os.path.exists(cloth_dir) or not os.path.exists(person_dir):
                    continue
                
                person_files = set(f for f in os.listdir(person_dir) if f.endswith(".png"))
                cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.endswith(".png")])
                
                category_count = 0
                for cloth_file in cloth_files:
                    parts = cloth_file.split("_")
                    if len(parts) >= 3:
                        person_base = f"{parts[0]}_{parts[1]}_{parts[2]}" 
                        expected_person_file = f"{person_base}.png"
                        
                        generated_base = cloth_file.replace(".png", "")
                        generated_file = cloth_file
                        
                        if expected_person_file not in person_files:
                            continue
                        
                        category_count += 1
                        
                        # Output path: tryon_image/ alongside initial_person_image/ and cloth_image/
                        output_path = os.path.join(tryon_dir, generated_file)
                        if os.path.exists(output_path):
                            # Already generated, skip
                            continue
                        
                        # Determine initial prompt using dress_info or fallback
                        prompt = "Make this person try on this cloth."
                        
                        # Match cloth filename suffix against DRESS_INFO entries
                        # Cloth names use _ for spaces: daura_suruwal → "Daura Suruwal"
                        cloth_no_ext = cloth_file.replace(".png", "").lower()
                        
                        best_match_len = -1
                        found_desc = None
                        found_type = None
                        found_name = None
                        
                        for d in DRESS_INFO:
                            d_name_underscored = d["name"].replace(" ", "_").lower()
                            
                            if cloth_no_ext.endswith(d_name_underscored):
                                if len(d_name_underscored) > best_match_len:
                                    best_match_len = len(d_name_underscored)
                                    found_desc = d["description"]
                                    found_type = d["dress_type"]
                                    found_name = d["name"]
                                    
                        if found_desc:
                            prompt = f"Make this person wear this dress, which is a {found_name}, a {found_type} characterized by its {found_desc}."

                        # Retry: use the evaluator's improved prompt from the last rejection
                        retry_json = os.path.join(retry_dir, f"{generated_base}.json")
                        if os.path.exists(retry_json):
                            try:
                                with open(retry_json, "r", encoding="utf-8") as rf:
                                    improved = (json.load(rf).get("improved_prompt") or "").strip()
                                if improved:
                                    prompt = improved
                            except (OSError, ValueError) as e:
                                log(f"Could not read {retry_json}: {e}")

                        all_samples.append({
                            "prompt": prompt,
                            "person_img_path": os.path.join(person_dir, expected_person_file),
                            "cloth_img_path": os.path.join(cloth_dir, cloth_file),
                            "output_img_path": output_path,
                            "base_name": generated_base,
                            "difficulty": diff,
                            "gender": gender,
                        })
                        
                        if self.category_limit > 0 and category_count >= self.category_limit:
                            break

        my_full_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]
        self.my_samples = my_full_samples
        self.skipped_batches = 0
        
        log(f"Dataset: Total={len(all_samples)} | Shard(Rank {rank})={len(my_full_samples)} items to process.")

    def __len__(self):
        return len(self.my_samples)

    def __getitem__(self, idx):
        s = self.my_samples[idx]
        person_image = Image.open(s["person_img_path"]).convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS
        )
        cloth_image = Image.open(s["cloth_img_path"]).convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS
        )

        # FLUX.2-klein-9B accepts multiple images as a list
        # Pass person and cloth separately — the pipeline handles conditioning internally
        return {
            "person_image": person_image,
            "cloth_image": cloth_image,
            "prompt": s["prompt"],
            "output_path": s["output_img_path"],
            "base_name": s["base_name"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "prompts":       [b["prompt"]        for b in batch],
        "person_images": [b["person_image"]  for b in batch],
        "cloth_images":  [b["cloth_image"]   for b in batch],
        "out_paths":     [Path(b["output_path"]) for b in batch],
        "base_names":    [b["base_name"]     for b in batch],
    }


class BackgroundWriter:
    """Async image writer — uses a thread pool to save images to disk
    without blocking the GPU inference loop."""

    def __init__(self, num_workers=8):
        self.queue = queue.Queue(maxsize=128)
        self.workers = []
        self._error_count = 0
        self._saved_count = 0
        self._lock = threading.Lock()
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

                # Cleanup _incorrect files if they exist
                incorrect_png = path.with_name(f"{path.stem}_incorrect.png")
                incorrect_json = path.with_name(f"{path.stem}_incorrect.json")
                if incorrect_png.exists():
                    os.remove(incorrect_png)
                if incorrect_json.exists():
                    os.remove(incorrect_json)

                with self._lock:
                    self._saved_count += 1
            except Exception as e:
                log(f"[BackgroundWriter] Error saving {path}: {e}")
                with self._lock:
                    self._error_count += 1
            finally:
                self.queue.task_done()

    def submit(self, img: Image.Image, path: Path):
        """Non-blocking submit — enqueue image for async disk write."""
        self.queue.put((img, path))

    def join(self):
        self.queue.join()

    def stop(self):
        self.join()
        for _ in range(len(self.workers)):
            self.queue.put(None)
        for t in self.workers:
            t.join()
        log(f"[BackgroundWriter] Done. Saved={self._saved_count}, Errors={self._error_count}")


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

            # torch.compile the transformer for faster inference
            try:
                pipe.transformer = torch.compile(
                    pipe.transformer,
                    mode="max-autotune",
                    fullgraph=False,
                )
                log("torch.compile applied to transformer (max-autotune)")
            except Exception as compile_err:
                log(f"torch.compile failed ({compile_err}), continuing without it")

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


@torch.inference_mode()
def run_inference(pipe, prompts, person_imgs, cloth_imgs, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    # FLUX.2-klein-9B takes images as a list: [person_image, cloth_image]
    image_list = [person_imgs[0], cloth_imgs[0]]
    result = pipe(
        prompt=prompts,
        image=image_list,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        generator=generator,
    )
    return result.images


def main():
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
    parser.add_argument("--category_limit", type=int, default=-1,
                        help="Per category (easy/medium/hard + male/female) sample limit")
    parser.add_argument("--dataset_v3_dir", type=str, required=True,
                        help="Path to the original dataset_v3 containing initial person and cloth images")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Which iteration we are in")
    args = parser.parse_args()

    import socket
    JOB_START = datetime.now()

    if IS_MAIN:
        print(f"\n{'='*60}", flush=True)
        print(f"Try-On Image Generator  |  {WORLD_SIZE} workers", flush=True)
        print(f"{'='*60}\n", flush=True)

    device_id = 0 if torch.cuda.device_count() == 1 else LOCAL_RANK
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    log("Loading dataset ...")
    dataset = TryonFluxDataset(
        args.root_dir,
        args.dataset_v3_dir,
        rank=RANK, 
        world_size=WORLD_SIZE, 
        batch_size=args.batch_size,
        category_limit=args.category_limit,
        iteration=args.iteration
    )

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

    pipe = load_pipeline(args.model_path, device)
    writer = BackgroundWriter(num_workers=8)

    log("Warming up ...")
    t_warmup = time.time()
    first_item = dataset[0]
    _ = run_inference(
        pipe,
        prompts=[first_item["prompt"]],
        person_imgs=[first_item["person_image"]],
        cloth_imgs=[first_item["cloth_image"]],
        device=device,
        seed=42 + RANK,
    )
    torch.cuda.synchronize(torch.device(device))
    log(f"Warmup done in {time.time() - t_warmup:.1f}s")

    t_start        = time.time()
    local_generated = 0
    errors          = 0
    local_limit     = args.test_limit

    for batch_idx, batch in enumerate(dataloader):
        if local_limit > 0 and local_generated >= local_limit:
            break

        prompts      = batch["prompts"]
        person_imgs  = batch["person_images"]
        cloth_imgs   = batch["cloth_images"]
        out_paths    = batch["out_paths"]

        try:
            t0 = time.time()
            result_images = run_inference(
                pipe,
                prompts=prompts,
                person_imgs=person_imgs,
                cloth_imgs=cloth_imgs,
                device=device,
                seed=hash(prompts[0]) % (2 ** 31),
            )
            torch.cuda.synchronize(torch.device(device))
            inf_time = time.time() - t0

            for img, out_path in zip(result_images, out_paths):
                writer.submit(img, out_path)

            local_generated += len(prompts)
            log(f"Batch {batch_idx+1}/{len(dataloader)} generated in {inf_time:.1f}s")

        except Exception as e:
            errors += len(prompts)
            log(f"ERROR: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    writer.stop()
    log("Flux Generation Stage Complete.")

if __name__ == "__main__":
    def _segfault_handler(signum, frame):
        print(f"[RANK {RANK}] SIGSEGV caught! Exiting gracefully.", flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(0) 

    signal.signal(signal.SIGSEGV, _segfault_handler)
    try:
        main()
    except Exception:
        tb.print_exc(file=sys.stdout)
        sys.exit(1)
