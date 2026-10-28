import os
import gc
import json
import time
import queue
import random
import threading
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# Removed importlib

# User specified pipeline
from diffusers import Flux2KleinPipeline
from PIL import Image
from tqdm import tqdm

# ========= CONFIG =========
ROOT_DIR         = "PATH_TO_ROOT_DIR"
LOCAL_OUTPUT_DIR = "PATH_TO_OUTPUT_DIR"
LOGS_DIR         = os.path.join(ROOT_DIR, "logs")
MODEL_NAME       = "black-forest-labs/FLUX.2-klein-9B"

BATCH_SIZE       = 1
IMAGE_HEIGHT     = 1024
IMAGE_WIDTH      = 1024
NUM_STEPS        = 4
GUIDANCE_SCALE   = 1.0  # Required for FLUX.2-klein

DATALOADER_WORKERS = 4
IMAGES_PER_GARMENT = 140

LOG_FILE = os.path.join(LOGS_DIR, "generate_cloth_log.txt")
# ========= CONFIG END =========

def load_garments_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        garments = json.load(f)
    return garments


def build_tasks(garments):
    # Fix the seed exactly here to guarantee all GPUs generate the EXACT same task list mathematically!
    random.seed(12345) 
    
    tasks = []
    male_counter = 1
    female_counter = 1
    
    # Generate all clothes in the list, then repeat 140 times
    for _ in range(IMAGES_PER_GARMENT):
        for g in garments:
            seed = random.randint(0, 2147483647)
            gender = g['gender']
            
            if gender == "male":
                fname = f"mc_{male_counter:06d}.png"
                male_counter += 1
            else:
                fname = f"fc_{female_counter:06d}.png"
                female_counter += 1
                
            prompt = f"A {g['name']} , which is a {g['description']} from {g['region']} region . The dress is of {g['dress_type']} type of dress . The background is either a solid soft color or grey background that contrasts the color of the dress. laid flat on the ground, top-down view, neutral floor, even light OR front view, soft even lighting, plain wall background, centered."
            
            tasks.append({
                "keywords": {
                    "name": g['name'],
                    "description": g['description'],
                    "region": g['region'],
                    "gender": gender,
                    "dress_type": g['dress_type'],
                },
                "prompt": prompt,
                "seed": seed,
                "filename": fname,
                "gender": gender
            })
            
    return tasks


class ClothDataset(Dataset):
    def __init__(self, tasks, root_dir):
        self.output_root = Path(root_dir)
        self.tasks = tasks
        
    def __len__(self):
        return len(self.tasks)
        
    def __getitem__(self, idx):
        task = self.tasks[idx]
        out_path = self.output_root / task["gender"] / task["filename"]
        
        return {
            "task_data": task,
            "prompt": task["prompt"],
            "seed": task["seed"],
            "out_path": str(out_path),
        }

def collate_fn(batch):
    return {
        "task_data": [b["task_data"] for b in batch],
        "prompts":   [b["prompt"]    for b in batch],
        "seeds":     [b["seed"]      for b in batch],
        "out_paths": [Path(b["out_path"]) for b in batch],
    }


# ── Background image writer ───────────────────────────────────────────────────
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
            img, path, task_data = item
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the image
                img.save(path, format="PNG")
                
                # Save the individual JSON sidecar file (with same base name as PNG)
                json_path = path.with_suffix('.json')
                record = {
                    "filename": task_data["filename"],
                    "keywords": task_data["keywords"],
                    "prompt": task_data["prompt"],
                    "seed": task_data["seed"]
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=4)
                    
            except Exception as e:
                print(f"Error saving {path}: {e}")
            finally:
                self.queue.task_done()

    def submit(self, img: Image.Image, path: Path, task_data: dict):
        self.queue.put((img, path, task_data))

    def stop(self):
        self.queue.join()
        for _ in range(len(self.workers)):
            self.queue.put(None)
        for t in self.workers:
            t.join()


# ── Distributed helpers ───────────────────────────────────────────────────────
def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def log(msg: str, rank: int = 0, force: bool = False):
    if rank != 0 and not force:
        return
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Pipeline ──────────────────────────────────────────────────────────────────
def load_pipeline(local_rank: int, rank: int) -> Flux2KleinPipeline:
    device = f"cuda:{local_rank}"
    log(f"[Rank {rank}] Loading model on {device} ...", rank, force=True)
    t0 = time.time()

    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        log(f"[Rank {rank}] Flash Attention 2 enabled", rank, force=True)
    except Exception as e:
        log(f"[Rank {rank}] FA2 failed ({e}), falling back to default attention", rank, force=True)
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Optionally compile
    log(f"[Rank {rank}] Compiling transformer (mode=default) ...", rank, force=True)
    pipe.transformer = torch.compile(pipe.transformer, mode="default")

    log(f"[Rank {rank}] Model ready in {time.time() - t0:.1f}s", rank, force=True)
    return pipe


def run_inference(
    pipe,
    prompts: list[str],
    seeds: list[int],
    device: str,
) -> list[Image.Image]:
    
    # Diffusers needs a list of generators for a batched seed if BATCH_SIZE > 1
    # But we usually just set one generator and let diffusers handle it or pass a list
    generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]
    
    # Note: explicitly omitting `image=` to trigger text-to-image behavior
    result = pipe(
        prompt=prompts,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        generator=generators,
    )
    return result.images


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"

    if rank == 0:
        Path(LOCAL_OUTPUT_DIR, "male").mkdir(parents=True, exist_ok=True)
        Path(LOCAL_OUTPUT_DIR, "female").mkdir(parents=True, exist_ok=True)
        Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Cloth Generation Log — {datetime.now()} ===\n")

    dist.barrier()

    log(f"{'='*70}", rank)
    log(f"Cloth Image Generator (Flux Text-to-Image) | world_size={world_size}", rank)
    log(f"Output dir: {LOCAL_OUTPUT_DIR}", rank)
    log(f"{'='*70}", rank)

    # ── Load data ────────────────────────────────────────────────────────────
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "CURVTON_assets", "dictionaries", "cloth_dictionary.json")
    garments = load_garments_dict(json_path)
    
    # Build complete uniform task list
    all_tasks = build_tasks(garments)
    
    # Shard for this GPU
    my_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == rank]

    log(f"Total tasks: {len(all_tasks)}. Rank {rank} assigned {len(my_tasks)} tasks.", rank, force=True)

    dataset = ClothDataset(my_tasks, LOCAL_OUTPUT_DIR)
    
    if len(dataset) == 0:
        log(f"[Rank {rank}] Nothing to do!", rank, force=True)
        cleanup_distributed()
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2 if DATALOADER_WORKERS > 0 else None,
        persistent_workers=True if DATALOADER_WORKERS > 0 else False,
        drop_last=False,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    pipe   = load_pipeline(local_rank, rank)
    writer = BackgroundWriter(num_workers=2)

    dist.barrier()
    log("All ranks ready — starting generation ...", rank)

    # ── Main inference loop ───────────────────────────────────────────────────
    t_start         = time.time()
    local_generated = 0
    errors          = 0

    pbar = tqdm(
        dataloader,
        desc=f"GPU {local_rank} (rank {rank})",
        position=local_rank,
        disable=(rank != 0),
        total=len(dataloader),
    )

    for batch in pbar:
        task_data_list = batch["task_data"]
        prompts    = batch["prompts"]
        seeds      = batch["seeds"]
        out_paths  = batch["out_paths"]

        try:
            t0 = time.time()
            result_images = run_inference(
                pipe,
                prompts=prompts,
                seeds=seeds,
                device=device,
            )
            torch.cuda.synchronize(device)
            inf_time = time.time() - t0

            for img, out_path, task_info in zip(result_images, out_paths, task_data_list):
                writer.submit(img, out_path, task_info)

            local_generated += len(prompts)

            if rank == 0:
                pbar.set_postfix(
                    imgs_s=f"{len(prompts) / inf_time:.2f}",
                    inf=f"{inf_time:.1f}s",
                    done=local_generated,
                )

        except Exception as e:
            errors += len(prompts)
            log(f"[Rank {rank}] ERROR: {e}", rank, force=True)
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    # ── Finalise ─────────────────────────────────────────────────────────────
    torch.cuda.synchronize(device)
    log(f"[Rank {rank}] Waiting for background IO ...", rank, force=True)
    writer.stop()
    dist.barrier()

    t_total    = time.time() - t_start
    gen_tensor = torch.tensor(local_generated, device=device, dtype=torch.int64)
    err_tensor = torch.tensor(errors,          device=device, dtype=torch.int64)
    dist.all_reduce(gen_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(err_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        total_gen  = gen_tensor.item()
        total_err  = err_tensor.item()
        throughput = total_gen / t_total if t_total > 0 else 0
        log("\n" + "=" * 70, rank)
        log("GENERATION COMPLETE", rank)
        log("=" * 70, rank)
        log(f"Total garments       : {len(garments)}", rank)
        log(f"Total images requested: {len(garments) * IMAGES_PER_GARMENT} ({IMAGES_PER_GARMENT}/garment)", rank)
        log(f"Total images gen'd   : {total_gen}", rank)
        log(f"Total errors         : {total_err}", rank)
        log(f"Wall-clock time      : {t_total:.2f}s  ({t_total / 60:.1f} min)", rank)
        log(f"Aggregate throughput : {throughput:.3f} images/s", rank)
        log("=" * 70, rank)

    cleanup_distributed()


if __name__ == "__main__":
    main()
