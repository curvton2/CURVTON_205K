import os
import gc
import json
import time
import queue
import threading
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from diffusers import Flux2KleinPipeline
from PIL import Image
from tqdm import tqdm

# ========= CONFIG =========
LOCAL_OUTPUT_DIR = "PATH_TO_OUTPUT_DIR"
MODEL_NAME       = "black-forest-labs/FLUX.2-klein-9B"
JSONL_PATTERN    = "PATH_TO_JSONL_PATTERN_WITH_RANK"
INPUT_IMAGES     = {
    "male":   "MALE_IMAGE.png",
    "female": "FEMALE_IMAGE.png",
}

BATCH_SIZE       = 4
IMAGE_HEIGHT     = 1024
IMAGE_WIDTH      = 1024
NUM_STEPS        = 4
GUIDANCE_SCALE   = 1.0      # Required for FLUX.2-klein — do NOT change
USE_COMPILE      = False    # Set True for ~2x speedup after warmup; False if compile errors

# DataLoader workers — prefetch next batch while GPU runs current one
DATALOADER_WORKERS = 4

LOGS_DIR         = os.path.join(LOCAL_OUTPUT_DIR, "logs")
LOG_FILE         = os.path.join(LOGS_DIR, "2k_edits_log.txt")
# ========= CONFIG END =========


# ── Dataset ──────────────────────────────────────────────────────────────────

class EditDataset(Dataset):
    """
    Each item is one record from the JSONL.
    Images are loaded and resized by the DataLoader workers in parallel,
    so the GPU is never waiting on image I/O or preprocessing.
    Only returns PENDING (not already generated) records.
    """
    def __init__(
        self,
        records: list[dict],
        source_images: dict[str, Image.Image],
        output_root: Path,
    ):
        self.source_images = source_images
        self.output_root   = output_root

        # Filter to only records whose output doesn't exist yet
        self.records = [r for r in records if not self._out_path(r).exists()]

    def _out_path(self, record: dict) -> Path:
        source_file = record.get("source_file", "").lower()
        gender      = record.get("gender", "").lower()
        file_name   = record["file_name"]
        tier = "hard" if "hard" in source_file else "medium"
        return self.output_root / tier / gender / f"{file_name}.png"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec    = self.records[idx]
        gender = rec["gender"].lower()
        img    = self.source_images[gender]   # already resized PIL image

        return {
            "prompt":    rec["edit_prompt"],
            "image":     img,
            "out_path":  str(self._out_path(rec)),
            "file_name": rec["file_name"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of dataset items into a batch dict.
    Keeps images as a list of PIL Images (what diffusers expects).
    """
    return {
        "prompts":    [b["prompt"]    for b in batch],
        "images":     [b["image"]     for b in batch],
        "out_paths":  [Path(b["out_path"]) for b in batch],
        "file_names": [b["file_name"] for b in batch],
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
            img, path = item
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path, format="PNG")
            except Exception as e:
                print(f"Error saving {path}: {e}")
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


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(filepath: str) -> list[dict]:
    records = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_source_images() -> dict[str, Image.Image]:
    images = {}
    for gender, path in INPUT_IMAGES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source image not found: '{path}'")
        images[gender] = Image.open(path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    return images


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

    if USE_COMPILE:
        log(f"[Rank {rank}] Compiling transformer ...", rank, force=True)
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=True,
        )

    log(f"[Rank {rank}] Model ready in {time.time() - t0:.1f}s", rank, force=True)
    return pipe


def run_inference(
    pipe,
    prompts: list[str],
    input_imgs: list[Image.Image],
    device: str,
    seed: int,
) -> list[Image.Image]:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"

    if rank == 0:
        for tier in ("medium", "hard"):
            for gender in ("male", "female"):
                Path(LOCAL_OUTPUT_DIR, tier, gender).mkdir(parents=True, exist_ok=True)
        Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== 2k Edit Images Log — {datetime.now()} ===\n")

    dist.barrier()

    log(f"{'='*70}", rank)
    log(f"FLUX.2-Klein 9B  |  world_size={world_size}  |  batch_size={BATCH_SIZE}  |  "
        f"guidance_scale={GUIDANCE_SCALE}  |  steps={NUM_STEPS}  |  "
        f"dataloader_workers={DATALOADER_WORKERS}  |  compile={USE_COMPILE}", rank)
    log(f"Output dir: {LOCAL_OUTPUT_DIR}", rank)
    log(f"{'='*70}", rank)

    # ── Load data ────────────────────────────────────────────────────────────
    jsonl_path = JSONL_PATTERN.format(rank=rank)
    records    = load_jsonl(jsonl_path)
    if not records:
        log(f"[Rank {rank}] No records in '{jsonl_path}' — exiting.", rank, force=True)
        cleanup_distributed()
        return

    log(f"[Rank {rank}] {len(records)} total records loaded", rank, force=True)

    source_images = load_source_images()
    output_root   = Path(LOCAL_OUTPUT_DIR)

    dataset = EditDataset(records, source_images, output_root)
    log(f"[Rank {rank}] {len(dataset)} images pending (skipping {len(records) - len(dataset)} already done)",
        rank, force=True)

    if len(dataset) == 0:
        log(f"[Rank {rank}] Nothing to do — all outputs exist.", rank, force=True)
        cleanup_distributed()
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True if DATALOADER_WORKERS > 0 else False,
        drop_last=False,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    pipe   = load_pipeline(local_rank, rank)
    writer = BackgroundWriter(num_workers=4)

    dist.barrier()

    # ── Warmup ───────────────────────────────────────────────────────────────
    log(f"[Rank {rank}] Warming up ...", rank, force=True)
    t_warmup = time.time()
    warmup_rec = records[0]
    _ = run_inference(
        pipe,
        prompts=[warmup_rec["edit_prompt"]],
        input_imgs=[source_images[warmup_rec["gender"]]],
        device=device,
        seed=42 + rank,
    )
    torch.cuda.synchronize(device)
    log(f"[Rank {rank}] Warmup done in {time.time() - t_warmup:.1f}s", rank, force=True)

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
            torch.cuda.synchronize(device)
            inf_time = time.time() - t0

            for img, out_path in zip(result_images, out_paths):
                writer.submit(img, out_path)

            local_generated += len(prompts)

            log(
                f"[Rank {rank}] {len(prompts)} imgs in {inf_time:.1f}s "
                f"({len(prompts)/inf_time:.2f} img/s) | done={local_generated}/{len(dataset)}",
                rank, force=True,
            )

            if rank == 0:
                pbar.set_postfix(
                    imgs_s=f"{len(prompts) / inf_time:.2f}",
                    inf=f"{inf_time:.1f}s",
                    done=local_generated,
                )

        except Exception as e:
            errors += len(prompts)
            log(f"[Rank {rank}] ERROR: {e}", rank, force=True)
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
        log(f"Total images generated : {total_gen}", rank)
        log(f"Total errors           : {total_err}", rank)
        log(f"Wall-clock time        : {t_total:.2f}s  ({t_total / 60:.1f} min)", rank)
        log(f"Aggregate throughput   : {throughput:.3f} images/s  (all {world_size} GPUs combined)", rank)
        log(f"Avg time per image     : {t_total / max(total_gen, 1):.3f}s", rank)
        log("=" * 70, rank)

    cleanup_distributed()


if __name__ == "__main__":
    main()