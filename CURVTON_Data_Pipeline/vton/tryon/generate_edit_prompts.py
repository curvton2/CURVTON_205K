"""
Edit Prompt Generator  — Qwen3-VL-32B-Instruct
================================================
For each keyword JSON entry (medium/hard × male/female), feed the VLM
the paired person image + keyword JSON and ask it to write a precise
image-editing instruction.

Parallel inference across N GPUs using srun (RANK / LOCAL_RANK from
SLURM env).  Each GPU runs fully independently — no torch.distributed.

Uses the proven model-loading and inference patterns from
qwen_edit_prompts.py, combined with the pipeline's own dataset logic
(image pool recycling, naming conventions, directory structure).
"""

import argparse
import faulthandler
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback as tb
from datetime import datetime
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
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError as e:
    print(f"[RANK {RANK}] CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

# Add parent dir to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import EDIT_SYSTEM_PROMPT

IMAGE_SIZE = 512


# ─────────────────────────────────────────────────────────────
# Dataset — same logic as before (image pool cycling, naming)
# ─────────────────────────────────────────────────────────────
class EditPromptDataset(Dataset):
    def __init__(self, humans_root, jsonl_dir, output_dir, rank=0, world_size=1, batch_size=1, start_batch=0):
        self.humans_root = humans_root
        all_samples = []

        # Discover images
        image_pool = {"male": [], "female": []}
        for gender in ["male", "female"]:
            img_dir = os.path.join(humans_root, gender)
            if os.path.exists(img_dir):
                files = sorted(
                    [f for f in os.listdir(img_dir) if f.endswith(".png")],
                    key=lambda x: int(''.join(filter(str.isdigit, x)))
                    if any(c.isdigit() for c in x) else 0
                )
                image_pool[gender] = [os.path.join(img_dir, f) for f in files]

        # Pre-scan output dirs ONCE to build set of existing VALID .txt files
        # This avoids 256K individual os.path.exists() calls on shared FS
        # Files < MIN_TXT_BYTES are treated as truncated from a crash
        MIN_TXT_BYTES = 50
        existing_files = set()
        truncated_count = 0
        for diff in ["easy"]:
            for gender_key in ["male", "female"]:
                scan_dir = os.path.join(output_dir, diff, gender_key)
                if os.path.isdir(scan_dir):
                    for f in os.listdir(scan_dir):
                        if f.endswith(".txt"):
                            fpath = os.path.join(scan_dir, f)
                            try:
                                if os.path.getsize(fpath) >= MIN_TXT_BYTES:
                                    existing_files.add(f)
                                else:
                                    truncated_count += 1
                            except OSError:
                                pass
        log(f"Resume scan: {len(existing_files)} valid .txt files found"
            f"{f', {truncated_count} truncated (will re-generate)' if truncated_count else ''}")

        # Strict order: Male -> Female, then Easy
        skipped = 0
        for gender_key in ["male", "female"]:
            for diff in ["easy"]:
                image_idx_counter = 0
                usage_counts = {}

                pattern = f"{diff}_{gender_key}"
                target_files = sorted([
                    f for f in os.listdir(jsonl_dir)
                    if (f.endswith(".json") or f.endswith(".jsonl"))
                    and pattern in f.lower()
                ])

                for f in target_files:
                    fpath = os.path.join(jsonl_dir, f)
                    with open(fpath, 'r', encoding='utf-8') as jf:
                        if f.endswith(".json"):
                            entries = json.load(jf)
                            if not isinstance(entries, list):
                                entries = [entries]
                        else:
                            entries = [json.loads(line) for line in jf if line.strip()]

                    for data in entries:
                        pool = image_pool[gender_key]
                        if not pool:
                            continue

                        pool_idx = image_idx_counter % len(pool)
                        img_path = pool[pool_idx]

                        base_id = os.path.splitext(os.path.basename(img_path))[0]
                        usage_counts[base_id] = usage_counts.get(base_id, 0) + 1
                        final_name = f"{base_id}_e{usage_counts[base_id]:02d}"

                        # NOTE: Removed individual file skipping here.
                        # We append all samples to preserve deterministic batching,
                        # and check for completeness batch-wise below.

                        all_samples.append({
                            "image_path": img_path,
                            "attributes": data,
                            "file_name": final_name,
                            "gender": gender_key,
                            "difficulty": diff,
                        })
                        image_idx_counter += 1

        # Shard across ranks from the COMPLETE list
        my_full_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]

        # Explicitly skip exact number of batches
        self.samples = []
        skipped_batches = 0
        
        for i in range(0, len(my_full_samples), batch_size):
            batch = my_full_samples[i:i+batch_size]
            if skipped_batches < start_batch:
                skipped_batches += 1
                continue
            
            self.samples.extend(batch)

        self.skipped_batches = skipped_batches
        log(f"Dataset: Total={len(all_samples)} | Shard(Rank {rank})={len(my_full_samples)} -> "
            f"Explicitly skipped {skipped_batches} batches. Pending samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
        )
        # Exclude internal metadata keys from keywords sent to the VLM
        exclude = {"image_path", "file_name", "gender", "difficulty"}
        keywords = {k: v for k, v in s["attributes"].items() if k not in exclude}

        return {
            "image": image,
            "user_prompt": json.dumps(keywords, ensure_ascii=False),
            "file_name": s["file_name"],
            "gender": s["gender"],
            "difficulty": s["difficulty"],
        }


# ─────────────────────────────────────────────────────────────
# Collate — pre-tokenize in the collate function (proven pattern)
# ─────────────────────────────────────────────────────────────
def make_collate_fn(processor):
    pad_id = processor.tokenizer.pad_token_id or 0

    def collate_fn(items: List[Dict]) -> Dict:
        encoded = []
        for it in items:
            msgs = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": EDIT_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": it["image"]},
                        {"type": "text",  "text": it["user_prompt"]},
                    ],
                },
            ]
            enc = processor.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            encoded.append(enc)

        max_len = max(e["input_ids"].shape[-1] for e in encoded)

        def lpad(t, fill):
            gap = max_len - t.shape[-1]
            return t if gap == 0 else torch.cat(
                [torch.full((1, gap), fill, dtype=t.dtype), t], dim=-1
            )

        return {
            "input_ids":      torch.cat([lpad(e["input_ids"],      pad_id) for e in encoded]),
            "attention_mask": torch.cat([lpad(e["attention_mask"], 0)      for e in encoded]),
            "pixel_values":   torch.cat([e["pixel_values"]   for e in encoded]),
            "image_grid_thw": torch.cat([e["image_grid_thw"] for e in encoded]),
            "file_names":     [it["file_name"]  for it in items],
            "genders":        [it["gender"]     for it in items],
            "difficulties":   [it["difficulty"] for it in items],
        }
    return collate_fn


# ─────────────────────────────────────────────────────────────
# Async Disk Writer (thread-based, proven pattern)
# ─────────────────────────────────────────────────────────────
_WRITER_SENTINEL = None

class AsyncWriter:
    def __init__(self, output_dir: str, queue_maxsize: int = 512):
        self.output_dir = output_dir
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._errors: List[Exception] = []
        self._thread = threading.Thread(
            target=self._worker, name=f"disk-writer-rank{RANK}", daemon=True
        )
        self._thread.start()

    def put(self, record: Dict):
        self._q.put(record)

    def _worker(self):
        while True:
            record = self._q.get()
            if record is _WRITER_SENTINEL:
                break
            try:
                # Write .txt prompt file
                target_dir = os.path.join(
                    self.output_dir,
                    record["difficulty"],
                    record["gender"],
                )
                os.makedirs(target_dir, exist_ok=True)
                txt_path = os.path.join(target_dir, f"{record['file_name']}.txt")
                with open(txt_path, "w", encoding="utf-8") as f_txt:
                    f_txt.write(record["edit_prompt"])
            except Exception as e:
                self._errors.append(e)
                log(f"[AsyncWriter] ERROR writing {record.get('file_name')}: {e}")
            finally:
                self._q.task_done()

    def close(self):
        self._q.join()
        self._q.put(_WRITER_SENTINEL)
        self._thread.join(timeout=60)
        if self._errors:
            log(f"[AsyncWriter] {len(self._errors)} write error(s) occurred.")
        return len(self._errors) == 0


# ─────────────────────────────────────────────────────────────
# Model loader with retry (handles transient CUDA errors)
# ─────────────────────────────────────────────────────────────
import gc

def load_model(model_name: str, max_retries: int = 3):
    device = torch.device("cuda:0")

    for attempt in range(1, max_retries + 1):
        log(f"Loading model (attempt {attempt}/{max_retries}) | FA2 | bfloat16")
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=None,
                trust_remote_code=True,
            )
            model.to(device)
            model.eval()
            log("Model loaded ✓ (FA2)")
            return model

        except torch.cuda.OutOfMemoryError:
            log("FATAL: CUDA OOM during model load — not retryable.")
            raise

        except Exception as e:
            log(f"Model load attempt {attempt} failed: {e}")
            if attempt < max_retries:
                log(f"Cleaning up and retrying in 30s ...")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(30)
            else:
                log("All retry attempts exhausted.")
                raise


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # Enable faulthandler: prints Python traceback on SIGSEGV before crash
    faulthandler.enable(all_threads=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--humans_root", type=str, required=True,
                        help="Path to humans/ directory")
    parser.add_argument("--jsonl_dir", type=str, required=True,
                        help="Path to source JSONL/JSON directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to save the .txt prompts")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="VLM batch size per GPU")
    parser.add_argument("--start_batch", type=int, default=0,
                        help="Explicitly skip this many batches instead of checking the filesystem")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for generation")
    parser.add_argument("--test_limit", type=int, default=-1,
                        help="Total samples to process for testing")
    args = parser.parse_args()

    import socket
    JOB_START = datetime.now()

    log(f"host={socket.gethostname()} | RANK={RANK} LOCAL_RANK={LOCAL_RANK} "
        f"WORLD_SIZE={WORLD_SIZE} | start={JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")

    if IS_MAIN:
        print(f"\n{'='*60}", flush=True)
        print(f"Edit Prompt Generator  |  {WORLD_SIZE} workers", flush=True)
        print(f"Batch size : {args.batch_size} | Max new tokens: {args.max_new_tokens} | "
              f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}", flush=True)
        print(f"Output dir : {args.output_dir}", flush=True)
        print(f"Job start  : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'='*60}\n", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate local limit for testing
    local_limit = -1
    if args.test_limit > 0:
        local_limit = args.test_limit // WORLD_SIZE
        if RANK < (args.test_limit % WORLD_SIZE):
            local_limit += 1

    # ── Processor ────────────────────────────────────────────
    log(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    log("Processor ready.")

    # ── Dataset ──────────────────────────────────────────────
    log("Loading keyword datasets ...")
    dataset = EditPromptDataset(
        args.humans_root, args.jsonl_dir, args.output_dir,
        rank=RANK, world_size=WORLD_SIZE, batch_size=args.batch_size, start_batch=args.start_batch
    )

    # Truncate dataset to local_limit so DataLoader has exact count
    if local_limit > 0 and len(dataset) > local_limit:
        dataset.samples = dataset.samples[:local_limit]
        log(f"Test mode: truncated dataset to {local_limit} samples")

    log(f"Dataset: {len(dataset)} entries | "
        f"batch_size={args.batch_size} | "
        f"batches={-(-len(dataset) // args.batch_size)} (ceil)")

    # ── DataLoader ───────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(processor),
        num_workers=8,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True,
    )
    log(f"DataLoader ready: {len(loader)} batches.")

    # ── Model ────────────────────────────────────────────────
    model = load_model(args.model_name)

    primary_device = torch.device("cuda:0")
    log(f"Primary device for inputs: {primary_device}")
    log("Starting inference — each rank independent, no barrier.")

    # ── Inference loop ───────────────────────────────────────
    total_start        = time.perf_counter()
    batch_latencies: List[float] = []
    total_samples_done = 0

    writer = AsyncWriter(
        output_dir=args.output_dir,
        queue_maxsize=args.batch_size * 8,
    )
    log(f"Async disk writer started (queue maxsize={args.batch_size * 8}).")

    batch_errors = 0
    for batch_idx, batch in enumerate(loader):
        if local_limit > 0 and total_samples_done >= local_limit:
            log(f"Reached local test limit of {local_limit}. Stopping.")
            break

        bsz = batch["input_ids"].shape[0]

        try:
            ids  = batch["input_ids"].to(primary_device)
            mask = batch["attention_mask"].to(primary_device)
            pix  = batch["pixel_values"].to(primary_device)
            grid = batch["image_grid_thw"].to(primary_device)

            torch.cuda.synchronize(primary_device)
            t0 = time.perf_counter()

            with torch.inference_mode():
                gen_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    pixel_values=pix,
                    image_grid_thw=grid,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                )

            torch.cuda.synchronize(primary_device)
            batch_time = time.perf_counter() - t0
            per_sample = batch_time / bsz
            batch_latencies.append(batch_time)
            total_samples_done += bsz

            # Decode — strip prompt tokens (only decode newly generated tokens)
            input_len = ids.shape[1]
            new_ids   = gen_ids[:, input_len:]
            texts     = processor.tokenizer.batch_decode(
                new_ids, skip_special_tokens=True
            )

            # Calculate true batch index considering skipped ones from resume
            true_idx = batch_idx + 1 + dataset.skipped_batches
            true_total = len(loader) + dataset.skipped_batches
            
            log(f"Batch {true_idx}/{true_total} | "
                f"samples={bsz} | "
                f"batch={batch_time:.2f}s | "
                f"per_sample={per_sample:.3f}s | "
                f"queue={writer._q.qsize()}")

            for j, text in enumerate(texts):
                record = {
                    "file_name":    batch["file_names"][j],
                    "gender":       batch["genders"][j],
                    "difficulty":   batch["difficulties"][j],
                    "edit_prompt":  text.strip(),
                    "gpu_rank":     RANK,
                    "batch_idx":    batch_idx,
                    "batch_time_s": round(batch_time, 4),
                    "per_sample_s": round(per_sample, 4),
                }
                writer.put(record)

        except torch.cuda.OutOfMemoryError:
            batch_errors += 1
            log(f"OOM on batch {batch_idx} (bsz={bsz}) — clearing VRAM and continuing")
            del ids, mask, pix, grid
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            batch_errors += 1
            log(f"ERROR in batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    # ── Flush writer ─────────────────────────────────────────
    log("Inference done — flushing writer queue ...")
    ok = writer.close()
    log(f"Writer finished. {'All files written OK.' if ok else 'Some write errors!'}")

    # ── Per-rank summary ─────────────────────────────────────
    total_time = time.perf_counter() - total_start
    avg_batch  = sum(batch_latencies) / len(batch_latencies) if batch_latencies else 0
    throughput = total_samples_done / total_time if total_time > 0 else 0
    JOB_END    = datetime.now()

    log(f"\n{'─'*50}")
    log(f"SUMMARY  (Rank {RANK} / GPU {LOCAL_RANK})")
    log(f"{'─'*50}")
    log(f"  Image size             : {IMAGE_SIZE}×{IMAGE_SIZE}")
    log(f"  Prompts generated      : {total_samples_done}")
    log(f"  Batches OK             : {len(batch_latencies)}")
    log(f"  Batches failed         : {batch_errors}")
    log(f"  Job start              : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Job end                : {JOB_END.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total wall time        : {total_time:.2f}s")
    log(f"  Avg batch latency      : {avg_batch:.3f}s")
    if total_samples_done > 0:
        log(f"  Avg per-sample latency : {total_time/total_samples_done:.4f}s")
    log(f"  Throughput             : {throughput:.3f} prompts/sec")
    log(f"  Output dir             : {args.output_dir}")
    log(f"{'─'*50}\n")

    # Force clear VRAM before process exit
    torch.cuda.empty_cache()
    log("Prompt Generation Stage Complete.")


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
