"""
Edit Image Quality Evaluator  — Qwen3-VL-32B-Instruct
======================================================
Evaluates generated edited images using the VLM as a quality judge.
Each image is scored; images above --threshold are renamed *_correct,
below are renamed *_incorrect with a companion .json metadata file.

Parallel inference across N GPUs using srun (RANK / LOCAL_RANK from
SLURM env).  Each GPU runs fully independently — no torch.distributed.

Uses the proven model-loading and inference patterns from
qwen_edit_prompts.py (device_map="auto", torch_dtype="auto",
flash_attention_2, pre-tokenized collation, torch.inference_mode,
strip-prompt decoding).
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

IMAGE_SIZE = 512


# ─────────────────────────────────────────────────────────────
# System prompt loader
# ─────────────────────────────────────────────────────────────
def load_system_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


# ─────────────────────────────────────────────────────────────
# Dataset — finds edited .png images not yet evaluated
# ─────────────────────────────────────────────────────────────
class EditedImagesDataset(Dataset):
    def __init__(self, root_dir, rank=0, world_size=1, batch_size=1):
        self.root_dir = root_dir
        all_samples = []

        edited_root = os.path.join(root_dir, "dataset_ultimate_13")

        for diff in ["easy"]:
            for gender in ["male", "female"]:
                dir_path = os.path.join(edited_root, diff, gender, "initial_person_image")
                if not os.path.exists(dir_path):
                    continue

                # Pre-scan directory ONCE to build lookup sets (avoids per-file stat calls)
                all_files = set(os.listdir(dir_path))
                files = sorted([f for f in all_files if f.endswith(".png")])

                skipped = 0
                for fname in files:
                    # Already evaluated — renamed to _correct or _incorrect
                    if "_correct" in fname or "_incorrect" in fname:
                        continue

                    all_samples.append({
                        "path": os.path.join(dir_path, fname),
                        "fname": fname,
                        "difficulty": diff,
                        "gender": gender,
                        "already_done": False # Resume disabled
                    })

        # Shard across ranks from the COMPLETE list
        my_full_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]

        # Explicit override: no resume logic at all
        self.samples = my_full_samples
        self.skipped_batches = 0
        
        log(f"Dataset: Total={len(all_samples)} | Shard(Rank {rank})={len(my_full_samples)} items to process.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
        )
        return {
            "image": image,
            "path": s["path"],
            "fname": s["fname"],
            "difficulty": s["difficulty"],
            "gender": s["gender"],
        }


# ─────────────────────────────────────────────────────────────
# Collate — pre-tokenize (proven pattern from qwen_edit_prompts.py)
# Batch size is 1 for evaluation, but this pattern works for any size
# ─────────────────────────────────────────────────────────────
def make_collate_fn(processor, system_prompt: str):
    pad_id = processor.tokenizer.pad_token_id or 0

    def collate_fn(items: List[Dict]) -> Dict:
        encoded = []
        for it in items:
            msgs = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": it["image"]},
                        {"type": "text",  "text": "Evaluate this edited human image."},
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
            "paths":          [it["path"]       for it in items],
            "fnames":         [it["fname"]      for it in items],
            "difficulties":   [it["difficulty"] for it in items],
            "genders":        [it["gender"]     for it in items],
        }
    return collate_fn


# ─────────────────────────────────────────────────────────────
# Output parser
# ─────────────────────────────────────────────────────────────
def parse_model_output(output: str) -> Dict:
    lines = output.strip().split('\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            data[key.strip().lower().replace(' ', '_')] = val.strip()
    try:
        data["score_value"] = float(data.get("score", 0.0))
    except (ValueError, TypeError):
        data["score_value"] = 0.0
    return data


# ─────────────────────────────────────────────────────────────
# Async evaluator writer (thread-based, with proper shutdown)
# ─────────────────────────────────────────────────────────────
_WRITER_SENTINEL = None

class AsyncEvaluatorWriter:
    def __init__(self, queue_maxsize: int = 512):
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._errors: List[Exception] = []
        self._thread = threading.Thread(
            target=self._worker, name=f"eval-writer-rank{RANK}", daemon=True
        )
        self._thread.start()

    def put(self, old_path: str, new_path: str, meta_path: str, meta_data: Dict):
        self._q.put((old_path, new_path, meta_path, meta_data))

    def _worker(self):
        while True:
            item = self._q.get()
            if item is _WRITER_SENTINEL:
                break
            old_path, new_path, meta_path, meta_data = item
            try:
                os.rename(old_path, new_path)
                with open(meta_path, "w", encoding="utf-8") as jf:
                    json.dump(meta_data, jf, indent=2)
            except Exception as e:
                self._errors.append(e)
                log(f"[AsyncWriter] ERROR: {e}")
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
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root dataset directory")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to system prompt file")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for _correct vs _incorrect")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for evaluation response")
    parser.add_argument("--test_limit", type=int, default=-1,
                        help="Total samples to process for testing")
    args = parser.parse_args()

    import socket
    JOB_START = datetime.now()

    log(f"host={socket.gethostname()} | RANK={RANK} LOCAL_RANK={LOCAL_RANK} "
        f"WORLD_SIZE={WORLD_SIZE} | start={JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")

    if IS_MAIN:
        print(f"\n{'='*60}", flush=True)
        print(f"Edit Image Evaluator  |  {WORLD_SIZE} workers", flush=True)
        print(f"Threshold  : {args.threshold}", flush=True)
        print(f"Root dir   : {args.root_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)

    # Calculate local limit for testing
    local_limit = -1
    if args.test_limit > 0:
        local_limit = args.test_limit // WORLD_SIZE
        if RANK < (args.test_limit % WORLD_SIZE):
            local_limit += 1

    # ── System prompt ────────────────────────────────────────
    system_prompt = load_system_prompt(args.prompt_file)
    log("System prompt loaded.")

    # ── Processor ────────────────────────────────────────────
    log(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    log("Processor ready.")

    # ── Dataset ──────────────────────────────────────────────
    log("Scanning for edited images ...")
    dataset = EditedImagesDataset(args.root_dir, rank=RANK, world_size=WORLD_SIZE, batch_size=args.batch_size)

    # Truncate dataset to local_limit so DataLoader has exact count
    if local_limit > 0 and len(dataset) > local_limit:
        dataset.samples = dataset.samples[:local_limit]
        log(f"Test mode: truncated dataset to {local_limit} samples")

    log(f"Dataset: {len(dataset)} images | batch_size={args.batch_size}")

    if len(dataset) == 0:
        log("Nothing to evaluate.")
        return

    # ── DataLoader ───────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(processor, system_prompt),
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
    log("Starting evaluation — each rank independent, no barrier.")

    # ── Inference loop ───────────────────────────────────────
    total_start    = time.perf_counter()
    batch_latencies: List[float] = []
    total_evaluated = 0
    correct_count   = 0
    incorrect_count = 0

    writer = AsyncEvaluatorWriter(queue_maxsize=256)
    log("Async evaluator writer started.")

    batch_errors = 0
    for batch_idx, batch in enumerate(loader):
        if local_limit > 0 and total_evaluated >= local_limit:
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
            batch_latencies.append(batch_time)

            # Decode — strip prompt tokens
            input_len = ids.shape[1]
            new_ids   = gen_ids[:, input_len:]
            texts     = processor.tokenizer.batch_decode(
                new_ids, skip_special_tokens=True
            )

            # Process each evaluated item
            for j, text in enumerate(texts):
                clean_response = text.strip()
                eval_data = parse_model_output(clean_response)
                score = eval_data.get("score_value", 0.0)

                label = "_correct" if score >= args.threshold else "_incorrect"
                if score >= args.threshold:
                    correct_count += 1
                else:
                    incorrect_count += 1

                path = batch["paths"][j]
                fname = batch["fnames"][j]

                base, ext = os.path.splitext(path)
                new_path  = f"{base}{label}{ext}"
                meta_path = f"{base}{label}.json"

                eval_data["original_file"] = fname
                eval_data["label"] = label.strip("_")
                eval_data["gpu_rank"] = RANK

                writer.put(path, new_path, meta_path, eval_data)

            total_evaluated += bsz

            true_idx = batch_idx + 1 + dataset.skipped_batches
            true_total = len(loader) + dataset.skipped_batches
            
            log(f"Batch {true_idx}/{true_total} | "
                f"{batch['fnames'][0]} | score={eval_data.get('score_value', '?')} | "
                f"label={label} | latency={batch_time:.2f}s")

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
    log("Evaluation done — flushing writer queue ...")
    ok = writer.close()
    log(f"Writer finished. {'All files written OK.' if ok else 'Some write errors!'}")

    # ── Per-rank summary ─────────────────────────────────────
    total_time = time.perf_counter() - total_start
    throughput = total_evaluated / total_time if total_time > 0 else 0
    JOB_END    = datetime.now()

    log(f"\n{'─'*50}")
    log(f"SUMMARY  (Rank {RANK} / GPU {LOCAL_RANK})")
    log(f"{'─'*50}")
    log(f"  Images evaluated       : {total_evaluated}")
    log(f"  Correct                : {correct_count}")
    log(f"  Incorrect              : {incorrect_count}")
    log(f"  Batches failed         : {batch_errors}")
    log(f"  Job start              : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Job end                : {JOB_END.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total wall time        : {total_time:.2f}s")
    if batch_latencies:
        log(f"  Avg eval latency       : {sum(batch_latencies)/len(batch_latencies):.3f}s")
    log(f"  Throughput             : {throughput:.3f} evals/sec")
    log(f"{'─'*50}\n")

    torch.cuda.empty_cache()
    log("Evaluation Stage Complete.")


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
