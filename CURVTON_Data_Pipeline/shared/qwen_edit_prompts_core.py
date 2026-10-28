"""
Qwen3-VL-32B Edit Prompt Generator
====================================
For each entry in the keyword JSON files (medium/hard  female/male),
feed the VLM the paired person image + keyword description and ask it to
write a precise image-editing instruction.

Parallel inference across N GPUs using srun (RANK/LOCAL_RANK from SLURM env).
Each GPU rank runs fully independently — no torch.distributed needed.

Output per entry:
    PATH_TO_OUTPUT_DIR/{file_name}.txt
    PATH_TO_OUTPUT_DIR/edit_prompts_rank{N}.jsonl
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback as tb
from datetime import datetime
from typing import List, Dict

# Unbuffered output — critical for SLURM log visibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────
# Rank info — works with srun (SLURM env vars)
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
    from PIL import Image
except ImportError as e:
    print(f"[RANK {RANK}] CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Image size — controls how many vision tokens are generated.
# 512×512 keeps token count low; increase if you need finer detail
# and have VRAM headroom.
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE = 512

# ─────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "CURVTON_assets", "system_prompts", "medium_hard_person_edit_prompt.txt"), encoding="utf-8").read()

def build_user_prompt(entry: Dict) -> str:
    # Exclude internal metadata from the keyword set sent to the VLM
    exclude = {"file_name", "serial_number", "gender", "source_file"}
    keywords = {k: v for k, v in entry.items() if k not in exclude}
    return json.dumps(keywords, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# Dataset — keyword entries sharded by rank
# ─────────────────────────────────────────────────────────────
class KeywordDataset(Dataset):
    def __init__(
        self,
        json_dir: str,
        image_dir: str,
        rank: int = 0,
        world_size: int = 1,
        max_samples: int = 3000,
    ):
        self.image_dir = image_dir

        json_files = [
            ("medium_female_sampler_output.json", "female"),
            ("medium_male_sampler_output.json",   "male"),
            ("hard_female_sampler_output.json",   "female"),
            ("hard_male_sampler_output.json",     "male"),
        ]

        all_entries = []
        for fname, gender in json_files:
            fpath = os.path.join(json_dir, fname)
            if not os.path.exists(fpath):
                if IS_MAIN:
                    log(f"WARNING: JSON not found: {fpath} — skipping")
                continue
            with open(fpath, encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                all_entries.append({
                    **entry,
                    "gender":      gender,
                    "source_file": fname,
                })
            if IS_MAIN:
                log(f"  Loaded {len(entries)} entries from {fname}")

        if len(all_entries) > max_samples:
            all_entries = all_entries[:max_samples]

        # Shard by rank (interleaved so load is balanced across files)
        self.my_entries = [e for i, e in enumerate(all_entries) if i % world_size == rank]

        if IS_MAIN:
            log(f"Total entries (capped at {max_samples}): {len(all_entries)} | "
                f"Per-GPU shard: {len(self.my_entries)}")

    def __len__(self):
        return len(self.my_entries)

    def __getitem__(self, i):
        entry    = self.my_entries[i]
        img_path = os.path.join(self.image_dir, f"{entry['gender']}.png")

        # Resize to IMAGE_SIZE × IMAGE_SIZE to control vision token count.
        # Without this the processor tiles the full-res image into many patches,
        # inflating KV cache and causing OOM during generate().
        image = Image.open(img_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
        )

        return {
            "image":       image,
            "entry":       entry,
            "user_prompt": build_user_prompt(entry),
        }


# ─────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────
def make_collate_fn(processor):
    pad_id = processor.tokenizer.pad_token_id or 0

    def collate_fn(items: List[Dict]) -> Dict:
        encoded = []
        for it in items:
            msgs = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": it["image"]},
                        {"type": "text",  "text":  it["user_prompt"]},
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
            "entries":        [it["entry"]       for it in items],
            "user_prompts":   [it["user_prompt"] for it in items],
        }
    return collate_fn


# ─────────────────────────────────────────────────────────────
# Async Disk Writer
# ─────────────────────────────────────────────────────────────
_WRITER_SENTINEL = None

class AsyncWriter:
    def __init__(self, output_dir: str, jsonl_path: str, queue_maxsize: int = 512):
        self.output_dir = output_dir
        self.jsonl_path = jsonl_path
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._errors: List[Exception] = []
        self._thread = threading.Thread(
            target=self._worker, name=f"disk-writer-rank{RANK}", daemon=True
        )
        self._thread.start()

    def put(self, record: Dict):
        self._q.put(record)

    def _worker(self):
        with open(self.jsonl_path, "w", encoding="utf-8", buffering=1) as f_jsonl:
            while True:
                record = self._q.get()
                if record is _WRITER_SENTINEL:
                    break
                try:
                    txt_path = os.path.join(self.output_dir, f"{record['file_name']}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f_txt:
                        f_txt.write(record["edit_prompt"])
                    f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
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
# Load model
# torch_dtype="auto"  → uses checkpoint's native dtype (~63GB on disk stays ~63GB in VRAM)
# device_map="auto"   → spreads layers across all GPUs visible to this process.
#                       SLURM sets CUDA_VISIBLE_DEVICES per rank, so each rank
#                       only sees its own assigned GPU(s) — no cross-rank sharing.
# ─────────────────────────────────────────────────────────────
def load_model(model_name: str):
    log("Loading model | torch_dtype=auto | device_map=auto | flash_attention_2")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        if hasattr(model, "hf_device_map"):
            devices = set(str(v) for v in model.hf_device_map.values())
            log(f"Model loaded ✓  |  devices: {devices}")
        else:
            log("Model loaded ✓")
        return model
    except torch.cuda.OutOfMemoryError:
        log("FATAL: CUDA OOM during model load.")
        raise
    except Exception as e:
        log(f"FATAL: Model load failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir",       type=str,
                        default="PATH_TO_JSON_DIR")
    parser.add_argument("--image_dir",      type=str,
                        default="PATH_TO_IMAGE_DIR")
    parser.add_argument("--output_dir",     type=str,
                        default="PATH_TO_OUTPUT_DIR")
    parser.add_argument("--max_samples",    type=int, default=3000)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_size",     type=int, default=512,
                        help="Resize input images to this square size before processing. "
                             "Smaller = fewer vision tokens = less VRAM during inference.")
    parser.add_argument("--model_name",     type=str,
                        default="Qwen/Qwen3-VL-32B-Instruct")
    args = parser.parse_args()

    # Override module-level IMAGE_SIZE if passed via CLI
    global IMAGE_SIZE
    IMAGE_SIZE = args.image_size

    import socket
    JOB_START = datetime.now()

    log(f"host={socket.gethostname()} | RANK={RANK} LOCAL_RANK={LOCAL_RANK} "
        f"WORLD_SIZE={WORLD_SIZE} | start={JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")

    if IS_MAIN:
        print(f"\n{'='*60}", flush=True)
        print(f"Edit Prompt Generator  |  {WORLD_SIZE} workers", flush=True)
        print(f"Batch size : {args.batch_size} | Max new tokens: {args.max_new_tokens} | "
              f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}", flush=True)
        print(f"JSON dir   : {args.json_dir}", flush=True)
        print(f"Image dir  : {args.image_dir}", flush=True)
        print(f"Output dir : {args.output_dir}", flush=True)
        print(f"Job start  : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'='*60}\n", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Processor ────────────────────────────────────────────
    log(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    log("Processor ready.")

    # ── Dataset ──────────────────────────────────────────────
    log("Loading keyword datasets ...")
    dataset = KeywordDataset(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        rank=RANK,
        world_size=WORLD_SIZE,
        max_samples=args.max_samples,
    )
    log(f"Dataset: {len(dataset)} entries | "
        f"batch_size={args.batch_size} | "
        f"batches={-(-len(dataset) // args.batch_size)} (ceil)")

    # ── DataLoader ───────────────────────────────────────────
    # pin_memory=False because device_map=auto manages its own placement
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(processor),
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    log(f"DataLoader ready: {len(loader)} batches.")

    # ── Model ────────────────────────────────────────────────
    model = load_model(args.model_name)

    # Determine the primary device for input tensors.
    # With device_map=auto the embedding layer (first layer) is on the first device —
    # inputs must be sent there.
    if hasattr(model, "hf_device_map"):
        non_cpu = [str(v) for v in model.hf_device_map.values() if str(v) != "cpu"]
        primary_device = torch.device(non_cpu[0]) if non_cpu else torch.device("cuda:0")
    else:
        primary_device = torch.device("cuda:0")

    log(f"Primary device for inputs: {primary_device}")
    log("Starting inference — each rank independent, no barrier.")

    # ── Inference loop ───────────────────────────────────────
    total_start        = time.perf_counter()
    batch_latencies: List[float] = []
    total_samples_done = 0
    jsonl_path         = os.path.join(args.output_dir, f"edit_prompts_rank{RANK}.jsonl")

    log(f"JSONL output: {jsonl_path}")

    writer = AsyncWriter(
        output_dir=args.output_dir,
        jsonl_path=jsonl_path,
        queue_maxsize=args.batch_size * 8,
    )
    log(f"Async disk writer started (queue maxsize={args.batch_size * 8}).")

    for batch_idx, batch in enumerate(loader):
        bsz = batch["input_ids"].shape[0]

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

        # Decode — strip prompt tokens
        input_len = ids.shape[1]
        new_ids   = gen_ids[:, input_len:]
        texts     = processor.tokenizer.batch_decode(
            new_ids, skip_special_tokens=True
        )

        log(f"Batch {batch_idx+1}/{len(loader)} | "
            f"samples={bsz} | "
            f"batch={batch_time:.2f}s | "
            f"per_sample={per_sample:.3f}s | "
            f"queue={writer._q.qsize()}")

        for entry, text in zip(batch["entries"], texts):
            record = entry.copy()
            record.update({
                "edit_prompt":   text.strip(),
                "gpu_rank":      RANK,
                "batch_idx":     batch_idx,
                "batch_time_s":  round(batch_time, 4),
                "per_sample_s":  round(per_sample, 4),
            })
            writer.put(record)

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
    log(f"  Batches                : {len(batch_latencies)}")
    log(f"  Job start              : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Job end                : {JOB_END.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total wall time        : {total_time:.2f}s")
    log(f"  Avg batch latency      : {avg_batch:.3f}s")
    log(f"  Avg per-sample latency : {total_time/total_samples_done:.4f}s")
    log(f"  Throughput             : {throughput:.3f} prompts/sec")
    log(f"  JSONL output           : {jsonl_path}")
    log(f"{'─'*50}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL EXCEPTION:")
        tb.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
