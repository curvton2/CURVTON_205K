"""
Try-On Image Quality Evaluator — Qwen3-VL-32B-Instruct
======================================================
Reads generated try-on images from:
    dataset_v3_dir/dataset_ultimate/{diff}/{gender}/tryon_image/

For each unevaluated tryon image (no _correct/_incorrect suffix):
  - Loads: initial_person_image/{person}.png  +  cloth_image/{tryon_fname}
  - Sends all 3 images to Qwen3-VL with garment-populated system prompt

After evaluation:
  CORRECT  → tryon stays in dataset_ultimate (original name, no suffix) with a
             sidecar {name}.json. Person + cloth stay in place.
  INCORRECT → tryon + cloth moved (person copied) to
             dataset_ultimate_1/{diff}/{gender}/{tryon_image,initial_person_image,cloth_image}/
             using the original filename (no suffix). Eval JSON (with improved_prompt)
             written alongside tryon.

In job_tryon_loop.sh, rename.py then moves the correct triplets to dataset_end and
re-queues the incorrect ones (with their improved_prompt) for the next iteration.
"""

import argparse
import faulthandler
import gc
import json
import os
import queue
import shutil
import signal
import sys
import threading
import random
import time
import traceback as tb
from datetime import datetime
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
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError as e:
    print(f"[RANK {RANK}] CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

IMAGE_SIZE = 512

from dress_info import DRESS_INFO

def load_system_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


class TryonEvaluatorDataset(Dataset):
    """
    Scans dataset_v3_dir/dataset_ultimate/{diff}/{gender}/tryon_image/ for
    unevaluated .png files (no _correct / _incorrect suffix), then loads
    the matching person and cloth images from the same ultimate directory.
    """

    def __init__(self, dataset_v3_dir: str, rank: int = 0, world_size: int = 1,
                 category_limit: int = -1):
        self.dataset_v3_dir = dataset_v3_dir
        self.category_limit = category_limit

        all_samples: List[Dict] = []
        ultimate_dir = os.path.join(dataset_v3_dir, "dataset_ultimate")

        if not os.path.exists(ultimate_dir):
            log(f"WARNING: dataset_ultimate not found: {ultimate_dir}")
            self.samples = []
            return

        for diff in ["easy", "medium", "hard"]:
            for gender in ["male", "female"]:
                tryon_dir  = os.path.join(ultimate_dir, diff, gender, "tryon_image")
                person_dir = os.path.join(ultimate_dir, diff, gender, "initial_person_image")
                cloth_dir  = os.path.join(ultimate_dir, diff, gender, "cloth_image")

                if not os.path.exists(tryon_dir):
                    continue
                if not os.path.exists(person_dir) or not os.path.exists(cloth_dir):
                    log(f"WARNING: Missing person or cloth dir for {diff}/{gender}, skipping.")
                    continue

                cloth_files  = {f for f in os.listdir(cloth_dir)  if f.endswith(".png")}
                person_files = {f for f in os.listdir(person_dir) if f.endswith(".png")}

                # Only unevaluated files: skip if a sidecar .json already exists
                # (correct tryons keep their original .png name but get a name.json written)
                tryon_files = sorted([
                    f for f in os.listdir(tryon_dir)
                    if f.endswith(".png")
                    and "_correct" not in f
                    and "_incorrect" not in f
                    and not os.path.exists(
                        os.path.join(tryon_dir, f.replace(".png", ".json"))
                    )
                ])

                category_count = 0
                for tryon_fname in tryon_files:
                    # Filename pattern: mh_000001_e01_mc_000001_garment_name.png
                    # Person base = first 3 underscore-joined tokens: mh_000001_e01
                    parts = tryon_fname.replace(".png", "").split("_")
                    if len(parts) < 3:
                        continue

                    person_base          = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    expected_person_file = f"{person_base}.png"

                    # Cloth file has same name as tryon file
                    if expected_person_file not in person_files:
                        log(f"Person not found: {expected_person_file} (tryon {tryon_fname})")
                        continue
                    if tryon_fname not in cloth_files:
                        log(f"Cloth not found: {tryon_fname}")
                        continue

                    # Garment matching: longest endswith on DRESS_INFO name (spaces→underscores)
                    cloth_no_ext    = tryon_fname.replace(".png", "").lower()
                    best_match_len  = -1
                    valid_garments: List[Dict] = []
                    for d in DRESS_INFO:
                        d_name_underscored = d["name"].replace(" ", "_").lower()
                        if cloth_no_ext.endswith(d_name_underscored):
                            if len(d_name_underscored) > best_match_len:
                                best_match_len = len(d_name_underscored)
                                valid_garments = [d]
                    if not valid_garments:
                        valid_garments = DRESS_INFO
                    selected_garment = random.choice(valid_garments)

                    all_samples.append({
                        "tryon_path":   os.path.join(tryon_dir,  tryon_fname),
                        "person_path":  os.path.join(person_dir, expected_person_file),
                        "cloth_path":   os.path.join(cloth_dir,  tryon_fname),
                        "tryon_fname":  tryon_fname,
                        "person_fname": expected_person_file,
                        "cloth_fname":  tryon_fname,
                        "difficulty":   diff,
                        "gender":       gender,
                        "garment":      selected_garment,
                        "tryon_dir":    tryon_dir,
                        "person_dir":   person_dir,
                        "cloth_dir":    cloth_dir,
                    })

                    category_count += 1
                    if self.category_limit > 0 and category_count >= self.category_limit:
                        break

        my_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]
        self.samples = my_samples
        log(f"Dataset: Total={len(all_samples)} | Shard(Rank {rank})={len(my_samples)} items.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        tryon_img  = Image.open(s["tryon_path"]).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        person_img = Image.open(s["person_path"]).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        cloth_img  = Image.open(s["cloth_path"]).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return {
            "tryon_image":  tryon_img,
            "person_image": person_img,
            "cloth_image":  cloth_img,
            "tryon_path":   s["tryon_path"],
            "person_path":  s["person_path"],
            "cloth_path":   s["cloth_path"],
            "tryon_fname":  s["tryon_fname"],
            "person_fname": s["person_fname"],
            "cloth_fname":  s["cloth_fname"],
            "difficulty":   s["difficulty"],
            "gender":       s["gender"],
            "garment":      s["garment"],
            "tryon_dir":    s["tryon_dir"],
            "person_dir":   s["person_dir"],
            "cloth_dir":    s["cloth_dir"],
        }


def make_collate_fn(processor, system_prompt: str):
    pad_id = processor.tokenizer.pad_token_id or 0

    def collate_fn(items: List[Dict]) -> Dict:
        encoded = []
        for it in items:
            garment = it["garment"]
            garment_str = (
                f'{{ name: "{garment.get("name", "")}", '
                f'dress_type: "{garment.get("dress_type", "")}", '
                f'description: "{garment.get("description", "")}" }}'
            )
            populated_prompt = system_prompt.replace(
                "{ name, dress_type, description }", garment_str
            )
            msgs = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": populated_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": "Image A (Person Ground Truth):"},
                        {"type": "image", "image": it["person_image"]},
                        {"type": "text",  "text": "\nImage B (Garment Ground Truth):"},
                        {"type": "image", "image": it["cloth_image"]},
                        {"type": "text",  "text": "\nImage C (Flux Try-On Result):"},
                        {"type": "image", "image": it["tryon_image"]},
                        {"type": "text",  "text": f"\nGarment metadata: {garment_str}"},
                        {"type": "text",  "text": "\nEvaluate this try-on result against the garment ground truth and identity ground truth."},
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
            # pass-through metadata as lists
            "tryon_paths":    [it["tryon_path"]   for it in items],
            "person_paths":   [it["person_path"]  for it in items],
            "cloth_paths":    [it["cloth_path"]   for it in items],
            "tryon_fnames":   [it["tryon_fname"]  for it in items],
            "person_fnames":  [it["person_fname"] for it in items],
            "cloth_fnames":   [it["cloth_fname"]  for it in items],
            "difficulties":   [it["difficulty"]   for it in items],
            "genders":        [it["gender"]       for it in items],
            "tryon_dirs":     [it["tryon_dir"]    for it in items],
            "person_dirs":    [it["person_dir"]   for it in items],
            "cloth_dirs":     [it["cloth_dir"]    for it in items],
        }
    return collate_fn


def parse_model_output(output: str) -> Dict:
    """Extract JSON block from model output."""
    try:
        start = output.find("{")
        end   = output.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(output[start:end + 1])
            data["score_value"] = float(data.get("score", 0.0))
            return data
    except Exception:
        pass
    return {"result": "ERROR", "score_value": 0.0, "raw_output": output}


# ---------------------------------------------------------------------------
# Async writer — handles rename, JSON persistence, and incorrect-set moves
# ---------------------------------------------------------------------------

_WRITER_SENTINEL = None


class AsyncEvaluatorWriter:
    """
    Background thread that performs all disk I/O after evaluation:

    CORRECT  — tryon file kept as-is in dataset_ultimate; person + cloth untouched.
    INCORRECT — tryon, person, cloth moved to dataset_ultimate_1 mirror dirs;
                eval JSON written to dataset_ultimate_1/.../tryon_image/name.json.
    """

    def __init__(self, dataset_ultimate_1_dir: str, queue_maxsize: int = 512):
        self.dataset_ultimate_1_dir = dataset_ultimate_1_dir
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._errors: List[Exception] = []
        self._thread = threading.Thread(
            target=self._worker, name=f"eval-writer-rank{RANK}", daemon=True
        )
        self._thread.start()

    def put(self, payload: Dict):
        """Enqueue a write payload (see _worker for schema)."""
        self._q.put(payload)

    def _worker(self):
        while True:
            item = self._q.get()
            if item is _WRITER_SENTINEL:
                break
            try:
                self._process(item)
            except Exception as e:
                self._errors.append(e)
                log(f"[AsyncWriter] ERROR: {e}")
            finally:
                self._q.task_done()

    def _process(self, p: Dict):
        old_tryon    = p["old_tryon_path"]  # original .png in dataset_ultimate/.../tryon_image/
        tryon_fname  = p["tryon_fname"]      # e.g. mh_000001_e01_mc_000001_daura_suruwal.png
        is_incorrect = p["is_incorrect"]

        if not is_incorrect:
            # Correct: keep file in place; write sidecar JSON so re-runs skip it.
            json_fname = tryon_fname.replace(".png", ".json")
            json_path  = os.path.join(os.path.dirname(old_tryon), json_fname)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(p["json_data"], jf, indent=2)
            return

        # Incorrect: move tryon + person + cloth to dataset_ultimate_1 mirror, keep original names.
        diff   = p["diff"]
        gender = p["gender"]
        u1_base = os.path.join(self.dataset_ultimate_1_dir, diff, gender)

        u1_tryon_dir  = os.path.join(u1_base, "tryon_image")
        u1_person_dir = os.path.join(u1_base, "initial_person_image")
        u1_cloth_dir  = os.path.join(u1_base, "cloth_image")
        os.makedirs(u1_tryon_dir,  exist_ok=True)
        os.makedirs(u1_person_dir, exist_ok=True)
        os.makedirs(u1_cloth_dir,  exist_ok=True)

        # Move tryon + cloth (original names, no suffix). The person image is copied,
        # not moved: other cloth pairs still being evaluated may share the same person.
        shutil.move(old_tryon,       os.path.join(u1_tryon_dir,  tryon_fname))
        shutil.copy2(p["person_path"], os.path.join(u1_person_dir, p["person_fname"]))
        shutil.move(p["cloth_path"],  os.path.join(u1_cloth_dir,  p["cloth_fname"]))

        # Write eval JSON alongside tryon in dataset_ultimate_1
        json_fname = tryon_fname.replace(".png", ".json")
        with open(os.path.join(u1_tryon_dir, json_fname), "w", encoding="utf-8") as jf:
            json.dump(p["json_data"], jf, indent=2)

    def close(self) -> bool:
        self._q.join()
        self._q.put(_WRITER_SENTINEL)
        self._thread.join(timeout=60)
        return len(self._errors) == 0


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
                log("Cleaning up and retrying in 30s …")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(30)
            else:
                log("All retry attempts exhausted.")
                raise


def main():
    faulthandler.enable(all_threads=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_v3_dir", type=str, required=True,
                        help="Root of dataset_v3; must contain dataset_ultimate/")
    parser.add_argument("--dataset_ultimate_1_dir", type=str, required=True,
                        help="Mirror of dataset_ultimate that receives incorrect triplets")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to CURVTON_assets/system_prompts/tryon_feedback_prompt.txt")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Score threshold: >= threshold → correct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--category_limit", type=int, default=-1,
                        help="Per-category sample cap (for testing)")
    args = parser.parse_args()

    JOB_START = datetime.now()

    if IS_MAIN:
        print(f"\n{'='*60}", flush=True)
        print(f"Try-On Evaluator  |  {WORLD_SIZE} workers", flush=True)
        print(f"dataset_v3_dir        : {args.dataset_v3_dir}", flush=True)
        print(f"dataset_ultimate_1_dir: {args.dataset_ultimate_1_dir}", flush=True)
        print(f"Threshold             : {args.threshold}", flush=True)
        print(f"{'='*60}\n", flush=True)

    system_prompt = load_system_prompt(args.prompt_file)
    log("System prompt loaded.")

    log(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    log("Processor ready.")

    log("Scanning for unevaluated tryon images …")
    dataset = TryonEvaluatorDataset(
        dataset_v3_dir=args.dataset_v3_dir,
        rank=RANK,
        world_size=WORLD_SIZE,
        category_limit=args.category_limit,
    )

    if len(dataset) == 0:
        log("Nothing to evaluate.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(processor, system_prompt),
        num_workers=16,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = load_model(args.model_name)
    primary_device = torch.device("cuda:0")

    total_start      = time.perf_counter()
    total_evaluated  = 0
    correct_count    = 0
    incorrect_count  = 0
    batch_errors     = 0

    writer = AsyncEvaluatorWriter(
        dataset_ultimate_1_dir=args.dataset_ultimate_1_dir,
        queue_maxsize=512,
    )

    for batch_idx, batch in enumerate(loader):
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

            input_len = ids.shape[1]
            texts = processor.tokenizer.batch_decode(
                gen_ids[:, input_len:], skip_special_tokens=True
            )

            for j, text in enumerate(texts):
                eval_data = parse_model_output(text.strip())
                score     = eval_data.get("score_value", 0.0)
                result    = eval_data.get("result", "ERROR")

                # REJECT counts as incorrect
                is_correct = result == "SUCCESS" and score >= args.threshold

                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

                eval_data["original_file"] = batch["tryon_fnames"][j]
                eval_data["label"]         = "correct" if is_correct else "incorrect"
                eval_data["gpu_rank"]      = RANK

                payload = {
                    "old_tryon_path": batch["tryon_paths"][j],
                    "tryon_fname":    batch["tryon_fnames"][j],
                    "json_data":      eval_data,
                    "is_incorrect":   not is_correct,
                    "diff":           batch["difficulties"][j],
                    "gender":         batch["genders"][j],
                    "person_path":    batch["person_paths"][j],
                    "cloth_path":     batch["cloth_paths"][j],
                    "person_fname":   batch["person_fnames"][j],
                    "cloth_fname":    batch["cloth_fnames"][j],
                }
                writer.put(payload)

                per_sample_time = batch_time / bsz if bsz > 0 else 0.0
                log(
                    f"Batch {batch_idx+1}/{len(loader)} | "
                    f"{batch['tryon_fnames'][j]} | result={result} score={score:.1f} | "
                    f"label={'correct' if is_correct else 'incorrect'} | "
                    f"batch={batch_time:.2f}s per_sample={per_sample_time:.2f}s"
                )

            total_evaluated += bsz

        except torch.cuda.OutOfMemoryError:
            batch_errors += 1
            log(f"OOM on batch {batch_idx} (bsz={bsz}) — clearing VRAM and continuing")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            batch_errors += 1
            log(f"ERROR in batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            gc.collect()

    writer.close()

    total_time = time.perf_counter() - total_start
    throughput = total_evaluated / total_time if total_time > 0 else 0
    JOB_END = datetime.now()

    log(f"\n{'─'*50}")
    log(f"SUMMARY  (Rank {RANK} / GPU {LOCAL_RANK})")
    log(f"{'─'*50}")
    log(f"  Images evaluated : {total_evaluated}")
    log(f"  Correct          : {correct_count}")
    log(f"  Incorrect        : {incorrect_count}")
    log(f"  Batches failed   : {batch_errors}")
    log(f"  Wall time        : {total_time:.2f}s")
    log(f"  Throughput       : {throughput:.3f} evals/sec")
    log(f"  Job start        : {JOB_START}")
    log(f"  Job end          : {JOB_END}")
    log(f"{'─'*50}\n")

    torch.cuda.empty_cache()
    log("Evaluation Stage Complete.")


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
