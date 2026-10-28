"""
Qwen3-VL-32B Edit Output Complexity Scorer
===========================================
Reads edited PNG images from:
  <edit_outputs_dir>/hard/male/   &  <edit_outputs_dir>/hard/female/
  <edit_outputs_dir>/medium/male/ &  <edit_outputs_dir>/medium/female/

For each image the VLM scores three dimensions on a 0-100 scale:
  - pose_complexity   : how complex / difficult the body pose is
  - occlusion_complexity : how complex the occlusion arrangement is
  - implausibility    : how strange / unrealistic / outlier the image is

Scores are returned as strict JSON so they can be parsed deterministically.

After all GPU ranks finish, the main rank (0) aggregates all per-rank JSONL
files, averages the three scores per difficulty tier (hard / medium), and
writes a final summary JSON.

Parallel inference: 2 nodes × 4 GPUs = 8 total workers (srun).
Each GPU rank operates fully independently — no torch.distributed needed.
DataLoader: num_workers=8, batch_size=32, drop_last=False.
"""

import argparse
import glob
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback as tb
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Unbuffered output for SLURM log visibility ──────────────────────────────
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Rank info from SLURM env vars ───────────────────────────────────────────
RANK       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID",  "0")))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",  "1")))
IS_MAIN    = (RANK == 0)

IMAGE_SIZE = 512   # resize to this before feeding the VLM


def log(msg: str) -> None:
    print(f"[GPU {LOCAL_RANK} | RANK {RANK}] {msg}", flush=True)


# ── Lazy heavy imports (after rank is known) ────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from PIL import Image
except ImportError as e:
    print(f"[RANK {RANK}] CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — instructs the VLM to return strict JSON scores
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert visual evaluator specialising in human pose analysis, \
occlusion assessment, and image plausibility. You will be shown a single \
edited image of a person. Your task is to rate it across three dimensions \
using a 0-to-100 integer scale, then return ONLY a JSON object — no prose, \
no markdown, no explanation.

DIMENSION DEFINITIONS
─────────────────────
pose_complexity (0-100)
  How physically demanding and geometrically complex the body pose is.
  0  = trivial standing or sitting with no notable limb arrangement.
  50 = one or two challenging limb positions (e.g. single-leg balance, \
       deep bend, reaching overhead).
  100 = extreme whole-body complexity (e.g. martial-arts kick, gymnastics \
        split, unusual contortion) that requires substantial muscular effort.

occlusion_complexity (0-100)
  How much and how intricately objects or scene elements occlude the body.
  0  = no occlusion, the entire body is clearly visible.
  50 = moderate occlusion (e.g. a bag covering part of the torso, one arm \
       behind an object).
  100 = heavy multi-layered occlusion that leaves only fragments of the body \
        visible and makes body-part attribution very hard.

implausibility (0-100)
  How strange, unrealistic, or outlier the image is — does it violate \
  physics, anatomy, or common sense?
  0  = perfectly natural and realistic.
  50 = noticeable but tolerable weirdness (e.g. slightly odd proportions, \
       minor rendering artefact, mildly unnatural lighting).
  100 = severe implausibility (floating body parts, impossible physics, \
        grotesque distortion, AI artefact that breaks realism entirely).

OUTPUT FORMAT (strict — parseable by json.loads)
────────────────────────────────────────────────
{
  "pose_complexity": <integer 0-100>,
  "occlusion_complexity": <integer 0-100>,
  "implausibility": <integer 0-100>
}

Return ONLY the JSON object above. No additional text before or after it.
"""

USER_PROMPT = (
    "Please evaluate this image according to your instructions "
    "and return only the JSON object."
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — discovers all PNG images; shards by rank
# ─────────────────────────────────────────────────────────────────────────────
class EditOutputDataset(Dataset):
    """
    Discovers every .png inside:
        <root_dir>/<difficulty>/<gender>/
    where difficulty ∈ {hard, medium} and gender ∈ {male, female}.

    The full list is built once, then sharded across ranks so that:
        rank k  →  indices  k, k+WORLD_SIZE, k+2*WORLD_SIZE, …
    (interleaved sharding keeps per-rank counts balanced).
    """

    DIFFICULTIES = ("hard", "medium")
    GENDERS      = ("male", "female")

    def __init__(
        self,
        root_dir: str,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.root_dir = root_dir
        self.image_size = IMAGE_SIZE

        all_items: List[Dict] = []

        for diff in self.DIFFICULTIES:
            for gender in self.GENDERS:
                subdir = os.path.join(root_dir, diff, gender)
                if not os.path.isdir(subdir):
                    if IS_MAIN:
                        log(f"WARNING: directory not found — skipping: {subdir}")
                    continue

                pngs = sorted(glob.glob(os.path.join(subdir, "*.png")))
                if IS_MAIN:
                    log(f"  Found {len(pngs):>5} PNGs in {diff}/{gender}")

                for png_path in pngs:
                    all_items.append({
                        "image_path":  png_path,
                        "difficulty":  diff,
                        "gender":      gender,
                        "image_name":  os.path.splitext(os.path.basename(png_path))[0],
                    })

        if IS_MAIN:
            log(f"Total images discovered: {len(all_items)}")

        # Interleaved shard by rank
        self.items: List[Dict] = [
            item for i, item in enumerate(all_items)
            if i % world_size == rank
        ]

        if IS_MAIN:
            log(f"Per-GPU shard (rank 0): {len(self.items)} images")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        try:
            image = (
                Image.open(item["image_path"])
                .convert("RGB")
                .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            )
        except Exception as exc:
            log(f"WARNING: could not open {item['image_path']}: {exc}; using blank image.")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))

        return {
            "image":      image,
            "image_path": item["image_path"],
            "image_name": item["image_name"],
            "difficulty": item["difficulty"],
            "gender":     item["gender"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate — builds batched processor inputs; keeps metadata lists
# ─────────────────────────────────────────────────────────────────────────────
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
                        {"type": "text",  "text":  USER_PROMPT},
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

        def lpad(t: "torch.Tensor", fill: int) -> "torch.Tensor":
            gap = max_len - t.shape[-1]
            if gap == 0:
                return t
            return torch.cat(
                [torch.full((1, gap), fill, dtype=t.dtype), t], dim=-1
            )

        return {
            "input_ids":      torch.cat([lpad(e["input_ids"],      pad_id) for e in encoded]),
            "attention_mask": torch.cat([lpad(e["attention_mask"], 0)      for e in encoded]),
            "pixel_values":   torch.cat([e["pixel_values"]   for e in encoded]),
            "image_grid_thw": torch.cat([e["image_grid_thw"] for e in encoded]),
            # Metadata lists (not tensors)
            "image_paths": [it["image_path"] for it in items],
            "image_names": [it["image_name"] for it in items],
            "difficulties": [it["difficulty"] for it in items],
            "genders":      [it["gender"]     for it in items],
        }

    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing — extract the three scores robustly
# ─────────────────────────────────────────────────────────────────────────────
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

SCORE_KEYS = ("pose_complexity", "occlusion_complexity", "implausibility")


def parse_scores(raw_text: str) -> Tuple[Optional[Dict[str, int]], str]:
    """
    Try multiple strategies to extract the JSON scores from model output.

    Returns (parsed_dict_or_None, cleaned_raw_text).
    """
    text = raw_text.strip()

    # Strategy 1: direct parse (model output is clean JSON)
    try:
        obj = json.loads(text)
        if all(k in obj for k in SCORE_KEYS):
            return {k: int(obj[k]) for k in SCORE_KEYS}, text
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 2: find first {...} block via regex
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group())
            if all(k in obj for k in SCORE_KEYS):
                return {k: int(obj[k]) for k in SCORE_KEYS}, text
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: key-value regex extraction
    scores: Dict[str, int] = {}
    for key in SCORE_KEYS:
        pat = re.search(
            rf'["\']?{key}["\']?\s*:\s*(\d+)', text, re.IGNORECASE
        )
        if pat:
            scores[key] = int(pat.group(1))

    if len(scores) == len(SCORE_KEYS):
        return scores, text

    # All strategies failed
    return None, text


# ─────────────────────────────────────────────────────────────────────────────
# Async Disk Writer — offloads I/O from the GPU loop
# ─────────────────────────────────────────────────────────────────────────────
_WRITER_SENTINEL = None


class AsyncWriter:
    def __init__(self, jsonl_path: str, queue_maxsize: int = 512) -> None:
        self.jsonl_path = jsonl_path
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._errors: List[Exception] = []
        self._thread = threading.Thread(
            target=self._worker,
            name=f"disk-writer-rank{RANK}",
            daemon=True,
        )
        self._thread.start()

    def put(self, record: Dict) -> None:
        self._q.put(record)

    def _worker(self) -> None:
        with open(self.jsonl_path, "w", encoding="utf-8", buffering=1) as f:
            while True:
                record = self._q.get()
                if record is _WRITER_SENTINEL:
                    break
                try:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    self._errors.append(exc)
                    log(f"[AsyncWriter] ERROR: {exc}")
                finally:
                    self._q.task_done()

    def close(self) -> bool:
        self._q.join()
        self._q.put(_WRITER_SENTINEL)
        self._thread.join(timeout=60)
        if self._errors:
            log(f"[AsyncWriter] {len(self._errors)} write error(s).")
        return len(self._errors) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
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
    except Exception as exc:
        log(f"FATAL: Model load failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation — run by rank 0 after all ranks finish
# ─────────────────────────────────────────────────────────────────────────────
def _avg_group(records: List[Dict]) -> Dict:
    """
    Given a list of valid score records, return a dict with:
      n, avg per score key, and combined complexity_score.
    """
    n = len(records)
    avgs: Dict[str, Optional[float]] = {}
    for key in SCORE_KEYS:
        vals = [r[key] for r in records if r.get(key) is not None]
        avgs[key] = round(sum(vals) / len(vals), 2) if vals else None

    non_none = [v for v in avgs.values() if v is not None]
    combined = round(sum(non_none) / len(non_none), 2) if non_none else None

    return {
        "n_samples":               n,
        "avg_pose_complexity":     avgs.get("pose_complexity"),
        "avg_occlusion_complexity": avgs.get("occlusion_complexity"),
        "avg_implausibility":      avgs.get("implausibility"),
        "complexity_score":        combined,
    }


def _fmt(v) -> str:
    """Format a nullable float for printing."""
    return f"{v:.2f}" if v is not None else "N/A"


def aggregate_scores(output_dir: str, world_size: int, summary_path: str) -> None:
    """
    Reads all per-rank JSONL files, groups records by difficulty × gender,
    prints scores for each sub-group, per-difficulty totals, and an overall
    average, then writes a full summary JSON.
    """
    log("Aggregating scores from all ranks …")

    all_records: List[Dict] = []
    for r in range(world_size):
        jpath = os.path.join(output_dir, f"scores_rank{r}.jsonl")
        if not os.path.exists(jpath):
            log(f"  WARNING: missing {jpath} — skipping")
            continue
        with open(jpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    log(f"  Total records collected : {len(all_records)}")

    valid_records   = [r for r in all_records if r.get("parse_ok")]
    invalid_records = [r for r in all_records if not r.get("parse_ok")]
    log(f"  Parse OK : {len(valid_records)} | Parse FAILED : {len(invalid_records)}")

    # ── Build groups ─────────────────────────────────────────────────────────
    DIFFICULTIES = ("medium", "hard")
    GENDERS      = ("male", "female")

    # difficulty → gender → [records]
    groups: Dict[str, Dict[str, List[Dict]]] = {
        d: {g: [] for g in GENDERS} for d in DIFFICULTIES
    }
    for rec in valid_records:
        diff   = rec.get("difficulty", "")
        gender = rec.get("gender", "")
        if diff in groups and gender in groups[diff]:
            groups[diff][gender].append(rec)

    # ── Compute stats ────────────────────────────────────────────────────────
    stats: Dict[str, Dict] = {}          # "medium_male", "medium_female", …
    diff_totals: Dict[str, Dict] = {}    # "medium", "hard"
    overall_all: List[Dict] = []

    for diff in DIFFICULTIES:
        diff_records: List[Dict] = []
        for gender in GENDERS:
            key = f"{diff}_{gender}"
            recs = groups[diff][gender]
            stats[key] = _avg_group(recs)
            diff_records.extend(recs)

        diff_totals[diff] = _avg_group(diff_records)
        overall_all.extend(diff_records)

    overall = _avg_group(overall_all)

    # ── Write summary JSON ───────────────────────────────────────────────────
    summary = {
        "generated_at":   datetime.now().isoformat(),
        "total_images":   len(all_records),
        "parse_ok":        len(valid_records),
        "parse_failed":   len(invalid_records),
        "by_group":       stats,
        "by_difficulty":  diff_totals,
        "overall":        overall,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"Summary JSON written → {summary_path}")

    # ── Pretty-print table ───────────────────────────────────────────────────
    W = 74
    SEP  = "─" * W
    DSEP = "═" * W

    def row(label: str, s: Dict) -> str:
        pose = _fmt(s["avg_pose_complexity"])
        occ  = _fmt(s["avg_occlusion_complexity"])
        impl = _fmt(s["avg_implausibility"])
        comp = _fmt(s["complexity_score"])
        n    = s["n_samples"]
        return (
            f"  ║  {label:<22} │ n={n:>5} │ "
            f"pose={pose:>6} │ occ={occ:>6} │ "
            f"impl={impl:>6} │ complexity={comp:>6}  ║"
        )

    lines = [
        f"\n  ╔{DSEP}╗",
        f"  ║{'COMPLEXITY SCORE REPORT':^{W}}║",
        f"  ╠{DSEP}╣",
        f"  ║  {'GROUP':<22} │ {'N':>7} │ "
        f"{'POSE':>10} │ {'OCCLUSION':>10} │ "
        f"{'IMPLAUS':>10} │ {'COMPLEXITY':>16}  ║",
        f"  ╠{SEP}╣",
    ]

    for diff in DIFFICULTIES:
        lines.append(f"  ║  {'── ' + diff.upper() + ' ──':<{W-2}}║")
        for gender in GENDERS:
            key   = f"{diff}_{gender}"
            label = f"  {diff}/{gender}"
            lines.append(row(label, stats[key]))
        lines.append(f"  ╟{SEP}╢")
        lines.append(row(f"  {diff}/TOTAL (avg)", diff_totals[diff]))
        lines.append(f"  ╠{SEP}╣")

    lines.append(row("  OVERALL TOTAL", overall))
    lines.append(f"  ╚{DSEP}╝\n")

    for line in lines:
        log(line)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM complexity scorer for edit output images"
    )
    parser.add_argument(
        "--edit_outputs_dir",
        type=str,
        default="PATH_TO_EDIT_OUTPUTS",
        help="Root directory containing hard/medium × male/female sub-dirs of PNGs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="PATH_TO_SCORE_OUTPUTS",
        help="Directory where per-rank JSONL and final summary are written.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per GPU (use 32 as default; reduce if OOM).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Token budget for VLM score output (JSON is very short).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize PNGs to this square size before feeding the VLM.",
    )
    args = parser.parse_args()

    # Propagate image size to module constant used by dataset
    global IMAGE_SIZE
    IMAGE_SIZE = args.image_size

    import socket
    JOB_START = datetime.now()

    log(
        f"host={socket.gethostname()} | RANK={RANK} LOCAL_RANK={LOCAL_RANK} "
        f"WORLD_SIZE={WORLD_SIZE} | start={JOB_START.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if IS_MAIN:
        print(f"\n{'='*60}", flush=True)
        print(f"Edit Output Complexity Scorer  |  {WORLD_SIZE} workers", flush=True)
        print(
            f"Batch/GPU : {args.batch_size} | num_workers: {args.num_workers} | "
            f"max_new_tokens: {args.max_new_tokens} | img_size: {IMAGE_SIZE}×{IMAGE_SIZE}",
            flush=True,
        )
        print(f"Edit outputs dir : {args.edit_outputs_dir}", flush=True)
        print(f"Output dir       : {args.output_dir}", flush=True)
        print(f"Model            : {args.model_name}", flush=True)
        print(f"Job start        : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'='*60}\n", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Processor ────────────────────────────────────────────────────────────
    log(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    log("Processor ready.")

    # ── Dataset ──────────────────────────────────────────────────────────────
    log("Scanning edit output directories …")
    dataset = EditOutputDataset(
        root_dir=args.edit_outputs_dir,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    n_batches = math.ceil(len(dataset) / args.batch_size)
    log(
        f"Dataset: {len(dataset)} images | "
        f"batch_size={args.batch_size} | "
        f"batches={n_batches} (ceil, no drop_last)"
    )

    if len(dataset) == 0:
        log("WARNING: no images found for this rank — exiting cleanly.")
        return

    # ── DataLoader ───────────────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=make_collate_fn(processor),
        num_workers=args.num_workers,
        pin_memory=False,   # device_map=auto manages placement itself
        drop_last=False,    # every sample scored, including remainder batch
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )
    log(f"DataLoader ready: {len(loader)} batches.")

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_model(args.model_name)

    # Primary (first) device for input tensors with device_map=auto
    if hasattr(model, "hf_device_map"):
        non_cpu = [str(v) for v in model.hf_device_map.values() if str(v) != "cpu"]
        primary_device = torch.device(non_cpu[0]) if non_cpu else torch.device("cuda:0")
    else:
        primary_device = torch.device("cuda:0")

    log(f"Primary device for inputs: {primary_device}")
    log("Starting inference — each rank independent, no barrier.")

    # ── Async writer ─────────────────────────────────────────────────────────
    jsonl_path = os.path.join(args.output_dir, f"scores_rank{RANK}.jsonl")
    writer = AsyncWriter(
        jsonl_path=jsonl_path,
        queue_maxsize=args.batch_size * 8,
    )
    log(f"Async disk writer started → {jsonl_path}")

    # ── Inference loop ───────────────────────────────────────────────────────
    total_start        = time.perf_counter()
    batch_latencies:   List[float] = []
    total_samples_done = 0
    parse_failures     = 0

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

        # Decode — strip prompt tokens, keep only generated text
        input_len = ids.shape[1]
        new_ids   = gen_ids[:, input_len:]
        texts     = processor.tokenizer.batch_decode(
            new_ids, skip_special_tokens=True
        )

        log(
            f"Batch {batch_idx+1}/{len(loader)} | "
            f"samples={bsz} | "
            f"batch={batch_time:.2f}s | "
            f"per_sample={per_sample:.3f}s | "
            f"queue={writer._q.qsize()}"
        )

        for i, (raw_text, image_path, image_name, difficulty, gender) in enumerate(
            zip(
                texts,
                batch["image_paths"],
                batch["image_names"],
                batch["difficulties"],
                batch["genders"],
            )
        ):
            scores, cleaned_text = parse_scores(raw_text)
            ok = scores is not None
            if not ok:
                parse_failures += 1
                log(
                    f"  [parse FAIL] rank={RANK} batch={batch_idx} item={i} "
                    f"path={image_path} | raw={repr(raw_text[:120])}"
                )

            record: Dict = {
                "image_path":  image_path,
                "image_name":  image_name,
                "difficulty":  difficulty,
                "gender":      gender,
                "parse_ok":    ok,
                "raw_output":  cleaned_text,
                "gpu_rank":    RANK,
                "batch_idx":   batch_idx,
                "batch_time_s": round(batch_time, 4),
                "per_sample_s": round(per_sample, 4),
            }
            if ok:
                record.update(scores)   # adds pose_complexity, occlusion_complexity, implausibility
            else:
                for key in SCORE_KEYS:
                    record[key] = None

            writer.put(record)

    # ── Flush writer ─────────────────────────────────────────────────────────
    log("Inference done — flushing writer queue …")
    ok = writer.close()
    log(f"Writer finished. {'All files written OK.' if ok else 'Some write errors!'}")

    # ── Per-rank summary ─────────────────────────────────────────────────────
    total_time = time.perf_counter() - total_start
    avg_batch  = sum(batch_latencies) / len(batch_latencies) if batch_latencies else 0
    throughput = total_samples_done / total_time if total_time > 0 else 0
    JOB_END    = datetime.now()

    log(f"\n{'─'*52}")
    log(f"SUMMARY  (Rank {RANK} / GPU {LOCAL_RANK})")
    log(f"{'─'*52}")
    log(f"  Images scored          : {total_samples_done}")
    log(f"  Parse failures         : {parse_failures}")
    log(f"  Batches                : {len(batch_latencies)}")
    log(f"  Job start              : {JOB_START.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Job end                : {JOB_END.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total wall time        : {total_time:.2f}s")
    log(f"  Avg batch latency      : {avg_batch:.3f}s")
    log(f"  Avg per-sample latency : {total_time / max(total_samples_done, 1):.4f}s")
    log(f"  Throughput             : {throughput:.3f} images/sec")
    log(f"  JSONL output           : {jsonl_path}")
    log(f"{'─'*52}\n")

    # ── Aggregation — only rank 0, after all ranks have written ──────────────
    # Because each rank runs independently (no barrier), rank 0 cannot know
    # when the other ranks are done. The aggregation step is therefore run as
    # a separate srun step or by calling this script with --aggregate_only.
    # We write a sentinel file to signal this rank's completion.
    sentinel = os.path.join(args.output_dir, f"done_rank{RANK}.sentinel")
    with open(sentinel, "w") as f:
        f.write(f"done at {JOB_END.isoformat()}\n")
    log(f"Sentinel written: {sentinel}")

    if IS_MAIN:
        # Wait (poll) until all other ranks have written their sentinel files
        log("Rank 0: waiting for all other ranks to finish …")
        deadline = time.time() + 1800  # 30-minute timeout
        while time.time() < deadline:
            missing = [
                r for r in range(WORLD_SIZE)
                if not os.path.exists(
                    os.path.join(args.output_dir, f"done_rank{r}.sentinel")
                )
            ]
            if not missing:
                break
            log(f"  Still waiting for ranks: {missing}")
            time.sleep(30)
        else:
            log("WARNING: timeout waiting for other ranks. Aggregating with available data.")

        summary_path = os.path.join(args.output_dir, "complexity_summary.json")
        aggregate_scores(args.output_dir, WORLD_SIZE, summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL EXCEPTION:")
        tb.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
