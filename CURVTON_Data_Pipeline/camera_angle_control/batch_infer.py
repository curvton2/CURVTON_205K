"""
Qwen Image Edit — Batch Camera Angle Control Inference
=======================================================
Supports ALL transformations the LoRA (dx8152/Qwen-Edit-2509-Multiple-angles) was trained on:

  ROTATION
    rotate_left_45       → Rotate the camera 45 degrees to the left.
    rotate_right_45      → Rotate the camera 45 degrees to the right.
    rotate_left_90       → Rotate the camera 90 degrees to the left.
    rotate_right_90      → Rotate the camera 90 degrees to the right.

  TRANSLATION (lateral / vertical)
    move_left            → Move the camera left.
    move_right           → Move the camera right.
    move_up              → Move the camera up.
    move_down            → Move the camera down.
    move_forward         → Move the camera forward.

  ZOOM / FOCAL LENGTH
    close_up             → Turn the camera to a close-up.
    wide_angle           → Turn the camera to a wide-angle lens.

  VERTICAL TILT (angle of view)
    top_down             → Turn the camera to a top-down view.   (bird's-eye)
    look_up              → Turn the camera to a worm's-eye view.

  CUSTOM PROMPT
    Pass any bilingual or free-form prompt directly, e.g.:
      "将镜头向左移动并转为广角镜头 Move the camera left and switch to wide-angle."

-------------------------------------------------------------------------------
WHY batch_size IS FIXED TO 1
-------------------------------------------------------------------------------
The pipeline's internal Qwen2VLProcessor builds image-token sequences one
image at a time.  Passing multiple images in a single call causes:

    IndexError: index 4 is out of bounds for dimension 0 with size 4

because the processor's image-grid index goes out of bounds at index 1+
when each prompt references exactly one image.

The correct approach for this pipeline is:
  • DataLoader with num_workers > 0  → parallel CPU I/O & image decoding
  • Pipeline called once per sample  → one image through the GPU at a time
  • CUDA Event latency tracking      → accurate per-sample GPU timing
  • tqdm progress bar                → live progress + per-sample timing

The CPU-side parallelism is real and meaningful: workers prefetch and decode
the next N images on CPU while the GPU is busy with the current one.

-------------------------------------------------------------------------------
CSV FORMAT  (header required)
-------------------------------------------------------------------------------
    image_path, transform, prompt, seed

    • transform  : one of the keys above (e.g. rotate_left_45) OR leave blank
    • prompt     : raw bilingual prompt — used AS-IS if transform is blank.
                   If both are provided they are COMBINED.
    • seed       : integer >= 0 for reproducibility, -1 for random.

Example rows:
    PATH_TO_IMAGE_1.jpg,rotate_left_45,,-1
    PATH_TO_IMAGE_2.jpg,top_down,,-1
    PATH_TO_IMAGE_3.jpg,wide_angle,,42
    PATH_TO_IMAGE_4.jpg,,"将镜头向左移动 Move the camera left.",-1
    PATH_TO_IMAGE_5.jpg,move_forward,"将镜头转为广角镜头 Turn the camera to a wide-angle lens.",123

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
    python batch_infer.py --csv samples.csv --output_dir ./results

    # More CPU I/O workers (speeds up image loading)
    python batch_infer.py --csv samples.csv --output_dir ./results --num_workers 4

    # Higher quality
    python batch_infer.py --csv samples.csv --output_dir ./results --steps 20
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel


# =============================================================================
# All supported transformations
# =============================================================================

TRANSFORM_PROMPTS: Dict[str, Tuple[str, str]] = {
    # Rotation
    "rotate_left_45":  ("将镜头向左旋转45度",  "Rotate the camera 45 degrees to the left."),
    "rotate_right_45": ("将镜头向右旋转45度",  "Rotate the camera 45 degrees to the right."),
    "rotate_left_90":  ("将镜头向左旋转90度",  "Rotate the camera 90 degrees to the left."),
    "rotate_right_90": ("将镜头向右旋转90度",  "Rotate the camera 90 degrees to the right."),
    # Lateral / vertical translation
    "move_left":       ("将镜头向左移动",       "Move the camera left."),
    "move_right":      ("将镜头向右移动",       "Move the camera right."),
    "move_up":         ("将镜头向上移动",       "Move the camera up."),
    "move_down":       ("将镜头向下移动",       "Move the camera down."),
    "move_forward":    ("将镜头向前移动",       "Move the camera forward."),
    # Zoom / focal length
    "close_up":        ("将镜头转为特写镜头",   "Turn the camera to a close-up."),
    "wide_angle":      ("将镜头转为广角镜头",   "Turn the camera to a wide-angle lens."),
    # Vertical tilt
    "top_down":        ("将镜头转为俯视",       "Turn the camera to a top-down view."),
    "look_up":         ("将相机切换到仰视视角", "Turn the camera to a worm's-eye view."),
}

TARGET_AREA = 1024 * 1024
VAE_STRIDE  = 16          # vae_scale_factor(8) x patch_size(2)


# =============================================================================
# Helpers
# =============================================================================

def build_prompt_from_transform(key: str) -> str:
    if key not in TRANSFORM_PROMPTS:
        valid = "\n  ".join(TRANSFORM_PROMPTS.keys())
        raise ValueError(f"Unknown transform '{key}'. Valid keys:\n  {valid}")
    zh, en = TRANSFORM_PROMPTS[key]
    return f"{zh} {en}"


def resolve_prompt(transform: str, custom_prompt: str) -> str:
    transform     = (transform     or "").strip()
    custom_prompt = (custom_prompt or "").strip()
    if not transform and not custom_prompt:
        raise ValueError("Each row needs a 'transform' key, a 'prompt', or both.")
    parts = []
    if transform:
        parts.append(build_prompt_from_transform(transform))
    if custom_prompt:
        parts.append(custom_prompt)
    return " ".join(parts)


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    w = math.sqrt(target_area * ratio)
    h = w / ratio
    w = round(w / VAE_STRIDE) * VAE_STRIDE
    h = round(h / VAE_STRIDE) * VAE_STRIDE
    return int(w), int(h)


# =============================================================================
# Sample dataclass
# =============================================================================

@dataclass
class InferenceSample:
    image_path:      str
    transform:       str
    custom_prompt:   str
    seed:            int = -1
    resolved_prompt: str = field(default="", init=False)

    def __post_init__(self):
        self.resolved_prompt = resolve_prompt(self.transform, self.custom_prompt)


# =============================================================================
# Dataset
# =============================================================================

class CameraEditDataset(Dataset):
    """
    Reads a CSV and loads images inside __getitem__ so DataLoader workers
    handle I/O in parallel on CPU while the GPU processes previous samples.

    force_size: if set to (width, height), ALL samples use that exact
    output resolution instead of auto-computing from aspect ratio.
    Required when using torch.compile to avoid per-shape recompilation.
    """

    def __init__(self, csv_path: str, force_size: tuple = None):
        self.samples: List[InferenceSample] = []
        self.force_size = force_size  # (width, height) or None
        self.samples: List[InferenceSample] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV '{csv_path}' is empty.")
            missing = {"image_path"} - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for line_num, row in enumerate(reader, start=2):
                image_path = row.get("image_path", "").strip()
                if not image_path:
                    print(f"[WARN] line {line_num}: empty image_path — skipped.")
                    continue
                if not os.path.isfile(image_path):
                    print(f"[WARN] line {line_num}: not found '{image_path}' — skipped.")
                    continue
                try:
                    sample = InferenceSample(
                        image_path=image_path,
                        transform=row.get("transform", "").strip(),
                        custom_prompt=row.get("prompt", "").strip(),
                        seed=int(row.get("seed", -1)),
                    )
                    self.samples.append(sample)
                except ValueError as e:
                    print(f"[WARN] line {line_num}: {e} — skipped.")

        if not self.samples:
            raise RuntimeError("No valid samples found in the CSV.")
        print(f"Loaded {len(self.samples)} valid samples from '{csv_path}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # Runs in worker process → parallel disk I/O + image decode
        image = Image.open(s.image_path).convert("RGB")
        orig_w, orig_h = image.size
        if self.force_size is not None:
            out_w, out_h = self.force_size
        else:
            out_w, out_h = calculate_dimensions(TARGET_AREA, orig_w / orig_h)
        seed = s.seed if s.seed >= 0 else random.randint(0, 2 ** 31 - 1)
        return {
            "idx":    idx,
            "image":  image,
            "prompt": s.resolved_prompt,
            "seed":   seed,
            "out_w":  out_w,
            "out_h":  out_h,
            "path":   s.image_path,
        }


# =============================================================================
# Collate function
# =============================================================================

def camera_edit_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    PIL Images cannot be torch.stack'd — keep everything as Python lists.
    batch_size=1 means each list has exactly one element, but writing it
    as a loop keeps the code forward-compatible.
    """
    return {
        "images":  [item["image"]  for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "seeds":   [item["seed"]   for item in batch],
        "out_ws":  [item["out_w"]  for item in batch],
        "out_hs":  [item["out_h"]  for item in batch],
        "idxs":    [item["idx"]    for item in batch],
        "paths":   [item["path"]   for item in batch],
    }


# =============================================================================
# Model loading
# =============================================================================

def load_pipeline(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    compile_transformer: bool = False,
) -> QwenImageEditPlusPipeline:
    """
    Load the pipeline.  Optionally torch.compile the diffusion transformer
    for 15-30% faster GPU execution after a one-time warmup cost.

    compile_transformer=True notes:
      • First inference call triggers compilation — takes ~2 min on H200.
      • Subsequent calls are faster (compiled kernel is cached in memory).
      • Compiled cache is NOT saved to disk — recompiles on every cold start.
      • Use mode='reduce-overhead' for repeated fixed-shape calls (our case).
      • fullgraph=False allows partial compilation — safer with diffusers.
    """

    print("─" * 62)
    print("1/3  Loading transformer (Rapid-AIO 4-step) …")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cuda",
    )

    print("2/3  Loading pipeline (Qwen-Image-Edit-2509) …")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    print("3/3  Loading & fusing camera-angles LoRA …")
    pipe.load_lora_weights(
        "dx8152/Qwen-Edit-2509-Multiple-angles",
        weight_name="镜头转换.safetensors",
        adapter_name="angles",
    )
    pipe.set_adapters(["angles"], adapter_weights=[1.0])
    pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
    pipe.unload_lora_weights()

    # ── torch.compile (optional) ──────────────────────────────────────────
    # Must be applied AFTER LoRA is fused and unloaded so the compiled graph
    # bakes in the final merged weights.  Compiling before fusing would
    # produce a compiled graph without the LoRA weights.
    if compile_transformer:
        print("Compiling transformer with torch.compile …")
        print("  (first inference will take ~2 min to compile — subsequent calls are faster)")
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",  # best for repeated calls with fixed shapes
            fullgraph=False,         # partial compilation — safer with diffusers internals
        )
        print("  Compilation graph built. Warmup will happen on first inference call.")

    print("Pipeline ready.")
    print("─" * 62 + "\n")
    return pipe


# =============================================================================
# Latency tracker
# =============================================================================

class LatencyTracker:

    def __init__(self):
        self.records: List[Dict] = []

    def record(
        self,
        sample_idx: int,
        image_path: str,
        prompt: str,
        gpu_s: float,
        wall_s: float,
    ):
        self.records.append({
            "sample_idx":  sample_idx,
            "image_path":  image_path,
            "prompt":      prompt,
            "gpu_time_s":  round(gpu_s,  4),
            "wall_time_s": round(wall_s, 4),
        })

    def summary(self) -> Dict:
        n          = len(self.records)
        total_gpu  = sum(r["gpu_time_s"]  for r in self.records)
        total_wall = sum(r["wall_time_s"] for r in self.records)
        gpu_times  = [r["gpu_time_s"]  for r in self.records]
        wall_times = [r["wall_time_s"] for r in self.records]
        return {
            "total_samples":            n,
            "total_gpu_time_s":         round(total_gpu,  3),
            "total_wall_time_s":        round(total_wall, 3),
            "avg_gpu_per_sample_s":     round(total_gpu  / max(n, 1), 3),
            "avg_wall_per_sample_s":    round(total_wall / max(n, 1), 3),
            "min_gpu_s":                round(min(gpu_times),  3),
            "max_gpu_s":                round(max(gpu_times),  3),
            "throughput_samples_per_s": round(n / max(total_wall, 1e-9), 4),
            "per_sample_details":       self.records,
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "═" * 62)
        print("  LATENCY REPORT")
        print("═" * 62)
        print(f"  Total samples          : {s['total_samples']}")
        print(f"  Total GPU time         : {s['total_gpu_time_s']:.2f}s")
        print(f"  Total wall-clock time  : {s['total_wall_time_s']:.2f}s")
        print(f"  Avg GPU  / sample      : {s['avg_gpu_per_sample_s']:.2f}s")
        print(f"  Avg wall / sample      : {s['avg_wall_per_sample_s']:.2f}s")
        print(f"  Min / Max GPU          : {s['min_gpu_s']:.2f}s  /  {s['max_gpu_s']:.2f}s")
        print(f"  Throughput             : {s['throughput_samples_per_s']:.3f} samples/s")
        print("═" * 62 + "\n")


# =============================================================================
# Inference runner
# =============================================================================

def run_inference(
    pipe: QwenImageEditPlusPipeline,
    dataloader: DataLoader,
    output_dir: Path,
    num_inference_steps: int = 4,
    true_guidance_scale: float = 1.0,
    device: str = "cuda",
    compiled: bool = False,
    force_size: tuple = None,
) -> LatencyTracker:
    """
    Iterate over the DataLoader and run one pipeline call per sample.

    Parallelism:
      CPU workers  → prefetch + decode next images while GPU is busy
      GPU          → full utilisation, one sample per forward pass

    If compiled=True a silent warmup call is made before the timed loop
    so that compilation cost is excluded from latency numbers.
    force_size: (width, height) used for the warmup call — must match
    the resolution used for actual inference to avoid shape mismatch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = LatencyTracker()
    total   = len(dataloader.dataset)

    # ── torch.compile warmup ──────────────────────────────────────────────────
    if compiled:
        warmup_w = force_size[0] if force_size else 1024
        warmup_h = force_size[1] if force_size else 1024
        print(f"Running torch.compile warmup at {warmup_w}×{warmup_h} (this takes ~2 min) …")
        _warmup_img    = Image.new("RGB", (warmup_w, warmup_h), color=(0, 0, 0))
        _warmup_prompt = list(TRANSFORM_PROMPTS.values())[0]
        _warmup_prompt = f"{_warmup_prompt[0]} {_warmup_prompt[1]}"
        _warmup_gen    = torch.Generator(device=device).manual_seed(0)
        with torch.no_grad():
            pipe(
                image=[_warmup_img],
                prompt=_warmup_prompt,
                height=warmup_h,
                width=warmup_w,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_guidance_scale,
                generator=_warmup_gen,
                num_images_per_prompt=1,
            )
        torch.cuda.synchronize()
        del _warmup_img, _warmup_gen
        print("Warmup complete. Starting timed inference.\n")

    with tqdm(total=total, desc="Inference", unit="img", dynamic_ncols=True) as pbar:
        for batch in dataloader:
            # batch_size=1, so each list has one element
            for image, prompt, seed, out_w, out_h, sample_idx, path in zip(
                batch["images"],
                batch["prompts"],
                batch["seeds"],
                batch["out_ws"],
                batch["out_hs"],
                batch["idxs"],
                batch["paths"],
            ):
                short_name = Path(path).name
                pbar.set_description(f"{short_name}")

                generator = torch.Generator(device=device).manual_seed(int(seed))

                # ── Latency measurement ────────────────────────────────────
                # CUDA Events  : pure GPU execution time (most accurate)
                # perf_counter : wall-clock (includes Python + memory transfer)
                torch.cuda.synchronize()
                ev_start   = torch.cuda.Event(enable_timing=True)
                ev_end     = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                ev_start.record()

                # ── Single-sample pipeline call ────────────────────────────
                result: Image.Image = pipe(
                    image=[image],       # list of exactly 1 PIL Image
                    prompt=prompt,       # single string
                    height=int(out_h),
                    width=int(out_w),
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                ).images[0]

                ev_end.record()
                torch.cuda.synchronize()
                wall_end = time.perf_counter()

                gpu_s  = ev_start.elapsed_time(ev_end) / 1000.0   # ms → s
                wall_s = wall_end - wall_start

                # ── Save ──────────────────────────────────────────────────
                stem     = Path(path).stem
                out_path = output_dir / f"{stem}_out{int(sample_idx):04d}.png"
                result.save(out_path)

                tracker.record(
                    sample_idx=int(sample_idx),
                    image_path=path,
                    prompt=prompt,
                    gpu_s=gpu_s,
                    wall_s=wall_s,
                )

                pbar.set_postfix(gpu=f"{gpu_s:.1f}s", wall=f"{wall_s:.1f}s", saved=out_path.name)
                pbar.update(1)

    return tracker


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen Image Edit — Batch Camera Angle Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",         required=True,        help="Path to the input CSV.")
    p.add_argument("--output_dir",  default="./results",  help="Directory for output images + latency JSON.")
    p.add_argument("--num_workers", type=int, default=4,  help="Parallel CPU image-loading workers.")
    p.add_argument("--prefetch",    type=int, default=2,  help="DataLoader prefetch_factor.")
    p.add_argument("--steps",    type=int,   default=4,   help="Denoising steps (4=fast, 20=quality).")
    p.add_argument("--guidance", type=float, default=1.0, help="True CFG scale (1.0 = disabled).")
    p.add_argument(
        "--force_size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help=(
            "Force ALL images to this exact output resolution (width height), e.g. --force_size 1024 1024. "
            "STRONGLY recommended when using --compile: torch.compile recompiles every time "
            "it sees a new tensor shape, so mixed resolutions cause repeated recompilation. "
            "With a fixed size, compilation happens once and all subsequent calls are fast."
        ),
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help=(
            "torch.compile the diffusion transformer for 15-30%% faster GPU execution. "
            "First inference call triggers ~2 min one-time compilation. "
            "Worth enabling when processing 20+ images. "
            "Use with --force_size to avoid recompilation on different image shapes."
        ),
    )
    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    # Print supported transform keys
    print("\nSupported transform keys:")
    col = max(len(k) for k in TRANSFORM_PROMPTS) + 2
    for key, (zh, en) in TRANSFORM_PROMPTS.items():
        print(f"  {key:<{col}} {zh}  {en}")
    print()

    # Warn if --compile used without --force_size
    if args.compile and args.force_size is None:
        print(
            "⚠  WARNING: --compile is set but --force_size is not.\n"
            "   Images with different aspect ratios produce different output resolutions.\n"
            "   torch.compile recompiles every time it sees a new tensor shape.\n"
            "   This will cause repeated ~2 min recompilations and SLOWER overall runtime.\n"
            "   Add --force_size 1024 1024 to fix all images to one resolution.\n"
        )

    # Load model
    pipe = load_pipeline(device=device, dtype=dtype, compile_transformer=args.compile)

    # Dataset
    force_size = tuple(args.force_size) if args.force_size else None
    if force_size:
        # Snap to VAE_STRIDE multiple
        force_size = (
            (force_size[0] // VAE_STRIDE) * VAE_STRIDE,
            (force_size[1] // VAE_STRIDE) * VAE_STRIDE,
        )
        print(f"Force size: all outputs will be {force_size[0]}×{force_size[1]}\n")
    dataset = CameraEditDataset(csv_path=args.csv, force_size=force_size)

    # DataLoader  (batch_size MUST be 1 — see module docstring)
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=camera_edit_collate,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"]    = args.prefetch
        loader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(dataset, **loader_kwargs)
    print(f"DataLoader: batch_size=1 (fixed) | "
          f"num_workers={args.num_workers} | "
          f"prefetch={args.prefetch if args.num_workers > 0 else 'N/A'} | "
          f"total={len(dataset)} samples\n")

    # Run
    output_dir = Path(args.output_dir)
    tracker = run_inference(
        pipe=pipe,
        dataloader=dataloader,
        output_dir=output_dir,
        num_inference_steps=args.steps,
        true_guidance_scale=args.guidance,
        device=device,
        compiled=args.compile,
        force_size=force_size,
    )

    # Report
    tracker.print_summary()

    report_path = output_dir / "latency_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(tracker.summary(), f, indent=2, ensure_ascii=False)
    print(f"Latency report → {report_path}\n")


if __name__ == "__main__":
    main()
