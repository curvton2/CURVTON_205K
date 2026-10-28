#!/usr/bin/env python3
"""Evaluate virtual try-on outputs with SSIM, PSNR, LPIPS, FID, and KID.

Expected generated image layout by default:

    <generated_root>/<model>/<split_or_category>/<image files>

Real images are read from each dataset subset. If ``--real-subdir`` is not
provided, the script tries common names such as image/images/target/ground_truth
and finally the subset directory itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MODELS = ("catvton", "idmvton", "stablevton")
CURVTON_SPLITS = ("easy", "medium", "hard", "overall")
CATEGORY_SPLITS = ("upper_body", "lower_body", "dresses")
TRADITIONALITY_SPLITS = ("traditional", "non_traditional")
REAL_SUBDIR_CANDIDATES = (
    "image",
    "images",
    "target",
    "targets",
    "real",
    "gt",
    "ground_truth",
    "person",
    "persons",
)


@dataclass(frozen=True)
class ImagePair:
    real: Path
    fake: Path


class ImagePathDataset:
    def __init__(self, paths: list[Path], image_size: int, mode: str) -> None:
        from torchvision import transforms

        self.paths = paths
        self.mode = mode
        base = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
        self.to_unit = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        import torch
        image = Image.open(self.paths[index]).convert("RGB")
        tensor = self.to_unit(image)
        if self.mode == "lpips":
            return tensor.mul(2.0).sub(1.0)
        if self.mode == "kid":
            return tensor.mul(255.0).round().to(torch.uint8)
        return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SSIM, PSNR, LPIPS, FID, and KID for virtual try-on model outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--curvton-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/anonymous_user/human_gen/dataset_v3_backup_1_1/dataset_ultimate_test"),
        help="CURVTON main test dataset root containing easy/medium/hard/overall.",
    )
    parser.add_argument(
        "--category-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/anonymous_user/human_gen/dataset_v3_backup/dataset_ultimate_stratified_category"),
        help="Category-stratified root containing upper_body/lower_body/dresses "
             "(output of stratified/stratify_by_body_region.py).",
    )
    parser.add_argument(
        "--traditionality-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/anonymous_user/human_gen/dataset_v3_backup/dataset_ultimate_stratified_traditionally"),
        help="Traditionality-stratified root containing traditional/non_traditional "
             "(output of stratified/stratify_by_traditionality.py).",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        required=True,
        help="Root containing generated outputs, usually <root>/<model>/<split_or_category>.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Model names to evaluate.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("curvton", "category", "traditionality"),
        default=("curvton", "category"),
        help="Dataset groups to evaluate.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("vton_eval_results"), help="Directory for JSON/CSV outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize all images to square size before metrics.")
    parser.add_argument("--sample-ratio", type=float, default=0.2, help="Fraction of paired images to evaluate.")
    parser.add_argument("--overall-per-split", type=int, default=350, help="For overall curvton: samples per split, total=3x this.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for 20 percent sampling.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for LPIPS, FID, and KID.")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex", help="LPIPS backbone.")
    parser.add_argument("--kid-subset-size", type=int, default=1000, help="Requested KID subset size.")
    parser.add_argument("--kid-subsets", type=int, default=100, help="Number of KID random subsets.")
    parser.add_argument(
        "--real-subdir",
        default=None,
        help="Subdirectory inside each split/category containing real target images. If unset, common names are tried.",
    )
    parser.add_argument(
        "--fake-subdir-template",
        default="{model}/{subset}",
        help="Generated output path relative to --generated-root. Supports {model}, {subset}, and {dataset}.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any real image does not have a matching fake image instead of evaluating matched pairs only.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", help="Torch device. Use 'auto' to prefer CUDA when available.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=getattr(logging, level))


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@lru_cache(maxsize=None)
def list_images(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return tuple(sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))


def resolve_real_dir(dataset_root: Path, subset: str, real_subdir: str | None) -> Path:
    subset_dir = dataset_root / subset
    if real_subdir:
        candidate = subset_dir / real_subdir
        if not candidate.exists():
            raise FileNotFoundError(f"Configured real subdir does not exist: {candidate}")
        return candidate

    for name in REAL_SUBDIR_CANDIDATES:
        candidate = subset_dir / name
        if candidate.exists() and any(candidate.rglob("*")):
            return candidate
    return subset_dir


def resolve_real_dirs(dataset_root: Path, subset: str, real_subdir: str | None) -> list[Path]:
    if subset == "overall" and not (dataset_root / subset).exists():
        dirs: list[Path] = []
        for split in ("easy", "medium", "hard"):
            dirs.extend(resolve_real_dirs(dataset_root, split, real_subdir))
        return dirs
    if real_subdir is None:
        gender_tryon_dirs = [
            dataset_root / subset / gender / "tryon_image"
            for gender in ("female", "male")
            if (dataset_root / subset / gender / "tryon_image").exists()
            and any((dataset_root / subset / gender / "tryon_image").iterdir())
        ]
        if gender_tryon_dirs:
            return gender_tryon_dirs
    return [resolve_real_dir(dataset_root, subset, real_subdir)]


def resolve_fake_dir(generated_root: Path, template: str, model: str, dataset: str, subset: str) -> Path:
    rel = template.format(model=model, dataset=dataset, subset=subset)
    return generated_root / rel


def resolve_fake_dirs(generated_root: Path, template: str, model: str, dataset: str, subset: str) -> list[Path]:
    fake_dir = resolve_fake_dir(generated_root, template, model, dataset, subset)
    if subset == "overall" and not fake_dir.exists():
        dirs: list[Path] = []
        for split in ("easy", "medium", "hard"):
            dirs.extend(resolve_fake_dirs(generated_root, template, model, dataset, split))
        return dirs
    gender_tryon_dirs = [
        fake_dir / gender / "tryon_image"
        for gender in ("female", "male")
        if (fake_dir / gender / "tryon_image").exists()
        and any((fake_dir / gender / "tryon_image").iterdir())
    ]
    if gender_tryon_dirs:
        return gender_tryon_dirs
    return [fake_dir]


def image_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return str(rel.with_suffix("")).replace("\\", "/")


def image_stem_key(path: Path) -> str:
    return path.stem


def make_pairs(real_dir: Path, fake_dir: Path, strict: bool) -> list[ImagePair]:
    real_images = list_images(real_dir)
    fake_images = list_images(fake_dir)
    if not real_images:
        raise ValueError(f"No real images found in {real_dir}")
    if not fake_images:
        raise ValueError(f"No fake images found in {fake_dir}")

    fake_by_rel = {image_key(path, fake_dir): path for path in fake_images}
    fake_by_stem: dict[str, list[Path]] = {}
    for path in fake_images:
        fake_by_stem.setdefault(image_stem_key(path), []).append(path)

    pairs: list[ImagePair] = []
    missing: list[Path] = []
    ambiguous: list[str] = []
    for real in real_images:
        rel_key = image_key(real, real_dir)
        fake = fake_by_rel.get(rel_key)
        if fake is None:
            candidates = fake_by_stem.get(image_stem_key(real), [])
            if len(candidates) == 1:
                fake = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append(real.name)
        if fake is None:
            missing.append(real)
            continue
        pairs.append(ImagePair(real=real, fake=fake))

    if strict and (missing or ambiguous):
        raise RuntimeError(
            f"Pairing failed for {real_dir} vs {fake_dir}: "
            f"{len(missing)} missing, {len(ambiguous)} ambiguous."
        )
    if missing:
        logging.warning("Skipping %d real images without a matching fake in %s", len(missing), fake_dir)
    if ambiguous:
        logging.warning("Skipping %d real images with ambiguous stem matches in %s", len(ambiguous), fake_dir)
    if not pairs:
        raise ValueError(f"No matched real/fake pairs found for {real_dir} and {fake_dir}")
    return pairs


def make_pairs_from_dirs(real_dirs: list[Path], fake_dirs: list[Path], strict: bool) -> list[ImagePair]:
    if len(real_dirs) != len(fake_dirs):
        raise ValueError(f"Need equal real/fake directory groups, got {len(real_dirs)} and {len(fake_dirs)}")
    pairs: list[ImagePair] = []
    for real_dir, fake_dir in zip(real_dirs, fake_dirs):
        pairs.extend(make_pairs(real_dir, fake_dir, strict))
    return pairs


def sample_20_percent(pairs: list[ImagePair], sample_ratio: float, seed: int) -> list[ImagePair]:
    if not 0.0 < sample_ratio <= 1.0:
        raise ValueError("--sample-ratio must be in (0, 1].")
    sample_count = int(sample_ratio * len(pairs))
    sample_count = max(1, sample_count)
    rng = random.Random(seed)
    return rng.sample(pairs, sample_count)


def load_rgb_unit(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_ssim(pairs: list[ImagePair], image_size: int) -> float:
    from skimage.metrics import structural_similarity as skimage_ssim

    scores = []
    for pair in pairs:
        real = load_rgb_unit(pair.real, image_size)
        fake = load_rgb_unit(pair.fake, image_size)
        scores.append(skimage_ssim(real, fake, channel_axis=2, data_range=1.0))
    return float(np.mean(scores))


def compute_psnr(pairs: list[ImagePair], image_size: int) -> float:
    vals = []
    for pair in pairs:
        real = load_rgb_unit(pair.real, image_size)
        fake = load_rgb_unit(pair.fake, image_size)
        mse = np.mean((real - fake) ** 2)
        vals.append(100.0 if mse < 1e-10 else float(20 * np.log10(1.0 / np.sqrt(mse))))
    return float(np.mean(vals)) if vals else 0.0


def compute_lpips(pairs: list[ImagePair], image_size: int, batch_size: int, device: Any, net: str, workers: int) -> float:
    import lpips
    import torch
    from torch.utils.data import DataLoader

    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    real_loader = DataLoader(
        ImagePathDataset([pair.real for pair in pairs], image_size, "lpips"),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    fake_loader = DataLoader(
        ImagePathDataset([pair.fake for pair in pairs], image_size, "lpips"),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    total = 0.0
    count = 0
    with torch.no_grad():
        for real, fake in zip(real_loader, fake_loader):
            real = real.to(device, non_blocking=True)
            fake = fake.to(device, non_blocking=True)
            values = loss_fn(real, fake).flatten()
            total += float(values.sum().item())
            count += int(values.numel())
    return total / count


def save_resized_rgb(src: Path, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    image.save(dst)


def prepare_sample_dirs(pairs: list[ImagePair], work_dir: Path, image_size: int) -> tuple[Path, Path]:
    real_dir = work_dir / "real"
    fake_dir = work_dir / "fake"
    for idx, pair in enumerate(pairs):
        save_resized_rgb(pair.real, real_dir / f"{idx:08d}.png", image_size)
        save_resized_rgb(pair.fake, fake_dir / f"{idx:08d}.png", image_size)
    return real_dir, fake_dir


def compute_fid(real_dir: Path, fake_dir: Path, batch_size: int, device: Any) -> float:
    from pytorch_fid import fid_score

    return float(
        fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=batch_size,
            device=str(device),
            dims=2048,
        )
    )


def compute_kid(
    pairs: list[ImagePair],
    image_size: int,
    batch_size: int,
    device: Any,
    requested_subset_size: int,
    subsets: int,
    workers: int,
) -> tuple[float, float]:
    import torch
    from torch.utils.data import DataLoader

    from torchmetrics.image.kid import KernelInceptionDistance

    subset_size = min(requested_subset_size, len(pairs))
    if subset_size < 2:
        logging.warning("KID needs at least two samples; returning NaN.")
        return math.nan, math.nan

    metric = KernelInceptionDistance(
        subset_size=subset_size,
        subsets=subsets,
        normalize=False,
    ).to(device)
    real_loader = DataLoader(
        ImagePathDataset([pair.real for pair in pairs], image_size, "kid"),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    fake_loader = DataLoader(
        ImagePathDataset([pair.fake for pair in pairs], image_size, "kid"),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        for images in real_loader:
            metric.update(images.to(device, non_blocking=True), real=True)
        for images in fake_loader:
            metric.update(images.to(device, non_blocking=True), real=False)
    mean, std = metric.compute()
    return float(mean.item()), float(std.item())


def evaluate_subset(
    *,
    dataset_name: str,
    subset: str,
    model: str,
    dataset_root: Path,
    generated_root: Path,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, float | int | str]:
    real_dirs = resolve_real_dirs(dataset_root, subset, args.real_subdir)
    fake_dirs = resolve_fake_dirs(generated_root, args.fake_subdir_template, model, dataset_name, subset)
    if len(real_dirs) > 1 and len(fake_dirs) > 0 and len(real_dirs) != len(fake_dirs):
        fake_genders = {d.parent.name for d in fake_dirs}
        real_dirs = [d for d in real_dirs if d.parent.name in fake_genders]
    pairs = make_pairs_from_dirs(real_dirs, fake_dirs, args.strict)
    # For overall curvton: sample 350 from each split so overall=1050
    if subset == "overall" and dataset_name == "curvton" and getattr(args, "overall_per_split", 350) > 0:
        per_split = args.overall_per_split
        split_pairs = {}
        for p in pairs:
            key = p.real.parts[-4] if len(p.real.parts) >= 4 else "unknown"
            split_pairs.setdefault(key, []).append(p)
        import random as _random; rng = _random.Random(args.seed)
        sampled = []
        for split_name in ("easy", "medium", "hard"):
            sp = split_pairs.get(split_name, [])
            k = min(per_split, len(sp))
            sampled.extend(rng.sample(sp, k))
    else:
        sampled = sample_20_percent(pairs, args.sample_ratio, args.seed)
    logging.info(
        "%s | %s | %s: %d matched, %d sampled",
        dataset_name,
        model,
        subset,
        len(pairs),
        len(sampled),
    )

    with tempfile.TemporaryDirectory(prefix=f"vton_eval_{dataset_name}_{model}_{subset}_") as tmp:
        sampled_real_dir, sampled_fake_dir = prepare_sample_dirs(sampled, Path(tmp), args.image_size)
        ssim = compute_ssim(sampled, args.image_size)
        psnr = compute_psnr(sampled, args.image_size)
        lpips_value = compute_lpips(
            sampled, args.image_size, args.batch_size, device, args.lpips_net, args.num_workers
        )
        fid = compute_fid(sampled_real_dir, sampled_fake_dir, args.batch_size, device)
        kid_mean, kid_std = compute_kid(
            sampled,
            args.image_size,
            args.batch_size,
            device,
            args.kid_subset_size,
            args.kid_subsets,
            args.num_workers,
        )

    return {
        "real_dir": ";".join(str(path) for path in real_dirs),
        "fake_dir": ";".join(str(path) for path in fake_dirs),
        "matched_images": len(pairs),
        "sampled_images": len(sampled),
        "ssim": ssim,
        "psnr": psnr,
        "lpips": lpips_value,
        "fid": fid,
        "kid": kid_mean,
        "kid_std": kid_std,
    }


def evaluate_all(args: argparse.Namespace) -> dict[str, dict[str, dict[str, dict[str, float | int | str]]]]:
    import torch

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    results: dict[str, dict[str, dict[str, dict[str, float | int | str]]]] = {}
    all_datasets = {
        "curvton": (args.curvton_root, CURVTON_SPLITS),
        "category": (args.category_root, CATEGORY_SPLITS),
        "traditionality": (args.traditionality_root, TRADITIONALITY_SPLITS),
    }
    datasets = {name: all_datasets[name] for name in args.datasets}

    for model in args.models:
        results[model] = {}
        for dataset_name, (dataset_root, subsets) in datasets.items():
            results[model][dataset_name] = {}
            for subset in subsets:
                results[model][dataset_name][subset] = evaluate_subset(
                    dataset_name=dataset_name,
                    subset=subset,
                    model=model,
                    dataset_root=dataset_root,
                    generated_root=args.generated_root,
                    args=args,
                    device=device,
                )
    return results


def write_json(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def write_csv(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "dataset",
        "subset",
        "matched_images",
        "sampled_images",
        "ssim",
        "psnr",
        "lpips",
        "fid",
        "kid",
        "kid_std",
        "real_dir",
        "fake_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model, model_results in results.items():
            for dataset_name, dataset_results in model_results.items():
                for subset, metrics in dataset_results.items():
                    row = {"model": model, "dataset": dataset_name, "subset": subset}
                    row.update(metrics)
                    writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    seed_everything(args.seed)
    logging.info("Using device: %s", args.device)
    logging.info("Generated output root: %s", args.generated_root)

    results = evaluate_all(args)
    write_json(results, args.output_dir / "vton_metrics.json")
    write_csv(results, args.output_dir / "vton_metrics.csv")
    logging.info("Saved %s", args.output_dir / "vton_metrics.json")
    logging.info("Saved %s", args.output_dir / "vton_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
