"""
Compute FID, KID (paired + unpaired), LPIPS, SSIM for a given
pair of directories.

Usage:
    python metrics_computation.py --real_dir /path/to/ground_truth \
                            --fake_dir /path/to/generated \
                            --num 2000
"""

import re
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

from tqdm import tqdm


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ID_RE = re.compile(r"(\d+)")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def extract_id(path: Path) -> str:
    m = ID_RE.search(path.stem)
    if m is None:
        raise ValueError(f"No numeric ID in filename: {path.name}")
    return m.group(1)

def id_to_path_map(root: Path) -> dict:
    mapping = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in EXTENSIONS:
            try:
                mapping[extract_id(p)] = p
            except ValueError:
                pass
    return mapping

def all_image_paths(root: Path) -> list:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class PairedDataset(Dataset):
    def __init__(self, paths: list, img_size: int):
        self.paths = paths
        self.resize = transforms.Resize((img_size, img_size), antialias=True)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = self.resize(read_image(str(self.paths[idx]), mode=ImageReadMode.RGB))
        return {"uint8": img, "float": img.float() / 255.0}


class UnpairedDataset(Dataset):
    def __init__(self, paths: list, img_size: int, max_images: int = None, seed: int = 42):
        if max_images is not None and max_images < len(paths):
            rng = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(paths), generator=rng)[:max_images].tolist()
            paths = [paths[i] for i in sorted(idx)]
        self.paths = paths
        self.resize = transforms.Resize((img_size, img_size), antialias=True)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        return self.resize(read_image(str(self.paths[idx]), mode=ImageReadMode.RGB))


# ──────────────────────────────────────────────────────────────────────────────
# Loader builders
# ──────────────────────────────────────────────────────────────────────────────

def build_paired_loaders(real_dir, fake_dir, batch_size, num_workers, img_size, max_pairs=None):
    real_map = id_to_path_map(real_dir)
    fake_map = id_to_path_map(fake_dir)
    common_ids = sorted(real_map.keys() & fake_map.keys())

    if not common_ids:
        raise RuntimeError(f"No matching IDs between:\n  {real_dir}\n  {fake_dir}")

    total_pairs = len(common_ids)
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("--num must be a positive integer")
        if max_pairs < total_pairs:
            common_ids = common_ids[:max_pairs]

    skipped_real = len(real_map) - total_pairs
    skipped_fake = len(fake_map) - total_pairs
    capped = max_pairs is not None and max_pairs < total_pairs
    print(f"    Matched pairs : {len(common_ids)}"
          + (f"  |  cap: {max_pairs}" if capped else "")
          + (f"  |  skipped real: {skipped_real}" if skipped_real else "")
          + (f"  |  skipped fake: {skipped_fake}" if skipped_fake else ""))

    real_paths = [real_map[i] for i in common_ids]
    fake_paths = [fake_map[i] for i in common_ids]
    kw = dict(batch_size=batch_size, shuffle=False,
              num_workers=num_workers, pin_memory=True, drop_last=False)
    return (DataLoader(PairedDataset(real_paths, img_size), **kw),
            DataLoader(PairedDataset(fake_paths, img_size), **kw),
            len(common_ids))


def build_unpaired_loaders(real_dir, fake_dir, batch_size, num_workers, img_size, seed, max_images=None):
    real_paths = all_image_paths(real_dir)
    fake_paths = all_image_paths(fake_dir)
    n_use = min(len(real_paths), len(fake_paths))
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--num must be a positive integer")
        if max_images < n_use:
            n_use = max_images
    capped = max_images is not None and max_images < min(len(real_paths), len(fake_paths))
    print(f"    Unpaired      : {n_use} each"
          f"  (real={len(real_paths)}, fake={len(fake_paths)})"
          + (f"  |  cap: {max_images}" if capped else ""))
    kw = dict(batch_size=batch_size, shuffle=False,
              num_workers=num_workers, pin_memory=True, drop_last=False)
    return (DataLoader(UnpairedDataset(real_paths, img_size, n_use, seed), **kw),
            DataLoader(UnpairedDataset(fake_paths, img_size, n_use, seed), **kw),
            n_use)


# ──────────────────────────────────────────────────────────────────────────────
# Metric factory
# ──────────────────────────────────────────────────────────────────────────────

def make_fid_kid(kid_subsets, kid_subset_size, device):
    fid = FrechetInceptionDistance(
        feature=2048, normalize=False, reset_real_features=False).to(device)
    kid = KernelInceptionDistance(
        feature=2048, normalize=False, reset_real_features=False,
        subsets=kid_subsets, subset_size=kid_subset_size).to(device)
    return fid, kid

def make_pixel_metrics(lpips_net, device):
    return (
        LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=True).to(device),
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-split computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_split(split, real_dir, fake_dir, args, device):
    print(f"\n{'━'*62}")
    print(f"  SPLIT : {split.upper()}")
    print(f"  real  : {real_dir}")
    print(f"  fake  : {fake_dir}")
    print(f"{'━'*62}")

    # Loaders
    p_real_loader, p_fake_loader, n_pairs = build_paired_loaders(
        real_dir, fake_dir, args.batch_size, args.num_workers, args.img_size, args.num)
    u_real_loader, u_fake_loader, n_unpaired = build_unpaired_loaders(
        real_dir, fake_dir, args.batch_size, args.num_workers, args.img_size, args.seed, args.num)

    kid_p_sub = min(args.kid_subset_size, n_pairs)
    kid_u_sub = min(args.kid_subset_size, n_unpaired)

    # Metrics
    fid_p, kid_p = make_fid_kid(args.kid_subsets, kid_p_sub, device)
    fid_u, kid_u = make_fid_kid(args.kid_subsets, kid_u_sub, device)
    lpips, ssim = make_pixel_metrics(args.lpips_net, device)

    # ── Paired pass ───────────────────────────────────────────────────────────
    print(f"\n  [Paired pass — {n_pairs} pairs]")
    for rb, fb in tqdm(zip(p_real_loader, p_fake_loader),
                       total=len(p_real_loader), leave=True):
        ru8, fu8 = rb["uint8"].to(device), fb["uint8"].to(device)
        fid_p.update(ru8, real=True);  fid_p.update(fu8, real=False)
        kid_p.update(ru8, real=True);  kid_p.update(fu8, real=False)

        rf, ff = rb["float"].to(device), fb["float"].to(device)
        lpips.update(ff, rf)
        ssim.update(ff, rf)

    # ── Unpaired pass ─────────────────────────────────────────────────────────
    print(f"\n  [Unpaired pass — {n_unpaired} images each]")
    print("    encoding real ...")
    for batch in tqdm(u_real_loader, leave=True):
        b = batch.to(device)
        fid_u.update(b, real=True);  kid_u.update(b, real=True)
    print("    encoding fake ...")
    for batch in tqdm(u_fake_loader, leave=True):
        b = batch.to(device)
        fid_u.update(b, real=False); kid_u.update(b, real=False)

    # ── Results ───────────────────────────────────────────────────────────────
    fid_p_val = fid_p.compute().item()
    kid_p_m, kid_p_s = [x.item() for x in kid_p.compute()]
    fid_u_val = fid_u.compute().item()
    kid_u_m, kid_u_s = [x.item() for x in kid_u.compute()]
    lpips_val = lpips.compute().item()
    ssim_val  = ssim.compute().item()

    return dict(
        fid_paired=fid_p_val,   kid_paired_mean=kid_p_m,   kid_paired_std=kid_p_s,
        fid_unpaired=fid_u_val, kid_unpaired_mean=kid_u_m, kid_unpaired_std=kid_u_s,
        lpips=lpips_val, ssim=ssim_val,
        n_pairs=n_pairs, n_unpaired=n_unpaired,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-split result printer
# ──────────────────────────────────────────────────────────────────────────────

def print_split_results(split, r):
    print("\n" + "═"*55)
    print(f"  RESULTS — {split.upper()}")
    print("═"*55)
    print(f"  {'Paired images':<24}: {r['n_pairs']}")
    print(f"  {'Unpaired images':<24}: {r['n_unpaired']}")
    print("  " + "─"*51)
    print(f"  {'FID (paired)':<24}: {r['fid_paired']:.4f}         (lower↓)")
    print(f"  {'FID (unpaired)':<24}: {r['fid_unpaired']:.4f}         (lower↓)")
    print(f"  {'KID mean (paired)':<24}: {r['kid_paired_mean']:.6f}   (lower↓)")
    print(f"  {'KID std  (paired)':<24}: {r['kid_paired_std']:.6f}")
    print(f"  {'KID mean (unpaired)':<24}: {r['kid_unpaired_mean']:.6f}   (lower↓)")
    print(f"  {'KID std  (unpaired)':<24}: {r['kid_unpaired_std']:.6f}")
    print("  " + "─"*51)
    print(f"  {'LPIPS':<24}: {r['lpips']:.4f}         (lower↓)")
    print(f"  {'SSIM':<24}: {r['ssim']:.4f}         (higher↑)")
    print("═"*55)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for a provided real/fake directory pair."
    )
    parser.add_argument("--real_dir",        type=str, required=True,
                        help="Directory containing ground-truth images.")
    parser.add_argument("--fake_dir",        type=str, required=True,
                        help="Directory containing generated images.")
    parser.add_argument("--split_name",      type=str, default="custom",
                        help="Label used in the results header.")
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--img_size",        type=int, default=256)
    parser.add_argument("--num",             type=int, default=None,
                        help="Max number of images/pairs to use (cap).")
    parser.add_argument("--kid_subsets",     type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--lpips_net",       type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--device",          type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"\nDevice : {device}")

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    if not real_dir.exists():
        raise FileNotFoundError(f"real_dir not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"fake_dir not found: {fake_dir}")

    results = compute_split(args.split_name, real_dir, fake_dir, args, device)
    print_split_results(args.split_name, results)


if __name__ == "__main__":
    main()
