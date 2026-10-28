"""Shared code for stratify_by_body_region.py and stratify_by_traditionality.py.

Input layout (CURVTON naming convention):

    <curvton_root>/{easy,medium,hard}/{male,female}/
        initial_person_image/  mh_000001_e01.png
        cloth_image/           mh_000001_e01_mc_000001_upper.png         (after final_cloth_rename.py)
                               mh_000001_e01_mc_000001_daura_suruwal.png (before it)
        tryon_image/           mh_000001_e01_mc_000001_daura_suruwal.png

Each try-on keeps the garment name in its file name. The garment is looked up in
DRESS_INFO (CURVTON_assets/dictionaries/dress_info.json, loaded through
CURVTON_Data_Pipeline/vton/tryon/core/dress_info.py) with the same longest-suffix
match the generation pipeline uses.

Output layout (one stratum per folder, read by zeroshot/vton_metrics/evaluate_vton.py):

    <output_root>/<stratum>/{male,female}/{initial_person_image,cloth_image,tryon_image}/
    <output_root>/stratification_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "CURVTON_Data_Pipeline" / "vton" / "tryon" / "core"))
from dress_info import DRESS_INFO  # noqa: E402

DIFFICULTIES = ("easy", "medium", "hard")
GENDERS = ("male", "female")
IMAGE_DIRS = ("initial_person_image", "cloth_image", "tryon_image")

# fh_017548_e01_fc_019460_qipao -> person "fh_017548_e01", pair key "fh_017548_e01_fc_019460", garment "qipao"
NAME_RE = re.compile(r"^(?P<person>[a-z]{2}_\d+_e\d+)_(?P<cloth>[a-z]{2}_?\d+)_(?P<garment>.+)$", re.IGNORECASE)

# Same matching as the generation pipeline: garment names with spaces -> underscores, longest suffix wins
_GARMENTS = sorted(
    ((g["name"].replace(" ", "_").lower(), g) for g in DRESS_INFO),
    key=lambda item: len(item[0]),
    reverse=True,
)


def match_garment(stem: str) -> dict | None:
    stem = stem.lower()
    for key, garment in _GARMENTS:
        if stem.endswith(key):
            return garment
    return None


def collect_triplets(curvton_root: Path) -> tuple[list[dict], Counter]:
    """Return one record per try-on with its person / cloth / try-on paths and DRESS_INFO garment."""
    rows, skipped = [], Counter()
    for diff in DIFFICULTIES:
        for gender in GENDERS:
            base = curvton_root / diff / gender
            tryon_dir = base / "tryon_image"
            if not tryon_dir.is_dir():
                continue
            persons = {p.name for p in (base / "initial_person_image").glob("*.png")}
            cloth_by_key = {}
            for c in (base / "cloth_image").glob("*.png"):
                m = NAME_RE.match(c.stem)
                if m:
                    cloth_by_key[f"{m['person']}_{m['cloth']}"] = c

            for tryon in sorted(tryon_dir.glob("*.png")):
                m = NAME_RE.match(tryon.stem)
                if not m:
                    skipped["unparseable_name"] += 1
                    continue
                person = f"{m['person']}.png"
                cloth = cloth_by_key.get(f"{m['person']}_{m['cloth']}")
                if person not in persons or cloth is None:
                    skipped["missing_person_or_cloth"] += 1
                    continue
                garment = match_garment(tryon.stem)
                if garment is None:
                    skipped["garment_not_in_dress_info"] += 1
                    continue
                rows.append({
                    "difficulty": diff,
                    "gender": gender,
                    "initial_person_image": base / "initial_person_image" / person,
                    "cloth_image": cloth,
                    "tryon_image": tryon,
                    "garment": garment,
                })
    return rows, skipped


def _place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def run(description: str, stratum_of: Callable[[dict], str], column: str) -> int:
    """CLI shared by the stratify_by_*.py scripts. `stratum_of(row)` names the output folder."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--curvton-root", type=Path, required=True,
                   help="CURVTON test root containing easy/ medium/ hard/.")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Folder that receives one sub-folder per stratum.")
    p.add_argument("--mode", choices=("copy", "symlink", "hardlink"), default="copy",
                   help="How images are placed in the stratified folders.")
    p.add_argument("--prefix-difficulty", action="store_true",
                   help="Prefix every file with its difficulty (easy_...) if names repeat across tiers.")
    args = p.parse_args()

    rows, skipped = collect_triplets(args.curvton_root)
    if not rows:
        print(f"No try-on triplets found under {args.curvton_root}")
        return 1

    # Files from easy/medium/hard land in one folder per stratum, so names must be unique
    seen: dict[tuple, str] = {}
    for r in rows:
        key = (r["gender"], r["tryon_image"].name)
        if key in seen and seen[key] != r["difficulty"] and not args.prefix_difficulty:
            print(f"{r['tryon_image'].name} ({r['gender']}) appears in both {seen[key]} and "
                  f"{r['difficulty']}; rerun with --prefix-difficulty.")
            return 1
        seen[key] = r["difficulty"]

    counts = Counter()
    for r in rows:
        r[column] = stratum_of(r)
        prefix = f"{r['difficulty']}_" if args.prefix_difficulty else ""
        for sub in IMAGE_DIRS:
            src = r[sub]
            _place(src, args.output_root / r[column] / r["gender"] / sub / f"{prefix}{src.name}", args.mode)
        counts[r[column]] += 1

    manifest = args.output_root / "stratification_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "gender", "tryon_image", "garment", "dress_type", column])
        for r in rows:
            w.writerow([r["difficulty"], r["gender"], r["tryon_image"].name, r["garment"]["name"],
                        r["garment"]["dress_type"], r[column]])

    print(f"Stratified {len(rows)} triplets by {column}: {args.curvton_root} -> {args.output_root}")
    for stratum, n in sorted(counts.items()):
        print(f"  {stratum:<16} {n}")
    for reason, n in skipped.items():
        print(f"  skipped ({reason}): {n}")
    print(f"Manifest: {manifest}")
    return 0
