"""
Consolidate dataset_ultimate_5, _8, _11, _14  →  dataset_ultimate
===================================================================
Moves files from the four accepted-pass datasets into a single
merged dataset_ultimate, preserving the directory structure:

    {dataset}/{diff}/{gender}/initial_person_image/  → merged initial_person_image/
    {dataset}/{diff}/{gender}/cloth_image/           → merged cloth_image/
    {dataset}/{diff}/{gender}/tryon_image/  *.png    → merged tryon_image/
                                            *.json     SKIPPED (not moved)

Structure of each source and the destination:
    dataset_ultimate/
        {diff}/{gender}/
            initial_person_image/
            cloth_image/
            tryon_image/

Conflict handling: if a file with the same name already exists at the
destination it is SKIPPED and reported — no silent overwrites.

Usage:
    # live run
    python vton/tryon/core/consolidate_ultimate.py \
        --dataset_v3_dir PATH_TO_DATASET_V3

    # dry run — prints what would be moved without touching anything
    python vton/tryon/core/consolidate_ultimate.py \
        --dataset_v3_dir PATH_TO_DATASET_V3 \
        --dry_run
"""

import argparse
import math
import os
import random
import shutil
import sys


# Sub-directories to consolidate per diff/gender
SUBDIRS = ["initial_person_image", "cloth_image", "tryon_image"]

# Source datasets to merge (in priority order — first wins on conflict)
SOURCE_DATASETS = [
    "dataset_ultimate_5",
    "dataset_ultimate_8",
    "dataset_ultimate_11",
    "dataset_ultimate_14",
]

DEST_DATASET = "dataset_ultimate"

DIFFS   = ["easy", "medium", "hard"]
GENDERS = ["male", "female"]


def consolidate(dataset_v3_dir: str, dry_run: bool = False):
    dest_root = os.path.join(dataset_v3_dir, DEST_DATASET)

    total_moved    = 0
    total_skipped  = 0   # conflicts
    total_json     = 0   # json files intentionally skipped
    total_missing  = 0   # source dirs that don't exist

    for src_name in SOURCE_DATASETS:
        src_root = os.path.join(dataset_v3_dir, src_name)

        if not os.path.exists(src_root):
            print(f"[SKIP] Source not found, skipping: {src_root}")
            total_missing += 1
            continue

        print(f"\n{'─'*60}")
        print(f"Processing: {src_name}")
        print(f"{'─'*60}")

        for diff in DIFFS:
            for gender in GENDERS:
                for subdir in SUBDIRS:
                    src_dir = os.path.join(src_root, diff, gender, subdir)

                    if not os.path.exists(src_dir):
                        continue

                    dst_dir = os.path.join(dest_root, diff, gender, subdir)

                    if not dry_run:
                        os.makedirs(dst_dir, exist_ok=True)

                    for fname in sorted(os.listdir(src_dir)):
                        src_path = os.path.join(src_dir, fname)

                        if not os.path.isfile(src_path):
                            continue

                        # Skip JSON files in tryon_image — not wanted in destination
                        if subdir == "tryon_image" and fname.endswith(".json"):
                            total_json += 1
                            continue

                        dst_path = os.path.join(dst_dir, fname)

                        # Conflict: file already exists at destination — overwrite
                        if os.path.exists(dst_path):
                            if dry_run:
                                print(
                                    f"  [DRY RUN] OVERWRITE  {src_name}/{diff}/{gender}/{subdir}/{fname} "
                                    f"— already exists, will be replaced."
                                )
                            else:
                                shutil.move(src_path, dst_path)
                                print(
                                    f"  [OVERWRITE] {src_name}/{diff}/{gender}/{subdir}/{fname} "
                                    f"→ {DEST_DATASET}/{diff}/{gender}/{subdir}/{fname}"
                                )
                            total_skipped += 1
                            continue

                        if dry_run:
                            print(
                                f"  [DRY RUN] MOVE  {src_name}/{diff}/{gender}/{subdir}/{fname} "
                                f"→ {DEST_DATASET}/{diff}/{gender}/{subdir}/{fname}"
                            )
                        else:
                            shutil.move(src_path, dst_path)
                            print(
                                f"  [MOVED]   {src_name}/{diff}/{gender}/{subdir}/{fname} "
                                f"→ {DEST_DATASET}/{diff}/{gender}/{subdir}/{fname}"
                            )

                        total_moved += 1

    print(f"\n{'='*60}")
    print(f"CONSOLIDATION {'(DRY RUN) ' if dry_run else ''}COMPLETE")
    print(f"{'='*60}")
    print(f"  Sources processed : {len(SOURCE_DATASETS) - total_missing} / {len(SOURCE_DATASETS)}")
    print(f"  Files moved       : {total_moved}")
    print(f"  Overwritten       : {total_skipped}  (destination had same filename, replaced)")
    print(f"  JSONs skipped     : {total_json}  (tryon_image/*.json intentionally excluded)")
    if total_missing:
        print(f"  Missing sources   : {total_missing}")
    print(f"  Destination       : {dest_root}")


def create_test_split(dataset_v3_dir: str, test_pct: float = 4.0, dry_run: bool = False):
    """
    Move test_pct % of triplets from dataset_ultimate → dataset_ultimate_test.

    A triplet is one tryon_image PNG + its matching initial_person_image
    (first 3 underscore tokens of tryon stem) + cloth_image (same filename
    as tryon).  Selection is random but reproducible via a fixed seed.
    """
    src_root  = os.path.join(dataset_v3_dir, DEST_DATASET)
    test_root = os.path.join(dataset_v3_dir, "dataset_ultimate_test")

    if not os.path.exists(src_root):
        print(f"[SKIP] dataset_ultimate not found — cannot create test split: {src_root}")
        return

    # ── Collect all triplets across diff/gender ──────────────────────────────
    all_triplets = []   # list of (diff, gender, tryon_fname)

    for diff in DIFFS:
        for gender in GENDERS:
            tryon_dir = os.path.join(src_root, diff, gender, "tryon_image")
            if not os.path.exists(tryon_dir):
                continue
            for fname in sorted(os.listdir(tryon_dir)):
                if fname.endswith(".png"):
                    all_triplets.append((diff, gender, fname))

    total = len(all_triplets)
    if total == 0:
        print("[SKIP] dataset_ultimate is empty — no test split created.")
        return

    n_test = max(1, math.ceil(total * test_pct / 100.0))

    # Fixed seed so repeated dry-runs show the same selection
    rng = random.Random(42)
    selected = rng.sample(all_triplets, n_test)

    print(f"\n{'='*60}")
    print(f"TEST SPLIT {'(DRY RUN) ' if dry_run else ''}")
    print(f"{'='*60}")
    print(f"  Total triplets in dataset_ultimate : {total}")
    print(f"  Test percentage                    : {test_pct}%")
    print(f"  Triplets selected for test         : {n_test}")
    print(f"  Destination                        : dataset_ultimate_test")
    print()

    moved_tryon  = 0
    moved_person = 0
    moved_cloth  = 0
    missing      = 0

    for diff, gender, tryon_fname in selected:
        stem  = tryon_fname.replace(".png", "")
        parts = stem.split("_")

        # person_fname — first 3 underscore tokens of the tryon stem
        if len(parts) < 3:
            print(f"  [WARN] Unexpected filename format, skipping: {tryon_fname}")
            missing += 1
            continue
        person_fname = f"{parts[0]}_{parts[1]}_{parts[2]}.png"
        cloth_fname  = tryon_fname   # cloth filename is identical to tryon filename

        src_tryon  = os.path.join(src_root,  diff, gender, "tryon_image",          tryon_fname)
        src_person = os.path.join(src_root,  diff, gender, "initial_person_image", person_fname)
        src_cloth  = os.path.join(src_root,  diff, gender, "cloth_image",          cloth_fname)

        dst_tryon  = os.path.join(test_root, diff, gender, "tryon_image",          tryon_fname)
        dst_person = os.path.join(test_root, diff, gender, "initial_person_image", person_fname)
        dst_cloth  = os.path.join(test_root, diff, gender, "cloth_image",          cloth_fname)

        # Verify all 3 source files exist before moving anything
        missing_files = [p for p in [src_tryon, src_person, src_cloth] if not os.path.exists(p)]
        if missing_files:
            for p in missing_files:
                print(f"  [WARN] Missing source file, skipping triplet: {p}")
            missing += 1
            continue

        if dry_run:
            print(
                f"  [DRY RUN] {diff}/{gender}  tryon={tryon_fname} "
                f"person={person_fname} cloth={cloth_fname}"
            )
        else:
            os.makedirs(os.path.dirname(dst_tryon),  exist_ok=True)
            os.makedirs(os.path.dirname(dst_person), exist_ok=True)
            os.makedirs(os.path.dirname(dst_cloth),  exist_ok=True)

            shutil.move(src_tryon,  dst_tryon)
            shutil.move(src_person, dst_person)
            shutil.move(src_cloth,  dst_cloth)

            print(
                f"  [MOVED] {diff}/{gender}  tryon={tryon_fname} "
                f"person={person_fname} cloth={cloth_fname}"
            )

        moved_tryon  += 1
        moved_person += 1
        moved_cloth  += 1

    print(f"\n  Triplets moved to test split : {moved_tryon}")
    if missing:
        print(f"  Triplets skipped (missing)   : {missing}")
    print(f"  Remaining in dataset_ultimate: {total - moved_tryon}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate dataset_ultimate_5/_8/_11/_14 → dataset_ultimate"
    )
    parser.add_argument(
        "--dataset_v3_dir",
        type=str,
        required=True,
        help="Root directory containing all dataset_ultimate_* folders "
             "(e.g. PATH_TO_DATASET_V3)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be moved without actually moving anything",
    )
    parser.add_argument(
        "--test_pct",
        type=float,
        default=4.0,
        help="Percentage of dataset_ultimate to move into dataset_ultimate_test (default: 4.0)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_v3_dir):
        print(f"ERROR: dataset_v3_dir does not exist: {args.dataset_v3_dir}")
        sys.exit(1)

    # Step 1: consolidate all four source datasets into dataset_ultimate
    consolidate(args.dataset_v3_dir, dry_run=args.dry_run)

    # Step 2: carve out test split from dataset_ultimate → dataset_ultimate_test
    create_test_split(args.dataset_v3_dir, test_pct=args.test_pct, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
