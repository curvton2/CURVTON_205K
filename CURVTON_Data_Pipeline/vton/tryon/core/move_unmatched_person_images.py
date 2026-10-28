"""
Cleanup Utility: Rename Test Cloth Images + Move Unmatched Person Images
=========================================================================

This script performs TWO sequential operations:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — Rename cloth images in dataset_ultimate_test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cloth images in dataset_ultimate_test arrive with the full garment name
appended after the cloth-code (mc_XXXXXX), e.g.:
    mh_000011_e01_mc_005764_gat.png
    mh_000018_e03_mc_009497_boubou.png
    mh_000021_e02_mc_017002_Lavalava_(men's).png
    mh_000031_e01_mc_010751_hanbok.png

These are renamed so the garment name is replaced by one of three canonical
category labels — _upper, _lower, or _dress:
    mh_000011_e01_mc_005764_upper.png
    mh_000018_e03_mc_009497_upper.png
    mh_000021_e02_mc_017002_upper.png
    mh_000031_e01_mc_010751_upper.png

HOW _upper / _lower / _dress IS DECIDED
----------------------------------------
Every garment in dress_info.py has a "dress_type" field.
The mapping is:

    dress_type contains "upperbody"  →  suffix = _upper
        Examples: upperbody inner dress, upperbody flex dress
        Garments: t-shirts, shirts, blouses, jackets, coats, sweaters, tunics

    dress_type contains "lowerbody"  →  suffix = _lower
        Examples: lowerbody dress
        Garments: trousers, skirts, wraps secured at the waist

    dress_type contains "fullbody"   →  suffix = _dress
        Examples: fullbody dress, fullbody flex dress
        Garments: sarees, kimonos, gowns, robes, full-length cloaks,
                  garments that cover both torso and legs as one piece

    No match found in dress_info  →  suffix = _upper  (safe fallback,
                                                            logged as [NO-MATCH])

The garment name fragment is matched against dress_info using:
    1. Exact match (after normalising spaces/punctuation → underscores)
    2. Suffix match  (fragment ends with the known garment name)
    3. Prefix match  (partial / truncated name)
    4. Fallback → _upper

Phase 1 runs BEFORE Phase 2 so the cloth_image directory already contains
clean _upper/_lower/_dress filenames when the person-prefix matching runs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — Move unmatched person images
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For both dataset_ultimate and dataset_ultimate_test, within every
{diff}/{gender}/ subdirectory, each person image in initial_person_image/
is checked against the cloth images in cloth_image/.

The person prefix embedded in a cloth filename is its first 3
underscore-separated tokens:
    mh_000001_e01_mc_005764_upper.png  →  person prefix = mh_000001_e01
    fh_000042_e02_mc_009497_lower.png  →  person prefix = fh_000042_e02

If a person image stem does NOT match any cloth-image person prefix it means
no cloth was paired with that person, so it is moved out:

    dataset_ultimate      →  <dataset_v3_dir>/person_images/{diff}/{gender}/
    dataset_ultimate_test →  <backup_dir>/person_images_test/{diff}/{gender}/

Covers all 6 split combinations: easy/medium/hard × male/female.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage:
    # dry run — no files are touched, only prints what would happen
    python vton/tryon/core/move_unmatched_person_images.py \
        --dataset_v3_dir PATH_TO_DATASET_V3 \
        --backup_dir PATH_TO_BACKUP_DIR \
        --dry_run

    # live run
    python vton/tryon/core/move_unmatched_person_images.py \
        --dataset_v3_dir PATH_TO_DATASET_V3 \
        --backup_dir PATH_TO_BACKUP_DIR

  Internally derived paths:
    --backup_dir/dataset_ultimate_test  →  source for Phase 1 rename + Phase 2 move
    --backup_dir/person_images_test/    →  destination for unmatched person images
"""

import argparse
import os
import re
import shutil
import sys

DIFFS   = ["easy", "medium", "hard"]
GENDERS = ["male", "female"]

# ── DRESS_INFO lookup (for Phase 1) ─────────────────────────────────────────
VTON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(VTON_ROOT, "tryon"))

try:
    from dress_info import DRESS_INFO
except ImportError:
    print("CRITICAL: dress_info.py not found.")
    sys.exit(1)

_PUNCT_RE = re.compile(r"[\s\-\(\)\'\"]+")
_MULTI_US = re.compile(r"_+")
_MC_RE    = re.compile(r"^(.+_mc_\d+)_(.+)$", re.IGNORECASE)


def _normalise(text: str) -> str:
    t = _PUNCT_RE.sub("_", text.lower())
    return _MULTI_US.sub("_", t).strip("_")


def _get_base_type(dress_type: str) -> str:
    dt = dress_type.lower()
    if "upperbody" in dt:
        return "upper"
    elif "lowerbody" in dt:
        return "lower"
    elif "fullbody" in dt:
        return "dress"
    return "upper"


# (normalised_name, base_type) sorted longest-first for greedy matching
_DRESS_LOOKUP = sorted(
    [(_normalise(d["name"]), _get_base_type(d["dress_type"])) for d in DRESS_INFO],
    key=lambda x: len(x[0]),
    reverse=True,
)


def _resolve_type(dress_name_raw: str) -> tuple[str, bool]:
    """Return (base_type, matched). matched=False means fallback to 'upper'."""
    norm = _normalise(dress_name_raw)
    for d_norm, d_type in _DRESS_LOOKUP:
        if norm == d_norm or norm.endswith(d_norm):
            return d_type, True
    for d_norm, d_type in _DRESS_LOOKUP:
        if d_norm.startswith(norm) or norm.startswith(d_norm):
            return d_type, True
    return "upper", False


def _collect_cloth_dirs(root: str) -> list[str]:
    """Return all cloth_image/ dirs under root (canonical structure first, walk fallback)."""
    dirs: list[str] = []
    for diff in DIFFS:
        for gender in GENDERS:
            p = os.path.join(root, diff, gender, "cloth_image")
            if os.path.isdir(p):
                dirs.append(p)
    if dirs:
        return dirs
    for dirpath, subdirs, _ in os.walk(root):
        for d in subdirs:
            if d == "cloth_image":
                dirs.append(os.path.join(dirpath, d))
    return dirs


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Rename test cloth images
# ════════════════════════════════════════════════════════════════════════════

def rename_test_cloth_images(test_dir: str, dry_run: bool = False) -> None:
    """
    Rename cloth images in dataset_ultimate_test from garment-name suffixes
    to _upper / _lower / _dress.
    """
    if not os.path.isdir(test_dir):
        print(f"[SKIP] test_dir not found: {test_dir}")
        return

    cloth_dirs = _collect_cloth_dirs(test_dir)
    if not cloth_dirs:
        print(f"[SKIP] No cloth_image/ directories found under: {test_dir}")
        return

    print(f"\n{'='*60}")
    print(f"PHASE 1: RENAME TEST CLOTH IMAGES {'(DRY RUN) ' if dry_run else ''}")
    print(f"{'='*60}")

    total_renamed  = 0
    total_skipped  = 0
    total_conflict = 0
    total_no_match = 0

    for cloth_dir in cloth_dirs:
        rel = os.path.relpath(cloth_dir, test_dir)
        print(f"\n  {rel}")

        for fname in sorted(os.listdir(cloth_dir)):
            if not fname.lower().endswith(".png"):
                continue

            stem = os.path.splitext(fname)[0]

            # Already has canonical suffix → skip
            if stem.lower().endswith(("_upper", "_lower", "_dress")):
                total_skipped += 1
                continue

            m = _MC_RE.match(stem)
            if not m:
                print(f"    [SKIP-NOPARSE] {fname}")
                total_skipped += 1
                continue

            base_part      = m.group(1)   # e.g. mh_000011_e01_mc_005764
            dress_name_raw = m.group(2)   # e.g. gat  /  Lavalava_(men's)

            base_type, matched = _resolve_type(dress_name_raw)
            if not matched:
                total_no_match += 1
                print(f"    [NO-MATCH→upper] {fname}  (fragment: '{dress_name_raw}')")

            new_fname = f"{base_part}_{base_type}.png"
            if new_fname == fname:
                total_skipped += 1
                continue

            old_path = os.path.join(cloth_dir, fname)
            new_path = os.path.join(cloth_dir, new_fname)

            if os.path.exists(new_path):
                print(f"    [CONFLICT] {new_fname} already exists, skipping")
                total_conflict += 1
                continue

            if dry_run:
                print(f"    [DRY RUN]  {fname}  →  {new_fname}")
            else:
                os.rename(old_path, new_path)
                print(f"    [RENAMED]  {fname}  →  {new_fname}")

            total_renamed += 1

    print(f"\n  Renamed                     : {total_renamed}")
    print(f"  Already correct / skipped   : {total_skipped}")
    print(f"  Conflicts (target existed)  : {total_conflict}")
    print(f"  No DRESS_INFO match (→upper): {total_no_match}")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Move unmatched person images
# ════════════════════════════════════════════════════════════════════════════

def _person_prefix_from_cloth(cloth_stem: str) -> str:
    """First 3 underscore tokens of a cloth stem → person prefix."""
    parts = cloth_stem.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return cloth_stem


def _process_one_dataset(
    src_root: str,
    dest_root: str,
    dataset_label: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Move unmatched person images for a single dataset root."""
    total_moved   = 0
    total_matched = 0
    total_missing = 0

    for diff in DIFFS:
        for gender in GENDERS:
            person_dir = os.path.join(src_root, diff, gender, "initial_person_image")
            cloth_dir  = os.path.join(src_root, diff, gender, "cloth_image")

            if not os.path.exists(person_dir):
                print(f"  [SKIP] Person dir missing: {diff}/{gender}/initial_person_image")
                total_missing += 1
                continue
            if not os.path.exists(cloth_dir):
                print(f"  [SKIP] Cloth dir missing: {diff}/{gender}/cloth_image")
                total_missing += 1
                continue

            # Build prefix set from cloth images
            cloth_prefixes: set[str] = set()
            for cloth_fname in os.listdir(cloth_dir):
                if not cloth_fname.lower().endswith(".png"):
                    continue
                cloth_stem = os.path.splitext(cloth_fname)[0]
                cloth_prefixes.add(_person_prefix_from_cloth(cloth_stem))

            dest_dir = os.path.join(dest_root, diff, gender)

            for person_fname in sorted(os.listdir(person_dir)):
                if not person_fname.lower().endswith(".png"):
                    continue

                person_stem = os.path.splitext(person_fname)[0]

                if person_stem in cloth_prefixes:
                    total_matched += 1
                    continue

                src_path  = os.path.join(person_dir, person_fname)
                dest_path = os.path.join(dest_dir, person_fname)

                if dry_run:
                    print(f"  [DRY RUN] MOVE  {dataset_label}/{diff}/{gender}/"
                          f"initial_person_image/{person_fname}  →  "
                          f"{os.path.relpath(dest_dir, os.path.dirname(dest_root))}/{person_fname}")
                else:
                    os.makedirs(dest_dir, exist_ok=True)
                    if os.path.exists(dest_path):
                        print(f"  [SKIP-CONFLICT] Already exists: {dest_path}")
                        continue
                    shutil.move(src_path, dest_path)
                    print(f"  [MOVED] {dataset_label}/{diff}/{gender}/"
                          f"initial_person_image/{person_fname}  →  {dest_path}")

                total_moved += 1

    return total_moved, total_matched, total_missing


def move_unmatched_person_images(
    dataset_v3_dir: str,
    test_dir: str,
    dry_run: bool = False,
) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE 2: MOVE UNMATCHED PERSON IMAGES {'(DRY RUN) ' if dry_run else ''}")
    print(f"{'='*60}")

    jobs = [
        # (src_root, dest_root, label)
        (
            os.path.join(dataset_v3_dir, "dataset_ultimate"),
            os.path.join(dataset_v3_dir, "person_images"),
            "dataset_ultimate",
        ),
        (
            test_dir,
            os.path.join(os.path.dirname(test_dir), "person_images_test"),
            "dataset_ultimate_test",
        ),
    ]

    grand_moved = grand_matched = grand_missing = 0

    for src_root, dest_root, label in jobs:
        if not os.path.isdir(src_root):
            print(f"\n[SKIP] Not found: {src_root}")
            continue

        print(f"\n{'─'*60}")
        print(f"  {label}  →  {dest_root}")
        print(f"{'─'*60}")

        moved, matched, missing = _process_one_dataset(src_root, dest_root, label, dry_run)
        grand_moved   += moved
        grand_matched += matched
        grand_missing += missing

    print(f"\n  Matched (kept)                  : {grand_matched}")
    print(f"  Unmatched (moved / would move)  : {grand_moved}")
    print(f"  Missing dirs skipped            : {grand_missing}")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: rename cloth images in backup/dataset_ultimate_test to "
            "_upper/_lower/_dress.  "
            "Phase 2: move unmatched person images out of dataset_ultimate "
            "and backup/dataset_ultimate_test."
        )
    )
    parser.add_argument(
        "--dataset_v3_dir",
        type=str,
        required=True,
        help="Root dir containing dataset_ultimate/ "
             "(e.g. PATH_TO_DATASET_V3)",
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        required=True,
        help="Backup directory that contains dataset_ultimate_test/ "
             "(e.g. PATH_TO_BACKUP_DIR). "
             "Unmatched person images will be moved to backup_dir/person_images_test/.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen without making any changes.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_v3_dir):
        print(f"ERROR: dataset_v3_dir not found: {args.dataset_v3_dir}")
        sys.exit(1)

    if not os.path.isdir(args.backup_dir):
        print(f"ERROR: backup_dir not found: {args.backup_dir}")
        sys.exit(1)

    # Derive the test dataset path from backup_dir
    test_dir = os.path.join(args.backup_dir, "dataset_ultimate_test")

    # Phase 1 — rename test cloth images first so Phase 2 sees clean names
    rename_test_cloth_images(test_dir, dry_run=args.dry_run)

    # Phase 2 — move unmatched person images (both datasets)
    move_unmatched_person_images(args.dataset_v3_dir, test_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
