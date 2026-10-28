"""
Move Unmatched Person Images — dataset_ultimate_test (backup)
==============================================================
Within  <backup_dir>/dataset_ultimate_test/{diff}/{gender}/
compares every person image in initial_person_image/ against the
person-prefix embedded in each cloth_image/ filename.

Cloth filename convention (after renaming):
    mh_000011_e01_mc_005764_upper.png
    fh_000042_e02_mc_009497_lower.png

Person prefix = first 3 underscore-separated tokens:
    mh_000011_e01_mc_005764_upper.png  →  prefix = mh_000011_e01

If a person image stem does NOT appear as a prefix in any cloth image
it is moved to:
    <backup_dir>/person_images_test/{diff}/{gender}/

Covers all 6 combinations: easy / medium / hard  ×  male / female.

Usage:
    # dry run — prints what would be moved, nothing is touched
    python vton/tryon/core/move_unmatched_person_images_test.py \\
           --backup_dir PATH_TO_BACKUP_DIR \\
        --dry_run

    # live run
    python vton/tryon/core/move_unmatched_person_images_test.py \\
           --backup_dir PATH_TO_BACKUP_DIR
"""

import argparse
import os
import shutil
import sys

DIFFS   = ["easy", "medium", "hard"]
GENDERS = ["male", "female"]


def person_prefix_from_cloth(cloth_stem: str) -> str:
    """Return the person prefix (first 3 underscore tokens) from a cloth stem."""
    parts = cloth_stem.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return cloth_stem


def move_unmatched_test(backup_dir: str, dry_run: bool = False) -> None:
    test_root = os.path.join(backup_dir, "dataset_ultimate_test")
    dest_root = os.path.join(backup_dir, "person_images_test")

    if not os.path.isdir(test_root):
        print(f"ERROR: dataset_ultimate_test not found: {test_root}")
        sys.exit(1)

    total_moved   = 0
    total_matched = 0
    total_missing = 0

    for diff in DIFFS:
        for gender in GENDERS:
            person_dir = os.path.join(test_root, diff, gender, "initial_person_image")
            cloth_dir  = os.path.join(test_root, diff, gender, "cloth_image")

            if not os.path.exists(person_dir):
                print(f"[SKIP] Person dir missing: {diff}/{gender}/initial_person_image")
                total_missing += 1
                continue

            if not os.path.exists(cloth_dir):
                print(f"[SKIP] Cloth dir missing: {diff}/{gender}/cloth_image")
                total_missing += 1
                continue

            # Build the set of person prefixes referenced by cloth images
            cloth_prefixes: set[str] = set()
            for cloth_fname in os.listdir(cloth_dir):
                if not cloth_fname.lower().endswith(".png"):
                    continue
                cloth_stem = os.path.splitext(cloth_fname)[0]
                cloth_prefixes.add(person_prefix_from_cloth(cloth_stem))

            dest_dir = os.path.join(dest_root, diff, gender)

            for person_fname in sorted(os.listdir(person_dir)):
                if not person_fname.lower().endswith(".png"):
                    continue

                person_stem = os.path.splitext(person_fname)[0]

                if person_stem in cloth_prefixes:
                    total_matched += 1
                    continue

                # Unmatched → move out
                src_path  = os.path.join(person_dir, person_fname)
                dest_path = os.path.join(dest_dir, person_fname)

                if dry_run:
                    print(f"  [DRY RUN] MOVE  dataset_ultimate_test/{diff}/{gender}/"
                          f"initial_person_image/{person_fname}"
                          f"  →  person_images_test/{diff}/{gender}/{person_fname}")
                else:
                    os.makedirs(dest_dir, exist_ok=True)
                    if os.path.exists(dest_path):
                        print(f"  [SKIP-CONFLICT] Already exists: {dest_path}")
                        continue
                    shutil.move(src_path, dest_path)
                    print(f"  [MOVED] dataset_ultimate_test/{diff}/{gender}/"
                          f"initial_person_image/{person_fname}"
                          f"  →  person_images_test/{diff}/{gender}/{person_fname}")

                total_moved += 1

    print(f"\n{'='*60}")
    print(f"MOVE UNMATCHED PERSON IMAGES (TEST) {'(DRY RUN) ' if dry_run else ''}COMPLETE")
    print(f"{'='*60}")
    print(f"  Source      : {test_root}")
    print(f"  Destination : {dest_root}")
    print(f"  Matched (kept)                  : {total_matched}")
    print(f"  Unmatched (moved / would move)  : {total_moved}")
    print(f"  Missing dirs skipped            : {total_missing}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Move unmatched person images from "
            "backup/dataset_ultimate_test/initial_person_image/ "
            "to backup/person_images_test/."
        )
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        required=True,
        help="Backup directory containing dataset_ultimate_test/ "
             "(e.g. PATH_TO_BACKUP_DIR)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be moved without making any changes.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.backup_dir):
        print(f"ERROR: backup_dir not found: {args.backup_dir}")
        sys.exit(1)

    move_unmatched_test(args.backup_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
