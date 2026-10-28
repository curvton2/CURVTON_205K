"""
Delete JSON sidecar files from dataset_ultimate/tryon_image directories
=======================================================================
Removes all *.json files found under:
    dataset_ultimate/{diff}/{gender}/tryon_image/*.json

PNG files and all other subdirectories are left untouched.

Usage:
    # dry run — list what would be deleted without deleting anything
    python vton/tryon/core/delete_tryon_jsons.py \
        --dataset_v3_dir PATH_TO_DATASET_V3 \
        --dry_run

    # live run
    python vton/tryon/core/delete_tryon_jsons.py \
        --dataset_v3_dir PATH_TO_DATASET_V3
"""

import argparse
import os
import sys

DIFFS   = ["easy", "medium", "hard"]
GENDERS = ["male", "female"]


def delete_tryon_jsons(dataset_v3_dir: str, dry_run: bool = False):
    ultimate_root = os.path.join(dataset_v3_dir, "dataset_ultimate")

    if not os.path.exists(ultimate_root):
        print(f"ERROR: dataset_ultimate not found: {ultimate_root}")
        sys.exit(1)

    total_deleted = 0
    total_missing = 0

    for diff in DIFFS:
        for gender in GENDERS:
            tryon_dir = os.path.join(ultimate_root, diff, gender, "tryon_image")

            if not os.path.exists(tryon_dir):
                continue

            json_files = sorted(f for f in os.listdir(tryon_dir) if f.endswith(".json"))

            if not json_files:
                continue

            for fname in json_files:
                fpath = os.path.join(tryon_dir, fname)
                if dry_run:
                    print(f"  [DRY RUN] DELETE  {diff}/{gender}/tryon_image/{fname}")
                else:
                    os.remove(fpath)
                    print(f"  [DELETED] {diff}/{gender}/tryon_image/{fname}")
                total_deleted += 1

    print(f"\n{'='*60}")
    print(f"JSON CLEANUP {'(DRY RUN) ' if dry_run else ''}COMPLETE")
    print(f"{'='*60}")
    print(f"  JSON files {'would be deleted' if dry_run else 'deleted'} : {total_deleted}")
    print(f"  Location : {ultimate_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Delete *.json files from dataset_ultimate/*/tryon_image/"
    )
    parser.add_argument(
        "--dataset_v3_dir",
        type=str,
        required=True,
        help="Root directory containing dataset_ultimate/ "
             "(e.g. PATH_TO_DATASET_V3)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be deleted without actually deleting anything",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_v3_dir):
        print(f"ERROR: dataset_v3_dir does not exist: {args.dataset_v3_dir}")
        sys.exit(1)

    delete_tryon_jsons(args.dataset_v3_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
