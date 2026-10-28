"""
sample_prompts.py
=================
Utility for sampling prompts from camera_control.json.

Can be used:
  1. As a standalone CLI — preview samples from the library
  2. As an imported module — call sample() from setup_and_run.py or any script

USAGE (CLI)
-----------
    # Sample 10 prompts from each difficulty
    python sample_prompts.py

    # Sample 5 easy prompts
    python sample_prompts.py --difficulty easy --n 5

    # Sample from a specific category only
    python sample_prompts.py --difficulty hard --category three_way_combinations

    # List all available categories per difficulty
    python sample_prompts.py --list

    # Sample 100 prompts from hard (with replacement if needed)
    python sample_prompts.py --difficulty hard --n 100 --replace

USAGE (import)
--------------
    from sample_prompts import load_library, sample, sample_flat

    lib = load_library()                          # loads camera_control.json

    # Sample 10 prompts from easy (no replacement, errors if n > pool)
    prompts = sample(lib, "easy", n=10)

    # Sample 100 prompts from hard (with replacement to fill n)
    prompts = sample(lib, "hard", n=100, replace=True)

    # Sample from one specific category
    prompts = sample(lib, "medium", n=5, category="rotation_and_zoom")

    # Get the flat list of all prompts for a difficulty (no sampling)
    all_hard = sample_flat(lib, "hard")
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_JSON = Path(__file__).resolve().parents[2] / "CURVTON_assets" / "dictionaries" / "camera_control.json"


# =============================================================================
# Core functions
# =============================================================================

def load_library(path: Path = DEFAULT_JSON) -> Dict:
    """Load camera_control.json and return the parsed dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_flat(lib: Dict, difficulty: str) -> List[str]:
    """
    Return a flat list of ALL prompts for a difficulty level,
    across all categories.
    """
    if difficulty not in lib:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from: {list(lib.keys())}")
    prompts_block = lib[difficulty].get("prompts", {})
    flat = []
    for category_prompts in prompts_block.values():
        flat.extend(category_prompts)
    return flat


def sample(
    lib:        Dict,
    difficulty: str,
    n:          int               = 10,
    category:   Optional[str]     = None,
    replace:    bool              = False,
    seed:       Optional[int]     = None,
) -> List[str]:
    """
    Sample n prompts from a difficulty level (or a specific category within it).

    Args:
        lib        : loaded library dict from load_library()
        difficulty : "easy", "medium", or "hard"
        n          : number of prompts to return
        category   : if set, sample only from this sub-category
                     (e.g. "rotation_only", "three_way_combinations")
        replace    : if True, sample with replacement (allows n > pool size)
                     if False and n > pool size, raises ValueError
        seed       : random seed for reproducibility

    Returns:
        List of n prompt strings
    """
    rng = random.Random(seed)

    if difficulty not in lib:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Available: {[k for k in lib if k != 'metadata']}"
        )

    prompts_block = lib[difficulty].get("prompts", {})

    if category is not None:
        if category not in prompts_block:
            available = list(prompts_block.keys())
            raise ValueError(
                f"Unknown category '{category}' for difficulty '{difficulty}'. "
                f"Available: {available}"
            )
        pool = prompts_block[category]
    else:
        # Flatten all categories
        pool = []
        for cat_prompts in prompts_block.values():
            pool.extend(cat_prompts)

    # Deduplicate while preserving order
    pool = list(dict.fromkeys(pool))

    if not replace:
        if n > len(pool):
            raise ValueError(
                f"Requested n={n} but pool only has {len(pool)} unique prompts "
                f"for difficulty='{difficulty}'"
                + (f", category='{category}'" if category else "")
                + f". Use replace=True to allow repeats."
            )
        return rng.sample(pool, n)
    else:
        # With replacement: cycle+shuffle to fill n
        result, buf = [], pool[:]
        while len(result) < n:
            rng.shuffle(buf)
            result.extend(buf)
        return result[:n]


def list_categories(lib: Dict, difficulty: str) -> List[str]:
    """Return all category names for a difficulty level."""
    return list(lib[difficulty].get("prompts", {}).keys())


def pool_size(lib: Dict, difficulty: str, category: Optional[str] = None) -> int:
    """Return the number of unique prompts available."""
    return len(sample_flat(lib, difficulty) if category is None
               else lib[difficulty]["prompts"].get(category, []))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample prompts from camera_control.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json",       default=str(DEFAULT_JSON),
                   help="Path to camera_control.json.")
    p.add_argument("--difficulty", default=None,
                   choices=["easy", "medium", "hard"],
                   help="Difficulty to sample from. Omit to sample all three.")
    p.add_argument("--n",          type=int, default=10,
                   help="Number of prompts to sample.")
    p.add_argument("--category",   default=None,
                   help="Sample from a specific sub-category only.")
    p.add_argument("--replace",    action="store_true",
                   help="Sample with replacement (allows n > pool size).")
    p.add_argument("--seed",       type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--list",       action="store_true",
                   help="List all categories and pool sizes then exit.")
    return p.parse_args()


def main():
    args = parse_args()
    lib  = load_library(Path(args.json))

    diffs = (
        ["easy", "medium", "hard"]
        if args.difficulty is None
        else [args.difficulty]
    )

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        for diff in diffs:
            cats = list_categories(lib, diff)
            total = pool_size(lib, diff)
            print(f"\n{diff.upper()}  ({total} total prompts)")
            print(f"  {'Category':<35}  Prompts")
            print(f"  {'─'*35}  ───────")
            for cat in cats:
                n = pool_size(lib, diff, cat)
                print(f"  {cat:<35}  {n}")
        return

    # ── Sample mode ───────────────────────────────────────────────────────────
    for diff in diffs:
        total = pool_size(lib, diff)
        print(f"\n{'═'*60}")
        print(f"  {diff.upper()}  —  {total} prompts in pool")
        print(f"{'═'*60}")

        try:
            sampled = sample(
                lib,
                difficulty=diff,
                n=args.n,
                category=args.category,
                replace=args.replace,
                seed=args.seed,
            )
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        for i, prompt in enumerate(sampled, 1):
            print(f"  {i:3d}.  {prompt}")


if __name__ == "__main__":
    main()