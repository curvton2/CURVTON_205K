from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Run from CURVTON_Experiments/ so relative paths (runs/, ckpt/) match the training scripts.
EXPERIMENTS_DIR = ROOT.parents[2] / "CURVTON_Experiments"


EVAL_TARGETS = {
    "base": ROOT / "evaluate_base.py",
}


def _run(target: str, passthrough: list[str]) -> int:
    script = EVAL_TARGETS[target]
    if not script.exists():
        raise FileNotFoundError(f"Target script not found: {script}")
    cmd = [sys.executable, str(script)] + passthrough
    print(f"[unified_eval] target={target}")
    print(f"[unified_eval] exec: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(EXPERIMENTS_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluation entrypoint for trained CURVTON_Experiments checkpoints.\n"
            "Use --target to select evaluator and pass remaining args through to that evaluator script."
        )
    )
    parser.add_argument("--target", required=True, choices=sorted(EVAL_TARGETS.keys()))
    parser.add_argument(
        "--profile",
        default=None,
        choices=["all_datasets", "baseline", "curvton_only", "curvton_splits", "curvton_extra"],
        help="Optional metadata-only label for run organization.",
    )
    args, passthrough = parser.parse_known_args()
    if args.profile:
        print(f"[unified_eval] profile={args.profile}")
    return _run(args.target, passthrough)


if __name__ == "__main__":
    sys.exit(main())

