from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from curvton_cache_autogen import (
    add_autogen_args,
    caches_requiring_generation,
    ensure_curvton_caches,
)


REQUIRED_SAMPLE_RATIO = 0.05


def _load_occ_map(npz_path: Path) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "occ_maps" not in data:
        raise KeyError(f"Missing 'occ_maps' in {npz_path}")
    occ_maps = data["occ_maps"]
    if occ_maps.ndim != 3:
        raise ValueError(f"Expected occ_maps with shape (N, H, W), got {occ_maps.shape}")
    return occ_maps


def _compute_mean_map(occ_maps: np.ndarray) -> np.ndarray:
    return occ_maps.mean(axis=0)


def _subsample_maps(occ_maps: np.ndarray, sample_ratio: float, seed: int) -> np.ndarray:
    if sample_ratio >= 1.0:
        return occ_maps
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = occ_maps.shape[0]
    if n <= 1:
        return occ_maps
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return occ_maps[idx]


def _apply_eccv_style() -> None:
    # Use the repository ECCV style settings when available.
    try:
        from EDA.plot_style import apply_paper_style

        apply_paper_style()
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
                "font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )


def _get_heatmap_cmap() -> str:
    try:
        from EDA.plot_style import get_cmap_for_heatmap

        return get_cmap_for_heatmap("hot")
    except Exception:
        return "inferno"


def plot_easy_medium_hard_heatmaps(
    maps_by_label: Dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
    vmin: float,
    vmax: float,
    show_colorbar: bool,
) -> Path:
    _apply_eccv_style()

    labels = list(maps_by_label.keys())
    mean_maps = [_compute_mean_map(maps_by_label[lbl]) for lbl in labels]

    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")

    fig, axes = plt.subplots(1, len(labels), figsize=(6.875, 2.5), dpi=150, constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]

    cmap = _get_heatmap_cmap()
    im = None

    for ax, label, mean_map in zip(axes, labels, mean_maps):
        im = ax.imshow(
            mean_map,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            origin="upper",
            interpolation="bilinear",
        )
        ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    if show_colorbar and im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("", fontsize=7)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ECCV-style spatial occlusion heatmap comparison for easy/medium/hard splits."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ file containing occ_maps for Easy split.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ file containing occ_maps for Medium split.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ file containing occ_maps for Hard split.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/occlusion_easy_medium_hard"))
    parser.add_argument("--stem", type=str, default="occlusion_heatmap_easy_medium_hard")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument(
        "--vmax",
        type=float,
        default=-1.0,
        help="Max value for color scaling. If negative, uses global max across splits.",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable the shared colorbar for a cleaner, text-free figure.",
    )
    add_autogen_args(parser)
    return parser.parse_args()


def _require_cache_files(paths: Tuple[Path, ...]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Missing cache files: "
            f"{missing_list}. Provide --curvton-root or pre-generate the caches."
        )


def main() -> int:
    args = parse_args()
    forced_ratio = REQUIRED_SAMPLE_RATIO
    if args.occ_maps_dir is None:
        raise ValueError(
            "--occ-maps-dir is mandatory. Previous mixed caches are disallowed for occlusion heatmaps."
        )

    auto_defaults = False
    if args.easy is None and args.medium is None and args.hard is None:
        pct = int(round(forced_ratio * 100))
        args.easy = args.occ_maps_dir / f"curvton_easy_{pct}pct_occ_maps.npz"
        args.medium = args.occ_maps_dir / f"curvton_medium_{pct}pct_occ_maps.npz"
        args.hard = args.occ_maps_dir / f"curvton_hard_{pct}pct_occ_maps.npz"
        auto_defaults = True

    missing_diffs = []
    if auto_defaults:
        # Check missing splits using separate occ_maps cache directory.
        pct = int(round(forced_ratio * 100))
        expected = {
            "easy": args.occ_maps_dir / f"curvton_easy_{pct}pct_occ_maps.npz",
            "medium": args.occ_maps_dir / f"curvton_medium_{pct}pct_occ_maps.npz",
            "hard": args.occ_maps_dir / f"curvton_hard_{pct}pct_occ_maps.npz",
        }
        for diff, p in expected.items():
            if not p.exists():
                missing_diffs.append(diff)
    if missing_diffs:
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["occ_maps"],
        )
    _require_cache_files((args.easy, args.medium, args.hard))

    maps_by_label = {
        "Easy": _load_occ_map(args.easy),
        "Medium": _load_occ_map(args.medium),
        "Hard": _load_occ_map(args.hard),
    }

    if not auto_defaults:
        maps_by_label = {
            k: _subsample_maps(v, forced_ratio, args.seed) for k, v in maps_by_label.items()
        }

    if args.vmax < 0:
        global_max = max(_compute_mean_map(m).max() for m in maps_by_label.values())
        vmax = float(global_max)
    else:
        vmax = float(args.vmax)

    plot_easy_medium_hard_heatmaps(
        maps_by_label=maps_by_label,
        out_dir=args.out_dir,
        stem=args.stem,
        vmin=float(args.vmin),
        vmax=vmax,
        show_colorbar=not args.no_colorbar,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
