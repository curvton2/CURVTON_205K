from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
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


def _apply_eccv_style() -> None:
    try:
        from EDA.plot_style import apply_paper_style

        apply_paper_style()
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )


def _difficulty_colors() -> Dict[str, str]:
    try:
        from EDA.plot_style import CURVTON_COLORS

        return {
            "Easy": CURVTON_COLORS.get("Easy", "#1B9E77"),
            "Medium": CURVTON_COLORS.get("Medium", "#D95F02"),
            "Hard": CURVTON_COLORS.get("Hard", "#7570B3"),
        }
    except Exception:
        return {
            "Easy": "#1B9E77",
            "Medium": "#D95F02",
            "Hard": "#7570B3",
        }


def _load_lum_maps(npz_path: Path) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "lum_maps" not in data:
        raise KeyError(f"Missing lum_maps in {npz_path}")
    lum = data["lum_maps"].astype(np.float32)
    if lum.ndim != 3:
        raise ValueError(f"lum_maps expected shape (N, H, W), got {lum.shape}")
    return lum


def _subsample(lum: np.ndarray, sample_ratio: float, seed: int) -> np.ndarray:
    if sample_ratio >= 1.0:
        return lum
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = lum.shape[0]
    if n <= 1:
        return lum
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return lum[idx]


def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X)
        return Z, pca.explained_variance_ratio_
    except Exception:
        Xc = X - X.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:2].T
        var = (S**2) / max(Xc.shape[0] - 1, 1)
        ev = var / max(var.sum(), 1e-12)
        return Z, ev[:2]


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_illumination_pca_overlay(
    lum_by_label: Dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
) -> None:
    _apply_eccv_style()
    colors = _difficulty_colors()

    labels = ["Easy", "Medium", "Hard"]
    mats = []
    label_list = []

    for label in labels:
        lum = lum_by_label.get(label)
        if lum is None or lum.size == 0:
            continue
        mats.append(lum)
        label_list.extend([label] * lum.shape[0])

    if not mats:
        return

    X = np.concatenate(mats, axis=0)
    X = X.reshape(X.shape[0], -1)
    X = np.nan_to_num(X)

    # Standardize features to reduce scale bias.
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sig

    Z, ev = _pca_2d(X)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)

    labels_arr = np.asarray(label_list)
    for label in labels:
        mask = labels_arr == label
        if not np.any(mask):
            continue
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=10,
            alpha=0.55,
            color=colors[label],
            edgecolors="white",
            linewidths=0.2,
            label=label,
            rasterized=True,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style illumination PCA overlay for easy/medium/hard splits."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ with lum_maps for Easy.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ with lum_maps for Medium.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ with lum_maps for Hard.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/illumination_pca_overlay"))
    parser.add_argument("--stem", type=str, default="illumination_pca_overlay_easy_medium_hard")
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

    auto_defaults = False
    if args.easy is None and args.medium is None and args.hard is None:
        pct = int(round(forced_ratio * 100))
        args.easy = args.cache_dir / f"curvton_easy_{pct}pct.npz"
        args.medium = args.cache_dir / f"curvton_medium_{pct}pct.npz"
        args.hard = args.cache_dir / f"curvton_hard_{pct}pct.npz"
        auto_defaults = True

    if auto_defaults:
        missing_diffs = caches_requiring_generation(
            cache_dir=args.cache_dir,
            sample_ratio=forced_ratio,
            difficulties=["easy", "medium", "hard"],
            required_keys=["lum_maps"],
        )
    else:
        missing_diffs = []
    if missing_diffs:
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["lum_maps"],
        )
    _require_cache_files((args.easy, args.medium, args.hard))

    lum_by_label: Dict[str, np.ndarray] = {}
    for label, path in [
        ("Easy", args.easy),
        ("Medium", args.medium),
        ("Hard", args.hard),
    ]:
        lum = _load_lum_maps(path)
        if not auto_defaults:
            lum = _subsample(lum, forced_ratio, args.seed)
        lum_by_label[label] = lum

    plot_illumination_pca_overlay(
        lum_by_label=lum_by_label,
        out_dir=args.out_dir,
        stem=args.stem,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
