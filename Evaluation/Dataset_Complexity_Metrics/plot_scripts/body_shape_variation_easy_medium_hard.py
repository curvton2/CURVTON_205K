from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
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


def _load_betas(npz_path: Path) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "betas" not in data:
        raise KeyError(f"Missing betas in {npz_path}")
    betas = data["betas"].astype(np.float32)
    if betas.ndim != 2:
        raise ValueError(f"betas expected shape (N, D), got {betas.shape}")
    return betas


def _subsample(betas: np.ndarray, sample_ratio: float, seed: int) -> np.ndarray:
    if sample_ratio >= 1.0:
        return betas
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = betas.shape[0]
    if n <= 1:
        return betas
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return betas[idx]


def _standardize(betas: np.ndarray) -> np.ndarray:
    mu = betas.mean(axis=0, keepdims=True)
    sig = betas.std(axis=0, keepdims=True) + 1e-8
    return (betas - mu) / sig


def _variation_score(betas: np.ndarray) -> float:
    if betas.shape[0] < 2:
        return 0.0
    z = _standardize(betas)
    return float(np.mean(np.var(z, axis=0)))


def _entropy_score(betas: np.ndarray, bins: int = 40) -> float:
    if betas.shape[0] < 2:
        return 0.0
    z = _standardize(betas)
    entropies = []
    for j in range(z.shape[1]):
        vals = z[:, j]
        counts, _ = np.histogram(vals, bins=bins, density=True)
        p = counts[counts > 0]
        if p.size == 0:
            continue
        p = p / p.sum()
        ent = -np.sum(p * np.log(p))
        entropies.append(float(ent))
    return float(np.mean(entropies)) if entropies else 0.0


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_body_shape_variation(
    variance_score: float,
    entropy_score: float,
    betas_all: np.ndarray,
    out_dir: Path,
    stem: str,
    show_legend: bool,
) -> None:
    _apply_eccv_style()
    colors = {"Variance": "#4C78A8", "Entropy": "#72B7B2"}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), dpi=150)
    ax_bar, ax_pdf, ax_cdf = axes

    # Bar plot: overall variance + entropy summary
    x = np.arange(2)
    vals = [variance_score, entropy_score]
    labels = ["Variance", "Entropy"]
    ax_bar.bar(
        x,
        vals,
        width=0.58,
        color=[colors["Variance"], colors["Entropy"]],
        alpha=0.9,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax_bar.set_xlabel("")
    ax_bar.set_ylabel("")
    ax_bar.tick_params(axis="x", bottom=False)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.5)

    # Empirical distribution for overall standardized betas (no Gaussian/KDE smoothing).
    z = _standardize(betas_all).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size >= 5:
        hist, edges = np.histogram(z, bins=80, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_pdf.plot(
            centers,
            hist,
            color="#4C78A8",
            linewidth=1.7,
            label="Overall empirical density",
        )
        ax_pdf.fill_between(centers, hist, 0.0, color="#4C78A8", alpha=0.15)

        zs = np.sort(z)
        ys = np.linspace(0.0, 1.0, zs.size)
        ax_cdf.plot(zs, ys, color="#72B7B2", linewidth=1.7, label="Overall ECDF")

    for ax in (ax_pdf, ax_cdf):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.grid(True, linestyle="--", alpha=0.2, linewidth=0.4)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#cccccc")

    if show_legend:
        ax_pdf.legend(loc="upper right", framealpha=0.9, fontsize=8)
        ax_cdf.legend(loc="lower right", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style body shape variation comparison across easy/medium/hard splits."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ with betas for Easy.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ with betas for Medium.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ with betas for Hard.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/body_shape_variation"))
    parser.add_argument("--stem", type=str, default="body_shape_variation_easy_medium_hard")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend for minimal text.")
    parser.add_argument("--entropy-bins", type=int, default=40)
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
            required_keys=["betas"],
        )
    else:
        missing_diffs = []
    if missing_diffs:
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["betas"],
        )
    _require_cache_files((args.easy, args.medium, args.hard))

    all_betas = []
    for label, path in [
        ("Easy", args.easy),
        ("Medium", args.medium),
        ("Hard", args.hard),
    ]:
        betas = _load_betas(path)
        if not auto_defaults:
            betas = _subsample(betas, forced_ratio, args.seed)
        all_betas.append(betas)

    betas_all = np.concatenate(all_betas, axis=0) if all_betas else np.zeros((0, 0), dtype=np.float32)
    var_score = _variation_score(betas_all)
    ent_score = _entropy_score(betas_all, bins=args.entropy_bins)

    plot_body_shape_variation(
        variance_score=var_score,
        entropy_score=ent_score,
        betas_all=betas_all,
        out_dir=args.out_dir,
        stem=args.stem,
        show_legend=not args.no_legend,
    )

    print("Body shape variation scores (overall dataset):")
    print(f"  Overall: variance={var_score:.4f}, entropy={ent_score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
