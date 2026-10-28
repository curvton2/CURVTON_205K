from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.lines import Line2D
import seaborn as sns

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


def _get_palette(label: str, idx: int) -> str:
    try:
        from EDA.plot_style import DATASET_COLORS, PALETTE

        return DATASET_COLORS.get(label, PALETTE[idx % len(PALETTE)])
    except Exception:
        fallback = ["#1B6CA8", "#D14D41", "#1B9E77", "#D95F02", "#7570B3"]
        return fallback[idx % len(fallback)]


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _load_bg_features(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "bg_entropy" not in data:
        raise KeyError(f"Missing bg_entropy in {npz_path}")
    ent = data["bg_entropy"].astype(np.float32)
    obj = data.get("bg_obj_count", np.zeros(len(ent))).astype(np.float32)
    return ent, obj


def _subsample_pair(
    ent: np.ndarray,
    obj: np.ndarray,
    sample_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_ratio >= 1.0:
        return ent, obj
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = ent.shape[0]
    if n <= 1:
        return ent, obj
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return ent[idx], obj[idx]


def _clean(ent: np.ndarray, obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(ent) & np.isfinite(obj)
    return ent[mask], obj[mask]


def _print_stats(label: str, ent: np.ndarray, obj: np.ndarray) -> None:
    if len(ent) == 0:
        print(f"{label}: no valid samples")
        return
    print(
        f"{label}: n={len(ent)} | entropy mean={ent.mean():.3f} var={ent.var():.3f} "
        f"| object mean={obj.mean():.3f} var={obj.var():.3f}"
    )


def plot_entropy_histogram(
    datasets: Dict[str, np.ndarray],
    out_dir: Path,
    bins: int,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.6))
    legend_handles: List[Line2D] = []

    for i, (label, ent) in enumerate(datasets.items()):
        ent = ent[np.isfinite(ent)]
        if len(ent) == 0:
            continue
        color = _get_palette(label, i)
        ax1.hist(ent, bins=bins, density=True, histtype="step", color=color, linewidth=1.4, alpha=0.95)
        ax1.hist(ent, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.20)
        ax1.axvline(ent.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        sns.kdeplot(ent, ax=ax2, fill=True, alpha=0.20, color=color, linewidth=1.6)
        ax2.axvline(ent.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.6, label=label)
        )

    for ax in (ax1, ax2):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.4)

    if show_legend and legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)))

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_entropy_histogram")
    plt.close(fig)


def plot_object_histogram(
    datasets: Dict[str, np.ndarray],
    out_dir: Path,
    bins: int,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.6))
    legend_handles: List[Line2D] = []

    for i, (label, obj) in enumerate(datasets.items()):
        obj = obj[np.isfinite(obj)]
        if len(obj) == 0:
            continue
        color = _get_palette(label, i)
        ax1.hist(obj, bins=bins, density=True, histtype="step", color=color, linewidth=1.4, alpha=0.95)
        ax1.hist(obj, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.20)
        ax1.axvline(obj.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        sns.kdeplot(obj, ax=ax2, fill=True, alpha=0.20, color=color, linewidth=1.6)
        ax2.axvline(obj.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.6, label=label)
        )

    for ax in (ax1, ax2):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.4)

    if show_legend and legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)))

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_object_density_histogram")
    plt.close(fig)


def plot_entropy_object_bar_overall(
    datasets_ent: Dict[str, np.ndarray],
    datasets_obj: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    _apply_eccv_style()

    all_ent = np.concatenate(list(datasets_ent.values())) if datasets_ent else np.array([])
    all_obj = np.concatenate(list(datasets_obj.values())) if datasets_obj else np.array([])
    all_ent = all_ent[np.isfinite(all_ent)]
    all_obj = all_obj[np.isfinite(all_obj)]
    if all_ent.size == 0 or all_obj.size == 0:
        return

    # Keep palette direction consistent with existing plot style.
    ent_color = _get_palette("Easy", 0)
    obj_color = _get_palette("Medium", 1)

    fig, ax = plt.subplots(figsize=(3.8, 3.1))
    x = np.arange(2)
    vals = [float(all_ent.mean()), float(all_obj.mean())]
    ax.bar(x, vals, width=0.58, color=[ent_color, obj_color], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(["Entropy", "Object Density"], fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", bottom=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.4)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_entropy_object_density_bar_overall")
    plt.close(fig)


def plot_entropy_object_bars_concat(
    datasets_ent: Dict[str, np.ndarray],
    datasets_obj: Dict[str, np.ndarray],
    out_dir: Path,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    labels = list(datasets_ent.keys())
    if not labels:
        return

    ent_means = []
    obj_means = []
    for lbl in labels:
        ent = datasets_ent.get(lbl, np.array([], dtype=np.float32))
        obj = datasets_obj.get(lbl, np.array([], dtype=np.float32))
        ent = ent[np.isfinite(ent)]
        obj = obj[np.isfinite(obj)]
        ent_means.append(float(np.mean(ent)) if ent.size else 0.0)
        obj_means.append(float(np.mean(obj)) if obj.size else 0.0)

    x = np.arange(len(labels))
    colors = [_get_palette(lbl, i) for i, lbl in enumerate(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150)

    ax1.bar(x, ent_means, color=colors, alpha=0.88, width=0.62)
    ax1.set_title("Background Entropy")
    ax1.set_xlabel("Dataset Split")
    ax1.set_ylabel("Mean Entropy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.4)

    ax2.bar(x, obj_means, color=colors, alpha=0.88, width=0.62)
    ax2.set_title("Object Density")
    ax2.set_xlabel("Dataset Split")
    ax2.set_ylabel("Mean Object Count")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.4)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#cccccc")

    if show_legend:
        handles = [
            Line2D([0], [0], color=colors[i], lw=6, label=labels[i]) for i in range(len(labels))
        ]
        fig.legend(handles=handles, loc="upper center", ncol=min(4, len(labels)), framealpha=0.95)

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_entropy_object_density_bars_concat")
    plt.close(fig)


def plot_entropy_object_kde_concat(
    datasets_ent: Dict[str, np.ndarray],
    datasets_obj: Dict[str, np.ndarray],
    out_dir: Path,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.6))
    legend_handles: List[Line2D] = []

    for i, label in enumerate(datasets_ent.keys()):
        ent = datasets_ent.get(label, np.array([], dtype=np.float32))
        obj = datasets_obj.get(label, np.array([], dtype=np.float32))
        ent = ent[np.isfinite(ent)]
        obj = obj[np.isfinite(obj)]
        if len(ent) == 0 and len(obj) == 0:
            continue

        color = _get_palette(label, i)
        if len(ent) > 0:
            sns.kdeplot(ent, ax=ax1, fill=True, alpha=0.20, color=color, linewidth=1.6)
            ax1.axvline(ent.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        if len(obj) > 0:
            sns.kdeplot(obj, ax=ax2, fill=True, alpha=0.20, color=color, linewidth=1.6)
            ax2.axvline(obj.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.6, label=label))

    ax1.set_title("Background Entropy")
    ax2.set_title("Object Density")
    ax1.set_xlabel("Entropy")
    ax1.set_ylabel("Density")
    ax2.set_xlabel("Object Count")
    ax2.set_ylabel("Density")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="x", which="major", labelsize=8)
        ax.tick_params(axis="x", which="minor", length=2)
        ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.4)

    # Fixed readable scale requested for object density.
    ax2.set_xlim(2.0, 24.0)
    ax2.set_xticks(np.arange(2.0, 26.0, 2.0))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(1))

    if show_legend and legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)))

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_entropy_object_density_kde_concat")
    plt.close(fig)


def plot_entropy_vs_objects(
    datasets_ent: Dict[str, np.ndarray],
    datasets_obj: Dict[str, np.ndarray],
    out_dir: Path,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    legend_handles: List[Line2D] = []

    for i, (label, ent) in enumerate(datasets_ent.items()):
        obj = datasets_obj.get(label, np.zeros(len(ent)))
        ent, obj = _clean(ent, obj)
        if len(ent) == 0:
            continue
        color = _get_palette(label, i)
        ax.scatter(
            obj,
            ent,
            s=10,
            alpha=0.5,
            color=color,
            linewidths=0.2,
            edgecolors="white",
            rasterized=True,
        )
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=label))

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.4)

    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="best", framealpha=0.9)

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_entropy_vs_objects")
    plt.close(fig)


def plot_density_heatmap(
    ent: np.ndarray,
    obj: np.ndarray,
    out_dir: Path,
    stem: str,
    bins: int,
    vmin: float,
    vmax: float,
) -> None:
    _apply_eccv_style()

    ent, obj = _clean(ent, obj)
    if len(ent) == 0:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    h = ax.hist2d(obj, ent, bins=bins, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(False)

    cbar = fig.colorbar(h[3], ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style EDA plots for background entropy and object density."
    )
    parser.add_argument("--features", nargs="+", default=None, help="NPZ files with bg_entropy/bg_obj_count.")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each NPZ file.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/background_eda"))
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--heatmap-bins", type=int, default=40)
    parser.add_argument("--heatmap-vmin", type=float, default=0.0)
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=-1.0,
        help="Max value for heatmap. If negative, uses data max.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legends for cleaner visuals.",
    )
    add_autogen_args(parser)
    return parser.parse_args()


def _require_cache_files(paths: List[Path]) -> None:
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
    if args.features is None and args.labels is None:
        pct = int(round(forced_ratio * 100))
        args.features = [
            args.cache_dir / f"curvton_easy_{pct}pct.npz",
            args.cache_dir / f"curvton_medium_{pct}pct.npz",
            args.cache_dir / f"curvton_hard_{pct}pct.npz",
        ]
        args.labels = ["Easy", "Medium", "Hard"]
        auto_defaults = True

    if not args.features or not args.labels:
        raise ValueError("Provide --features and --labels, or omit both to use defaults.")
    if len(args.features) != len(args.labels):
        raise ValueError("--features and --labels must have the same length")

    if auto_defaults:
        missing_diffs = caches_requiring_generation(
            cache_dir=args.cache_dir,
            sample_ratio=forced_ratio,
            difficulties=["easy", "medium", "hard"],
            required_keys=["bg_entropy", "bg_obj_count"],
        )
    else:
        missing_diffs = []
    if missing_diffs:
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["bg_entropy", "bg_obj_count"],
        )
    _require_cache_files([Path(p) for p in args.features])

    ent_data: Dict[str, np.ndarray] = {}
    obj_data: Dict[str, np.ndarray] = {}

    for path, label in zip(args.features, args.labels):
        ent, obj = _load_bg_features(Path(path))
        if not auto_defaults:
            ent, obj = _subsample_pair(ent, obj, forced_ratio, args.seed)
        ent, obj = _clean(ent, obj)
        ent_data[label] = ent
        obj_data[label] = obj
        _print_stats(label, ent, obj)

    show_legend = not args.no_legend

    plot_entropy_object_kde_concat(ent_data, obj_data, args.out_dir, show_legend=show_legend)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
