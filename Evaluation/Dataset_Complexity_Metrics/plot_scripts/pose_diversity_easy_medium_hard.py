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

from curvton_cache_autogen import (
    add_autogen_args,
    caches_requiring_generation,
    ensure_curvton_caches,
)


REQUIRED_SAMPLE_RATIO = 0.25


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


def _load_pose_arrays(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "pose_vecs" not in data:
        raise KeyError(f"Missing pose_vecs in {npz_path}")
    if "angles" not in data:
        raise KeyError(f"Missing angles in {npz_path}")
    pose_vecs = data["pose_vecs"].astype(np.float32)
    angles = data["angles"].astype(np.float32)
    return pose_vecs, angles


def _subsample_pair(
    pose_vecs: np.ndarray,
    angles: np.ndarray,
    sample_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_ratio >= 1.0:
        return pose_vecs, angles
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = pose_vecs.shape[0]
    if n <= 1:
        return pose_vecs, angles
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return pose_vecs[idx], angles[idx]


def _pose_diversity_score(pose_vecs: np.ndarray) -> float:
    if pose_vecs.ndim != 2:
        raise ValueError(f"pose_vecs expected shape (N, D), got {pose_vecs.shape}")
    if pose_vecs.shape[0] < 2:
        return 0.0
    return float(np.mean(np.var(pose_vecs, axis=0)))


def _angle_diversity_score(angles: np.ndarray) -> float:
    if angles.ndim != 2:
        raise ValueError(f"angles expected shape (N, K), got {angles.shape}")
    if angles.shape[0] < 2:
        return 0.0
    angles_deg = angles * (180.0 / np.pi)
    return float(np.mean(np.var(angles_deg, axis=0)))


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _select_pose_feature_indices(
    pose_by_split: Dict[str, np.ndarray],
    n_features: int,
) -> np.ndarray:
    all_pose = np.concatenate([pose_by_split[k] for k in ["Easy", "Medium", "Hard"]], axis=0)
    global_var = np.var(all_pose, axis=0)
    n = min(n_features, global_var.shape[0])
    idx = np.argsort(-global_var)[:n]
    return np.sort(idx)


def plot_pose_diversity(
    pose_by_split: Dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
    show_legend: bool,
    n_features: int,
) -> None:
    _apply_eccv_style()
    base_colors = _difficulty_colors()
    colors = {
        "Easy": base_colors["Easy"],
        "Medium": base_colors["Medium"],
        "Hard": base_colors["Hard"],
    }

    difficulties = ["Easy", "Medium", "Hard"]
    feat_idx = _select_pose_feature_indices(pose_by_split, n_features=n_features)
    metric_labels: List[str] = [f"f{int(i)}" for i in feat_idx]

    # Normalize variance by mean magnitude per split to reduce scale discrepancies.
    per_split_feature_var: Dict[str, np.ndarray] = {}
    for d in difficulties:
        pose = pose_by_split[d]
        feat = pose[:, feat_idx]
        mean_abs = np.mean(np.abs(feat), axis=0) + 1e-8
        var = np.var(feat, axis=0)
        per_split_feature_var[d] = var / (mean_abs ** 2)

    # Normalize for display across all plotted values (not per-feature across 3 splits).
    var_matrix = np.stack([per_split_feature_var[d] for d in difficulties], axis=0)  # (3, F)
    vmax = float(np.max(var_matrix))
    if vmax < 1e-12:
        vmax = 1.0
    var_matrix_norm = var_matrix / vmax
    per_split_feature_var = {d: var_matrix_norm[i] for i, d in enumerate(difficulties)}
    radial_max = 1.0

    n_axes = len(metric_labels)
    if n_axes < 3:
        raise ValueError(
            f"Need at least 3 pose features for a spider plot, got {n_axes}. "
            "Increase --num-pose-features or check pose_vec dimensionality."
        )

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(4.8, 4.0), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, radial_max * 1.05)
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels, fontsize=8)
    y_ticks = np.linspace(0.2 * radial_max, radial_max, num=5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.2f}" for v in y_ticks], fontsize=7, color="#6B7280")
    ax.grid(color="#D9DEE7", linewidth=0.8, alpha=0.9)
    ax.spines["polar"].set_color("#C5CCD8")
    ax.spines["polar"].set_linewidth(0.8)

    line_styles = {"Easy": "-", "Medium": "--", "Hard": "-."}
    for d in difficulties:
        vals = per_split_feature_var[d].astype(np.float32)
        vals_closed = np.concatenate([vals, [vals[0]]])
        ax.plot(
            angles_closed,
            vals_closed,
            color=colors[d],
            linewidth=2.0,
            linestyle=line_styles[d],
            label=d,
            zorder=3,
        )
        ax.fill(angles_closed, vals_closed, color=colors[d], alpha=0.07, zorder=2)

    ax.set_title("Pose Feature Variance (Normalized)", fontsize=9, pad=14)

    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), framealpha=0.95, fontsize=8)

    fig.tight_layout()
    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style pose diversity comparison across easy/medium/hard splits."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ with pose_vecs and angles for Easy.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ with pose_vecs and angles for Medium.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ with pose_vecs and angles for Hard.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/pose_diversity"))
    parser.add_argument("--stem", type=str, default="pose_diversity_easy_medium_hard")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend for minimal text.")
    parser.add_argument(
        "--num-pose-features",
        type=int,
        default=12,
        help="Number of highest-variance pose features to display in spider plot.",
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
            required_keys=["pose_vecs", "angles"],
        )
    else:
        missing_diffs = []
    if missing_diffs:
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["pose_vecs", "angles"],
        )
    _require_cache_files((args.easy, args.medium, args.hard))

    metrics: Dict[str, Tuple[float, float]] = {}
    pose_by_split: Dict[str, np.ndarray] = {}
    for label, path in [
        ("Easy", args.easy),
        ("Medium", args.medium),
        ("Hard", args.hard),
    ]:
        pose_vecs, angles = _load_pose_arrays(path)
        if not auto_defaults:
            pose_vecs, angles = _subsample_pair(pose_vecs, angles, forced_ratio, args.seed)
        pose_by_split[label] = pose_vecs
        pose_score = _pose_diversity_score(pose_vecs)
        angle_score = _angle_diversity_score(angles)
        metrics[label] = (pose_score, angle_score)

    plot_pose_diversity(
        pose_by_split=pose_by_split,
        out_dir=args.out_dir,
        stem=args.stem,
        show_legend=not args.no_legend,
        n_features=args.num_pose_features,
    )

    print("Pose diversity scores:")
    for label, (pose_score, angle_score) in metrics.items():
        print(f"  {label}: pose_var={pose_score:.4f}, angle_var={angle_score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
