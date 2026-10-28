from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
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


def _load_angles(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    data = dict(np.load(npz_path, allow_pickle=True))
    az = data.get("azimuths", data.get("azimuth", np.array([])))
    el = data.get("elevations", data.get("elevation", np.array([])))
    conf = data.get("camera_confidence", None)
    az = np.asarray(az, dtype=np.float32)
    el = np.asarray(el, dtype=np.float32)
    if az.size == 0:
        raise KeyError(f"Missing azimuths in {npz_path}")
    if el.size == 0:
        el = np.zeros_like(az)
    if conf is None:
        conf_arr = None
        mask = np.isfinite(az) & np.isfinite(el)
        return az[mask], el[mask], conf_arr
    conf_arr = np.asarray(conf, dtype=np.float32)
    if conf_arr.shape[0] != az.shape[0]:
        conf_arr = None
        mask = np.isfinite(az) & np.isfinite(el)
        return az[mask], el[mask], conf_arr
    mask = np.isfinite(az) & np.isfinite(el) & np.isfinite(conf_arr)
    return az[mask], el[mask], conf_arr[mask]


def _subsample_triplet(
    az: np.ndarray,
    el: np.ndarray,
    conf: Optional[np.ndarray],
    sample_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if sample_ratio >= 1.0:
        return az, el, conf
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = az.shape[0]
    if n <= 1:
        return az, el, conf
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    if conf is None:
        return az[idx], el[idx], conf
    return az[idx], el[idx], conf[idx]


def _wrap_azimuth(az: np.ndarray) -> np.ndarray:
    """Map azimuth to [-180, 180] for centered plotting."""
    az = np.remainder(az, 360.0)
    az = np.where(az > 180.0, az - 360.0, az)
    return az


def _smooth2d(H: np.ndarray, iters: int = 2) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    kernel = kernel / kernel.sum()
    out = H.astype(np.float32)
    for _ in range(max(1, iters)):
        # Convolve rows
        out = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = (
            out[:, :-2] * kernel[0]
            + out[:, 1:-1] * kernel[1]
            + out[:, 2:] * kernel[2]
        )
        # Convolve cols
        out = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = (
            out[:-2, :] * kernel[0]
            + out[1:-1, :] * kernel[1]
            + out[2:, :] * kernel[2]
        )
    return out


def plot_camera_overlay(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    out_dir: Path,
    stem: str,
    bins_az: int,
    bins_el: int,
) -> None:
    _apply_eccv_style()
    colors = _difficulty_colors()

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150, constrained_layout=True)

    # Optional radius proxy intensity map from camera confidence (r), if available.
    conf_maps = []
    for label in ["Easy", "Medium", "Hard"]:
        if label not in datasets:
            continue
        az, el, conf = datasets[label]
        if conf is None or conf.shape[0] == 0:
            continue
        az = _wrap_azimuth(az)
        el = np.clip(el, -90.0, 90.0)
        H_sum, xedges, yedges = np.histogram2d(
            az, el, bins=[bins_az, bins_el], range=[[-180, 180], [-90, 90]], weights=conf
        )
        H_cnt, _, _ = np.histogram2d(az, el, bins=[bins_az, bins_el], range=[[-180, 180], [-90, 90]])
        H_mean = np.divide(H_sum, np.maximum(H_cnt, 1e-8), where=H_cnt > 0)
        H_mean = _smooth2d(H_mean, iters=1)
        conf_maps.append(H_mean)

    if conf_maps:
        conf_bg = np.mean(np.stack(conf_maps, axis=0), axis=0)
        im = ax.imshow(
            conf_bg.T,
            origin="lower",
            extent=[-180, 180, -90, 90],
            cmap="magma",
            alpha=0.30,
            aspect="auto",
            vmin=float(np.nanmin(conf_bg)),
            vmax=float(np.nanmax(conf_bg)),
            interpolation="bilinear",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("r (camera confidence)", fontsize=8)

    for label in ["Easy", "Medium", "Hard"]:
        if label not in datasets:
            continue
        az, el, _ = datasets[label]
        az = _wrap_azimuth(az)
        el = np.clip(el, -90.0, 90.0)
        H, xedges, yedges = np.histogram2d(az, el, bins=[bins_az, bins_el], range=[[-180, 180], [-90, 90]])
        H = _smooth2d(H, iters=2)
        X = 0.5 * (xedges[:-1] + xedges[1:])
        Y = 0.5 * (yedges[:-1] + yedges[1:])
        Xg, Yg = np.meshgrid(X, Y, indexing="xy")

        # Use percentile-based contour levels for stable overlays.
        levels = np.percentile(H[H > 0], [60, 80, 92]) if np.any(H > 0) else [1, 2, 3]
        ax.contour(
            Xg,
            Yg,
            H.T,
            levels=levels,
            colors=[colors[label]],
            linewidths=1.4,
            alpha=0.9,
        )

    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.grid(True, linestyle="--", alpha=0.22, linewidth=0.45)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    legend_handles = [
        plt.Line2D([0], [0], color=colors["Easy"], linewidth=2.0, label="Easy"),
        plt.Line2D([0], [0], color=colors["Medium"], linewidth=2.0, label="Medium"),
        plt.Line2D([0], [0], color=colors["Hard"], linewidth=2.0, label="Hard"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style overlay of camera angle distributions for easy/medium/hard splits."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ with azimuths/elevations for Easy.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ with azimuths/elevations for Medium.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ with azimuths/elevations for Hard.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/camera_angle_overlay"))
    parser.add_argument("--stem", type=str, default="camera_angle_overlay_easy_medium_hard")
    parser.add_argument("--bins-az", type=int, default=36)
    parser.add_argument("--bins-el", type=int, default=18)
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
            required_keys=["azimuths", "elevations"],
        )
    else:
        missing_diffs = []
    if missing_diffs:
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["azimuths", "elevations"],
        )
    _require_cache_files((args.easy, args.medium, args.hard))

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    for label, path in [
        ("Easy", args.easy),
        ("Medium", args.medium),
        ("Hard", args.hard),
    ]:
        az, el, conf = _load_angles(path)
        if not auto_defaults:
            az, el, conf = _subsample_triplet(az, el, conf, forced_ratio, args.seed)
        datasets[label] = (az, el, conf)

    plot_camera_overlay(
        datasets=datasets,
        out_dir=args.out_dir,
        stem=args.stem,
        bins_az=args.bins_az,
        bins_el=args.bins_el,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
