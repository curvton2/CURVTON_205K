from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import yaml

from pretrained_metrics.cache_setup import DEFAULT_MODEL_BASE, configure_model_caches
from pretrained_metrics.dataloader import get_dataloader


def _g(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _r(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _fv(v):
    if isinstance(v, float):
        return "NA" if np.isnan(v) else f"{v:.6g}"
    if isinstance(v, (np.floating,)):
        val = float(v)
        return "NA" if np.isnan(val) else f"{val:.6g}"
    return str(v) if v is not None else "NA"


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--download_base", default=DEFAULT_MODEL_BASE)
    p.add_argument("--config", type=str, default="configs/pretrained_metrics_datasets.yaml")
    p.add_argument("--only_datasets", nargs="*", default=None)
    p.add_argument(
        "--only_dresscode_categories",
        nargs="*",
        default=None,
        help="Optional filter for DressCode categories (e.g., dresses upper_body lower_body).",
    )
    p.add_argument("--max_datasets", type=int, default=0)
    p.add_argument("--split", type=str, default=None, help="Optional split override for all datasets")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_size", type=int, nargs=2, default=[512, 384])
    p.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Number of batches per dataset (0 = all batches, default).",
    )
    p.add_argument(
        "--include_curvton_hard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include curvton hard split entries from YAML (default: enabled).",
    )
    p.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue all dataset entries even if one fails (default: enabled).",
    )
    p.add_argument(
        "--error_log",
        type=str,
        default=None,
        help="Optional path to append dataset failure logs.",
    )

    # Legacy args retained for backward-compatibility with older SLURM scripts.
    p.add_argument("--curvton_root", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default=None)


def _to_bchw(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 4:
        raise RuntimeError(f"[{name}] Expected 4-D tensor, got {tuple(x.shape)}")
    if x.shape[1] != 3 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    if x.shape[1] != 3:
        raise RuntimeError(f"[{name}] Expected C=3, got {tuple(x.shape)}")
    return x


def _is_curvton_hard(entry: Dict) -> bool:
    name = str(entry.get("name", "")).lower()
    root = str(entry.get("root", "")).replace("\\", "/").rstrip("/")
    return name == "curvton" and root.endswith("/hard")


def _load_entries(args: argparse.Namespace) -> List[Dict]:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = dict(raw.get("defaults", {}))
    entries = [dict(e) for e in raw.get("datasets", [])]

    if args.only_datasets:
        filt = {d.lower() for d in args.only_datasets}
        entries = [e for e in entries if str(e.get("name", "")).lower() in filt]

    if args.only_dresscode_categories:
        allowed = {c.lower() for c in args.only_dresscode_categories}
        filtered: List[Dict] = []
        for e in entries:
            name = str(e.get("name", "")).lower()
            if name != "dresscode":
                filtered.append(e)
                continue
            cat = str(e.get("dresscode_category", "all")).lower()
            if cat in allowed:
                filtered.append(e)
        entries = filtered

    merged: List[Dict] = []
    for e in entries:
        cfg = {**defaults, **e}
        if args.split:
            cfg["split"] = args.split
        if not args.include_curvton_hard and _is_curvton_hard(cfg):
            continue
        merged.append(cfg)

    if args.max_datasets > 0:
        merged = merged[: args.max_datasets]
    return merged


def _expand_loader_targets(cfg: Dict) -> List[Tuple[str, str, str, str | None]]:
    dataset_name = str(cfg["name"])
    root = str(cfg["root"])
    split = str(cfg.get("split", "test"))
    is_dresscode = "dresscode" in dataset_name.lower()
    dresscode_cat = cfg.get("dresscode_category", None) if is_dresscode else None

    if is_dresscode and str(dresscode_cat).lower() == "all":
        return [(dataset_name, root, split, c) for c in ("upper_body", "lower_body", "dresses")]
    return [(dataset_name, root, split, (str(dresscode_cat) if dresscode_cat else None))]


def _collect_batches(dataset_name: str, root: str, split: str, category: str | None,
                     args: argparse.Namespace) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    kwargs = {}
    if category:
        kwargs["category"] = category
    loader = get_dataloader(
        dataset_name=dataset_name,
        root=root,
        split=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
        **kwargs,
    )

    batches: List[Dict[str, torch.Tensor]] = []
    n_images = 0
    for i, batch in enumerate(loader):
        person = _to_bchw(batch["person"].float(), "person")
        cloth = _to_bchw(batch["cloth"].float(), "cloth")
        batches.append({"person": person, "cloth": cloth})
        n_images += int(person.shape[0])
        if args.max_batches > 0 and (i + 1) >= args.max_batches:
            break

    if not batches:
        raise RuntimeError("No batches loaded.")
    return batches, n_images


def run_metric_on_all_datasets(
    *,
    args: argparse.Namespace,
    metric_title: str,
    probe_fn: Callable[[str, List[Dict[str, torch.Tensor]], argparse.Namespace], Tuple[Dict, str, float]],
    paper_score_key: str,
    paper_score_label: str,
    set_home_for_hmr2: bool = False,
) -> int:
    configure_model_caches(args.download_base, set_home_for_hmr2=set_home_for_hmr2)
    entries = _load_entries(args)
    if not entries:
        print(f"{_r('FAILED')} No dataset entries selected from config.")
        return 1

    print("\n" + "=" * 96)
    print(f"  {metric_title} — ALL DATASETS")
    print(f"  config={args.config} | include_curvton_hard={args.include_curvton_hard} | max_batches={args.max_batches}")
    print("=" * 96)

    results: List[Dict] = []
    n_fail = 0
    paper_scores: List[float] = []

    seen_targets = set()

    def _append_error_log(entry: Dict[str, object]) -> None:
        log_path = getattr(args, "error_log", None)
        if not log_path:
            return
        try:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().isoformat(timespec="seconds")
            with open(p, "a", encoding="utf-8") as f:
                f.write(
                    f"[{ts}] "
                    f"dataset={entry.get('dataset')} "
                    f"split={entry.get('split')} "
                    f"category={entry.get('category')} "
                    f"root={entry.get('root')} "
                    f"error={entry.get('error')}\n"
                )
        except Exception as log_e:
            print(f"  [WARN] Could not write error log '{log_path}': {log_e}")

    for cfg in entries:
        targets = _expand_loader_targets(cfg)
        for dataset_name, root, split, category in targets:
            target_key = (dataset_name, root, split, category or "-")
            if target_key in seen_targets:
                continue
            seen_targets.add(target_key)

            label = f"{dataset_name} | split={split}"
            if category:
                label += f" | category={category}"
            print("\n" + "-" * 96)
            print(f"  Dataset: {label}")
            print(f"  Root   : {root}")
            print("-" * 96)

            try:
                batches, n_images = _collect_batches(dataset_name, root, split, category, args)
                result, backend, elapsed = probe_fn(args.device, batches, args)
                results.append(
                    {
                        "dataset": dataset_name,
                        "root": root,
                        "split": split,
                        "category": category or "-",
                        "status": "PASS",
                        "backend": backend,
                        "images": n_images,
                        "time": f"{elapsed:.2f}s",
                        "paper_score": result.get(paper_score_key, float("nan")),
                        "error": "-",
                    }
                )
                print(f"  {_g('PASS')} backend={backend} images={n_images} time={elapsed:.2f}s")
                print("  Computed values:")
                for k in sorted(result.keys()):
                    print(f"    {k:40s} = {_fv(result[k])}")
                primary = result.get(paper_score_key, float("nan"))
                print(f"  Primary paper score [{paper_score_label} / {paper_score_key}] = {_fv(primary)}")
                try:
                    primary_f = float(primary)
                    if not np.isnan(primary_f):
                        paper_scores.append(primary_f)
                except Exception:
                    pass
            except Exception as e:
                n_fail += 1
                fail_entry = {
                    "dataset": dataset_name,
                    "root": root,
                    "split": split,
                    "category": category or "-",
                    "status": "FAIL",
                    "backend": "-",
                    "images": "-",
                    "time": "-",
                    "paper_score": float("nan"),
                    "error": f"{type(e).__name__}: {e}",
                }
                results.append(fail_entry)
                _append_error_log(fail_entry)
                print(f"  {_r('FAIL')} {type(e).__name__}: {e}")
                if args.verbose:
                    traceback.print_exc()
                if not args.continue_on_error:
                    print("\nStopping due to --no-continue_on_error")
                    break
        if n_fail > 0 and not args.continue_on_error:
            break

    score_col = f"PrimaryScore ({paper_score_key})"
    headers = ["Dataset", "Root", "Split", "Category", "Status", "Backend", "Images", "Time", score_col, "Error"]
    rows = [
        [
            r["dataset"],
            r["root"],
            r["split"],
            r["category"],
            r["status"],
            r["backend"],
            r["images"],
            r["time"],
            _fv(r["paper_score"]),
            r["error"],
        ]
        for r in results
    ]
    _print_table("Dataset Execution Summary", headers, rows)
    print(f"\nCompleted entries={len(results)} | failures={n_fail}")
    if paper_scores:
        mean_score = float(np.mean(paper_scores))
        print(
            f"Metric-level paper score [{paper_score_label}] "
            f"(mean over successful datasets): {_fv(mean_score)}"
        )
    else:
        print(f"Metric-level paper score [{paper_score_label}]: NA")
    return 0 if n_fail == 0 else 1


def _print_table(title: str, headers: List[str], rows: List[List[object]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    print("\n" + title)
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)
