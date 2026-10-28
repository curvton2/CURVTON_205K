from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict


DEFAULT_MODEL_BASE = os.getenv("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/curvton_metrics"))


def configure_model_caches(
    base_path: str = DEFAULT_MODEL_BASE,
    set_home_for_hmr2: bool = False,
) -> Dict[str, str]:
    """
    Configure cache/download roots so heavyweight model artifacts are stored
    under the provided absolute base path (HPC scratch-friendly).

    Parameters
    ----------
    base_path:
        Absolute base directory where cache folders should live.
    set_home_for_hmr2:
        HMR2 uses ~/.cache/4DHumans internally; setting HOME redirects this.
    """
    base = Path(base_path).expanduser().resolve()
    cache_root = base / ".cache"
    hf_home = cache_root / "huggingface"
    hf_hub = hf_home / "hub"
    hf_datasets = hf_home / "datasets"
    torch_home = cache_root / "torch"
    fourd = cache_root / "4DHumans"

    # Best-effort directory creation.
    for p in (cache_root, hf_home, hf_hub, hf_datasets, torch_home, fourd):
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_hub)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

    # Not guaranteed to be used by hmr2 internals, but harmless and explicit.
    os.environ["FOURDHUMANS_CACHE_DIR"] = str(fourd)

    if set_home_for_hmr2:
        os.environ["HOME"] = str(base)

    return {
        "base_path": str(base),
        "cache_root": str(cache_root),
        "hf_home": str(hf_home),
        "hf_hub": str(hf_hub),
        "torch_home": str(torch_home),
        "fourdhumans_cache": str(fourd),
        "home": os.environ.get("HOME", ""),
    }


def ensure_hmr2_smpl_model(base_path: str = DEFAULT_MODEL_BASE) -> Dict[str, str]:
    """
    Ensure HMR2's relative lookup path `data/basicModel_...pkl` exists
    in the current working directory by linking/copying from scratch cache.
    """
    filename = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    src = Path(base_path).expanduser().resolve() / ".cache" / "4DHumans" / "data" / filename
    dst = Path.cwd() / "data" / filename
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return {"status": "exists", "src": str(src), "dst": str(dst)}
    if not src.exists():
        return {"status": "missing_source", "src": str(src), "dst": str(dst)}

    try:
        dst.symlink_to(src)
        return {"status": "linked", "src": str(src), "dst": str(dst)}
    except Exception:
        shutil.copy2(src, dst)
        return {"status": "copied", "src": str(src), "dst": str(dst)}
