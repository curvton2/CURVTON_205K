"""
pretrained_metrics/dataloader.py
=================================
**Thin delegation layer** — all dataset logic lives in ``datasets/``.

This module re-exports :func:`get_dataloader` and :data:`ALL_DATASETS`
so that every script inside ``pretrained_metrics/`` and ``EDA/`` can do:

    from pretrained_metrics.dataloader import get_dataloader, ALL_DATASETS

without importing dataset classes directly.

The *only* source of truth for:
  - Dataset folder structures  →  ``datasets/loaders.py``
  - Percentage / split logic   →  ``datasets/loaders.py``   (split=)
  - Image transforms           →  ``datasets/base_dataset.py``
  - Collation                  →  this file  (thin collate that handles
                                  None tensors from BaseTryOnDataset)

No dataset classes are defined here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# ── canonical dataset package ────────────────────────────────────────────────
_WORKSPACE = Path(__file__).parent.parent   # pretrained_metrics_evals/
sys.path.insert(0, str(_WORKSPACE))

from datasets.loaders import get_dataset, DATASET_REGISTRY   # noqa: E402

# ── public API ────────────────────────────────────────────────────────────────
ALL_DATASETS: List[str] = list(DATASET_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def _collate(batch: List[Dict]) -> Dict:
    """
    Stack tensors from BaseTryOnDataset.__getitem__.
    Keys: person, cloth, gt  → (B,3,H,W)  float32
          mask               → (B,1,H,W)  float32
          meta               → list[dict]

    Gracefully handles None items (replaced with zeros of the same size
    inferred from the first non-None tensor in the batch).
    """
    # Find a reference shape for each key
    ref: Dict[str, Optional[torch.Tensor]] = {}
    for key in ("person", "cloth", "gt"):
        for b in batch:
            v = b.get(key)
            if v is not None:
                ref[key] = v
                break
        else:
            ref[key] = None

    def _safe_stack(key: str, C: int = 3) -> torch.Tensor:
        r = ref.get(key)
        if r is None:
            # All items missing → zero tensor (B, C, 1, 1)
            return torch.zeros(len(batch), C, 1, 1)
        H, W = r.shape[-2], r.shape[-1]
        tensors = []
        for b in batch:
            v = b.get(key)
            if v is None:
                tensors.append(torch.zeros(C, H, W))
            else:
                tensors.append(v)
        return torch.stack(tensors)         # (B, C, H, W)

    # mask is (1,H,W) from base_dataset
    ref_mask = next((b["mask"] for b in batch
                     if b.get("mask") is not None), None)
    if ref_mask is not None:
        H_m, W_m = ref_mask.shape[-2], ref_mask.shape[-1]
        masks = []
        for b in batch:
            m = b.get("mask")
            masks.append(m if m is not None else torch.zeros(1, H_m, W_m))
        mask_t = torch.stack(masks)         # (B, 1, H, W)
    else:
        mask_t = torch.zeros(len(batch), 1, 1, 1)

    return {
        "person": _safe_stack("person", 3),
        "cloth":  _safe_stack("cloth",  3),
        "gt":     _safe_stack("gt",     3),
        "mask":   mask_t,
        "meta":   [
            {**b.get("meta", {}), "dataset": b.get("meta", {}).get("dataset", "?")}
            for b in batch
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public factory — delegates entirely to datasets/loaders.py
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    dataset_name: str,
    root: str,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (512, 384),
    shuffle: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    """
    Build a :class:`~torch.utils.data.DataLoader` for any of the 10 VTON
    datasets using the canonical ``datasets/`` package.

    Parameters
    ----------
    dataset_name : str
        One of :data:`ALL_DATASETS`.
    root : str
        Absolute path to the dataset root directory.
    split : str
        ``"train"`` | ``"test"`` | ``"val"``.
        The percentage of data used is determined by the pairs file present
        for that split inside each dataset's root.
    batch_size : int
    num_workers : int
    img_size : (H, W)
        Images are resized to this resolution by :func:`datasets.base_dataset.default_transform`.
    shuffle : bool
        Whether to shuffle (False for evaluation, True for training).
    **dataset_kwargs
        Any extra kwargs forwarded to the dataset class constructor, e.g.
        ``category="upper_body"`` for DressCode.

    Returns
    -------
    DataLoader
        Each batch is a dict::

            {
              "person" : Tensor (B, 3, H, W)  float32 [0, 1]
              "cloth"  : Tensor (B, 3, H, W)  float32 [0, 1]
              "gt"     : Tensor (B, 3, H, W)  float32 [0, 1]
              "mask"   : Tensor (B, 1, H, W)  float32 [0, 1]
              "meta"   : list[dict]           [{id, dataset}, ...]
            }

    Raises
    ------
    ValueError  – unknown dataset name
    RuntimeError – dataset has 0 samples (wrong root or missing pairs file)
    """
    name = dataset_name.lower().replace("-", "_")

    # get_dataset raises ValueError for unknown names automatically
    ds = get_dataset(name, root, split=split, img_size=img_size, **dataset_kwargs)

    if len(ds) == 0:
        raise RuntimeError(
            f"Dataset '{name}' at '{root}' (split='{split}') has 0 samples.\n"
            "Check that the root path is correct and the pairs file exists."
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate,
    )
