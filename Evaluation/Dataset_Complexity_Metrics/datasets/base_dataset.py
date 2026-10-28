"""
base_dataset.py
---------------
Abstract base dataset.  Every concrete dataset must return a dict with keys:
  - 'cloth'   : PIL Image or Tensor (RGB, [0,1])
  - 'person'  : PIL Image or Tensor (RGB, [0,1])
  - 'gt'      : PIL Image or Tensor (RGB, [0,1]) — ground-truth try-on result
  - 'mask'    : PIL Image or Tensor (Greyscale, [0,1]) — optional segmentation mask
  - 'meta'    : dict with at least {'id': str}
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# --------------------------------------------------------------------------- #
#  Default transforms                                                          #
# --------------------------------------------------------------------------- #
def default_transform(size: Tuple[int, int] = (512, 384)):
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),          # [0, 1]
    ])

def mask_transform(size: Tuple[int, int] = (512, 384)):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


# --------------------------------------------------------------------------- #
#  Abstract base                                                               #
# --------------------------------------------------------------------------- #
class BaseTryOnDataset(Dataset, ABC):
    """Abstract base class for virtual try-on evaluation datasets."""

    def __init__(
        self,
        root: str,
        split: str = "test",
        img_size: Tuple[int, int] = (512, 384),
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transform = transform or default_transform(img_size)
        self.mask_tf = mask_transform(img_size)
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> list:
        """Return a list of dicts: {cloth_path, person_path, gt_path, mask_path(opt), id}"""
        ...

    def _load_image(self, path: Optional[Path]) -> Optional[Image.Image]:
        if path is None or not Path(path).exists():
            return None
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Optional[Path]) -> Optional[Image.Image]:
        if path is None or not Path(path).exists():
            return None
        return Image.open(path).convert("L")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        cloth_img  = self._load_image(sample.get("cloth_path"))
        person_img = self._load_image(sample.get("person_path"))
        gt_img     = self._load_image(sample.get("gt_path"))
        mask_img   = self._load_mask(sample.get("mask_path"))

        # Fallback: white mask if no mask is provided
        if mask_img is None:
            mask_img = Image.fromarray(np.ones(self.img_size, dtype=np.uint8) * 255)

        return {
            "cloth":  self.transform(cloth_img)  if cloth_img  else None,
            "person": self.transform(person_img) if person_img else None,
            "gt":     self.transform(gt_img)     if gt_img     else None,
            "mask":   self.mask_tf(mask_img),
            "meta":   {"id": sample.get("id", str(idx))},
        }
