"""
datasets/loaders.py
--------------------
10 concrete dataset loaders for Virtual Try-On evaluation.
Each loader extends BaseTryOnDataset and implements _load_samples().

Datasets:
  1.  VITON           – classic paired try-on benchmark (256×192)
  2.  VITON-HD        – high-res VITON variant (1024×768)
  3.  DressCode        – multi-category garment benchmark
  4.  MPV              – multi-pose virtual try-on
  5.  DeepFashion-TryOn – fashion compatibility pairs
  6.  ACGPN            – ACGPN dataset split
  7.  CP-VTON          – cloth-person VTON original split
  8.  HR-VTON          – high-resolution VTON
  9.  LaDI-VTON        – latent diffusion tryon dataset
  10. OVNet            – outfit-virtual-net dataset

HOW TO ADAPT
------------
Each loader reads the ACTUAL folder structure of the respective dataset.
Point `root` to the dataset root directory when instantiating the loader.
The docstring of each class describes the expected folder layout.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

from .base_dataset import BaseTryOnDataset
from .overlay_loaders import (
    OverlayDressCodeDataset,
    OverlayVITONHDDataset,
    OverlayLAIONDataset,
    OverlayStreetTryOnDataset,
    OverlayCurvTONDataset
)



# ─────────────────────────────────────────────────────────────────────────────
# 1. VITON
# ─────────────────────────────────────────────────────────────────────────────
class VITONDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/          <- person images (*.jpg)
            cloth/          <- garment images (*.jpg)
            image-parse/    <- segmentation masks (*.png)
          test_pairs.txt    <- "person_name.jpg cloth_name.jpg" per line
    The ground-truth is the person image itself (paired benchmark).
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                person_path = self.root / self.split / "image" / person_name
                cloth_path  = self.root / self.split / "cloth" / cloth_name
                gt_path     = person_path  # paired – person IS the GT
                mask_path   = self.root / self.split / "image-parse" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=person_path,
                    cloth_path=cloth_path,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 2. VITON-HD
# ─────────────────────────────────────────────────────────────────────────────
class VITONHDDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/          <- person images (*.jpg)
            cloth/          <- garment flat images (*.jpg)
            agnostic-mask/  <- binary body mask (*.png)
          test_pairs.txt
    """

    def _load_samples(self) -> List[Dict]:
        # Look for a pairs file at the dataset root level.
        pairs_file = None
        for candidate in (
            self.root / f"{self.split}_pairs.txt",
            self.root / f"{self.split}_pairs_paired.txt",
        ):
            if candidate.exists():
                pairs_file = candidate
                break

        split_root = self.root / self.split

        if pairs_file is None:
            # Fallback: scan image/ directory and pair each person with same-name cloth.
            image_dir = split_root / "image"
            if not image_dir.exists():
                raise FileNotFoundError(
                    f"Neither a pairs file nor '{split_root / 'image'}' found for "
                    f"split '{self.split}' in {self.root}"
                )
            all_images = sorted(
                f for f in os.listdir(image_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            samples = []
            for person_name in all_images:
                person_path = split_root / "image" / person_name
                cloth_path  = split_root / "cloth" / person_name
                # agnostic mask: prefer 'agnostic-mask' dir, fall back to 'agnostic'
                mask_path = split_root / "agnostic-mask" / person_name.replace(".jpg", "_mask.png")
                if not mask_path.exists():
                    mask_path = split_root / "agnostic" / person_name
                samples.append(dict(
                    id=os.path.splitext(person_name)[0],
                    person_path=person_path,
                    cloth_path=cloth_path if cloth_path.exists() else person_path,
                    gt_path=person_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
            return samples

        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                person_path = split_root / "image" / person_name
                cloth_path  = split_root / "cloth" / cloth_name
                gt_path     = person_path
                # agnostic mask: prefer 'agnostic-mask' dir, fall back to 'agnostic'
                mask_path = split_root / "agnostic-mask" / person_name.replace(".jpg", "_mask.png")
                if not mask_path.exists():
                    mask_path = split_root / "agnostic" / person_name
                samples.append(dict(
                    id=person_name,
                    person_path=person_path,
                    cloth_path=cloth_path,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 3. DressCode
# ─────────────────────────────────────────────────────────────────────────────
class DressCodeDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          upper_body/  (or lower_body / dresses)
            images/        <- person + GT images (id_0.jpg = person, id_1.jpg = GT)
            clothes/       <- flat garment images (id_1.jpg)
            skeletons/     <- pose skeletons (id_0.json) [optional]
            label_maps/    <- segmentation maps (id_0.png) [optional]
          <category>_pairs_<split>.txt
    `category` defaults to 'upper_body'.
    """

    def __init__(self, root, split="test", category="upper_body", **kwargs):
        self.category = category
        super().__init__(root, split, **kwargs)

    def _load_samples(self) -> List[Dict]:
        cat_dir = self.root / self.category

        # New structure: pairs files live inside each category folder.
        # Search order: test_pairs_paired.txt, test_pairs_unpaired.txt,
        # train_pairs.txt, then legacy root-level locations.
        pairs_file = None
        for candidate in (
            cat_dir / f"{self.split}_pairs_paired.txt",
            cat_dir / f"{self.split}_pairs_unpaired.txt",
            cat_dir / f"{self.split}_pairs.txt",
            cat_dir / "train_pairs.txt",
            self.root / f"{self.category}_pairs_{self.split}.txt",
            self.root / f"{self.split}_pairs.txt",
        ):
            if candidate.exists():
                pairs_file = candidate
                break

        if pairs_file is None:
            raise FileNotFoundError(
                f"No pairs file found for category '{self.category}', split '{self.split}' "
                f"in {cat_dir} or {self.root}"
            )

        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name = parts[0]                    # e.g. "020714_0.jpg"
                cloth_name  = parts[1] if len(parts) > 1 else parts[0]  # e.g. "020714_1.jpg"
                person_path = cat_dir / "image"  / person_name
                cloth_path  = cat_dir / "cloth"  / cloth_name
                gt_path     = person_path   # paired – person IS the GT
                mask_path   = cat_dir / "mask" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=os.path.splitext(person_name)[0],
                    person_path=person_path,
                    cloth_path=cloth_path,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. MPV (Multi-Pose Virtual Try-On)
# ─────────────────────────────────────────────────────────────────────────────
class MPVDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          MPV_dataset/
            image/         <- person images
            cloth/         <- garment images
            pose/          <- DensePose or OpenPose JSONs [optional]
          all_poseA_poseB_clothes.txt   <- "person cloth gt" TSV
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / "all_poseA_poseB_clothes.txt"
        if not pairs_file.exists():
            # Fallback
            possible = list(self.root.glob("*.txt"))
            if possible:
                pairs_file = possible[0]
            else:
                raise FileNotFoundError(f"No pairs file found in {self.root}")

        dataset_dir = self.root / "MPV_dataset"
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                person_name = parts[0]
                cloth_name  = parts[1]
                gt_name     = parts[2] if len(parts) > 2 else parts[0]

                samples.append(dict(
                    id=person_name,
                    person_path=dataset_dir / "image" / person_name,
                    cloth_path=dataset_dir / "cloth"  / cloth_name,
                    gt_path=dataset_dir / "image"     / gt_name,
                    mask_path=None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 5. DeepFashion-TryOn
# ─────────────────────────────────────────────────────────────────────────────
class DeepFashionTryOnDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/       <- person images
            cloth/       <- garment images
            gt/          <- ground-truth try-on results
            mask/        <- body/cloth masks [optional]
          test_pairs.txt   "person_img   cloth_img   gt_img"
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name = parts[0]
                cloth_name  = parts[1]
                gt_name     = parts[2] if len(parts) > 2 else parts[0]

                mask_path = base / "mask" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "gt"    / gt_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACGPN
# ─────────────────────────────────────────────────────────────────────────────
class ACGPNDataset(BaseTryOnDataset):
    """
    Expected structure (mirrors VITON naming kept by ACGPN paper):
        <root>/
          test/
            image/
            cloth/
            cloth-mask/   <- white/black garment mask
            image-parse/
          test_pairs.txt
    GT = person image (paired).
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                mask_path = base / "image-parse" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "image" / person_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 7. CP-VTON
# ─────────────────────────────────────────────────────────────────────────────
class CPVTONDataset(BaseTryOnDataset):
    """
    Expected structure (identical to VITON; re-defined separately for clarity):
        <root>/
          test/
            image/
            cloth/
            cloth-mask/
          test_pairs.txt
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "image" / person_name,
                    mask_path=None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 8. HR-VTON
# ─────────────────────────────────────────────────────────────────────────────
class HRVTONDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            image/            <- person images (1024×768)
            cloth/
            cloth_mask/
            agnostic/         <- agnostic-v3.2 person images
            agnostic_mask/
            image_parse_v3/   <- ATR or CIHP parsing maps (*.png)
          test_pairs.txt
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                mask_path = base / "image_parse_v3" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "image" / person_name,
                    cloth_path=base  / "cloth" / cloth_name,
                    gt_path=base     / "image" / person_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 9. LaDI-VTON
# ─────────────────────────────────────────────────────────────────────────────
class LaDIVTONDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            images/       <- person images
            clothes/      <- garment images
            gt/           <- ground-truth results (if available)
            masks/        <- segmentation masks
          test_pairs.txt   "person_name  cloth_name"
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                gt_path   = base / "gt"   / person_name
                if not gt_path.exists():
                    gt_path = base / "images" / person_name  # fallback
                mask_path = base / "masks" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "images"  / person_name,
                    cloth_path=base  / "clothes" / cloth_name,
                    gt_path=gt_path,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 10. OVNet
# ─────────────────────────────────────────────────────────────────────────────
class OVNetDataset(BaseTryOnDataset):
    """
    Expected structure:
        <root>/
          test/
            person/     <- wearing reference (person image)
            outfit/     <- target outfit / garment images
            gt/         <- ground-truth composed results
            mask/       <- body parsing mask
          test_pairs.txt  "person_name outfit_name gt_name"
    """

    def _load_samples(self) -> List[Dict]:
        pairs_file = self.root / f"{self.split}_pairs.txt"
        if not pairs_file.exists():
            raise FileNotFoundError(f"pairs file not found: {pairs_file}")

        base = self.root / self.split
        samples = []
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                person_name = parts[0]
                outfit_name = parts[1]
                gt_name     = parts[2] if len(parts) > 2 else parts[0]

                mask_path = base / "mask" / person_name.replace(".jpg", ".png")
                samples.append(dict(
                    id=person_name,
                    person_path=base / "person" / person_name,
                    cloth_path=base  / "outfit" / outfit_name,
                    gt_path=base     / "gt"     / gt_name,
                    mask_path=mask_path if mask_path.exists() else None,
                ))
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 11. StreetTryOn
# ─────────────────────────────────────────────────────────────────────────────
class StreetTryOnDataset(BaseTryOnDataset):
    """
    Street-style try-on dataset.

    Expected structure (benchmark_datasets/street_tryon/):
        <root>/
          train/
            image/
            segm_simplified_8labels/
            ...
          validation/
            image/
            segm_simplified_8labels/
            ...
          annotations/
            shop2street_test_pairs_top.csv
            street2street_test_pairs_*.csv
            ...

    Note: there is no explicit 'test/' split.  When split='test' the loader
    falls back to 'validation/' images.
    """

    def _load_samples(self) -> List[Dict]:
        # street_tryon has only train/ and validation/ — map 'test' → 'validation'
        actual_split = "validation" if self.split == "test" else self.split
        split_dir = self.root / actual_split
        image_dir = split_dir / "image"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}\n"
                f"Expected street_tryon root at: {self.root}"
            )

        samples = []
        for img_file in sorted(os.listdir(image_dir)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            person_path = image_dir / img_file
            # Use simplified segmentation as mask when available
            seg_path = split_dir / "segm_simplified_8labels" / img_file.replace(".jpg", ".png")
            samples.append(dict(
                id=os.path.splitext(img_file)[0],
                person_path=person_path,
                cloth_path=person_path,   # no separate garment images in this split
                gt_path=person_path,
                mask_path=seg_path if seg_path.exists() else None,
            ))
        return samples

# ─────────────────────────────────────────────────────────────────────────────
# 12. CurvTON (Custom Dataset)
# ─────────────────────────────────────────────────────────────────────────────
class CurvTONDataset(VITONHDDataset):
    """
    Custom CurvTON dataset.

    Supports two layouts:
    1) VITON-HD-like (fallback to parent loader):
         <root>/<split>/image, cloth, ... and optional pairs files
    2) CurvTON native difficulty/gender layout:
         <root>/
           easy|medium|hard/
             female|male/
               cloth_image/
               tryon_image/
               initial_person_image/

       It can also be pointed directly at one difficulty folder:
         <root>/easy/{female,male}/...
    """
    _DIFFICULTIES = ("easy", "medium", "hard")
    _GENDERS = ("female", "male")

    def _load_samples(self) -> List[Dict]:
        # First, try the existing VITON-HD behavior for backward compatibility.
        try:
            return super()._load_samples()
        except FileNotFoundError:
            pass

        # Then parse the native CurvTON folder structure.
        root = self.root

        # If root itself looks like a single difficulty folder (contains female/male),
        # only parse that folder; otherwise parse easy/medium/hard under root.
        if any((root / g).exists() for g in self._GENDERS):
            difficulty_roots = [root]
        else:
            difficulty_roots = [root / d for d in self._DIFFICULTIES if (root / d).exists()]

        if not difficulty_roots:
            raise FileNotFoundError(
                f"CurvTON root not recognized: {root}\n"
                f"Expected either VITON-HD style '<root>/<split>/image' or native "
                f"CurvTON style with easy/medium/hard and female/male subfolders."
            )

        samples: List[Dict] = []

        for diff_root in difficulty_roots:
            diff_name = diff_root.name.lower()
            for gender in self._GENDERS:
                g_root = diff_root / gender
                if not g_root.exists():
                    continue

                cloth_dir = g_root / "cloth_image"
                tryon_dir = g_root / "tryon_image"
                person_dir = g_root / "initial_person_image"

                if not (cloth_dir.exists() and tryon_dir.exists() and person_dir.exists()):
                    continue

                for tryon_name in sorted(os.listdir(tryon_dir)):
                    if not tryon_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue

                    tryon_path = tryon_dir / tryon_name
                    cloth_path = cloth_dir / tryon_name

                    # Expected naming example:
                    #   fh_008106_e03_fc_018511_kasavu_saree.png
                    #   mh_009219_e01_mc_005387_yukata.png
                    # Person ID is the first 3 tokens: fh_008106_e03 / mh_009219_e01
                    stem = Path(tryon_name).stem
                    parts = stem.split("_")
                    person_id = "_".join(parts[:3]) if len(parts) >= 3 else stem

                    # Prefer .png, then .jpg/.jpeg
                    person_path = person_dir / f"{person_id}.png"
                    if not person_path.exists():
                        for ext in (".jpg", ".jpeg"):
                            alt = person_dir / f"{person_id}{ext}"
                            if alt.exists():
                                person_path = alt
                                break

                    # Skip broken tuples rather than crashing whole dataset.
                    if not (person_path.exists() and cloth_path.exists() and tryon_path.exists()):
                        continue

                    samples.append(dict(
                        id=f"{diff_name}_{gender}_{stem}",
                        person_path=person_path,
                        cloth_path=cloth_path,
                        gt_path=tryon_path,
                        mask_path=None,
                    ))

        if not samples:
            raise FileNotFoundError(
                f"Found CurvTON-style directories under {root}, but no valid triplets were built.\n"
                f"Check that tryon_image and cloth_image filenames align and that person ids "
                f"can be resolved in initial_person_image."
            )

        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 13. Qwen Camera Edit Control Results
# ─────────────────────────────────────────────────────────────────────────────
class QwenCameraEditResultsDataset(BaseTryOnDataset):
    """
    Loader for generated result folders shaped like:

        <root>/
          easy/
            easy_000/
              img1_out0000.png
              ...
              latency.json
            ...
          medium/
            medium_000/
            ...
          hard/
            hard_000/
            ...

    It also supports pointing root directly at a single difficulty folder
    (e.g. <root>/easy) or at a shard folder (e.g. <root>/easy/easy_000).

    Since this dataset is generation-output-only, person/cloth/gt are all set to
    the same output image path so metric modules can consume standard keys.
    """

    _DIFFICULTIES = ("easy", "medium", "hard")
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

    def _load_samples(self) -> List[Dict]:
        root = self.root
        split_l = str(self.split).lower()

        # Collect candidate difficulty roots.
        difficulty_roots: List[Path] = []
        if any((root / d).is_dir() for d in self._DIFFICULTIES):
            # Root points to the top-level "results" folder.
            selected = self._DIFFICULTIES if split_l in {"all", "test", "train", "val", "validation"} else (split_l,)
            for d in selected:
                p = root / d
                if p.is_dir():
                    difficulty_roots.append(p)
        elif root.name.lower() in self._DIFFICULTIES:
            # Root points directly to one difficulty folder.
            difficulty_roots = [root]
        else:
            # Root might already be one shard directory (e.g. easy_000).
            difficulty_roots = [root]

        if not difficulty_roots:
            raise FileNotFoundError(
                f"Qwen camera-results root not recognized: {root}. "
                f"Expected either '<root>/easy|medium|hard' or a direct difficulty directory."
            )

        samples: List[Dict] = []

        for diff_root in difficulty_roots:
            diff_name = diff_root.name.lower()

            # Prefer shard dirs named like easy_000, medium_004, ...
            shard_dirs = [
                d for d in sorted(diff_root.iterdir())
                if d.is_dir() and d.name.lower().startswith(f"{diff_name}_")
            ]
            if not shard_dirs:
                shard_dirs = [diff_root]

            for shard in shard_dirs:
                for img in sorted(shard.iterdir()):
                    if not img.is_file():
                        continue
                    if img.suffix.lower() not in self._IMAGE_EXTS:
                        continue

                    rel_id = f"{diff_name}/{shard.name}/{img.stem}"
                    samples.append(dict(
                        id=rel_id,
                        person_path=img,
                        cloth_path=img,
                        gt_path=img,
                        mask_path=None,
                    ))

        if not samples:
            raise FileNotFoundError(
                f"No images found under {root}. Expected output image files inside difficulty folders."
            )

        return samples


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "viton":              VITONDataset,
    "vton":               VITONDataset,
    "viton_hd":           VITONHDDataset,
    "vitonhd":            VITONHDDataset,
    "dresscode":          DressCodeDataset,
    "mpv":                MPVDataset,
    "deepfashion_tryon":  DeepFashionTryOnDataset,
    "deepfashion":        DeepFashionTryOnDataset,
    "acgpn":              ACGPNDataset,
    "cp_vton":            CPVTONDataset,
    "hr_vton":            HRVTONDataset,
    "ladi_vton":          LaDIVTONDataset,
    "ovnet":              OVNetDataset,

    "street_tryon":       StreetTryOnDataset,
    "curvton":            CurvTONDataset,
    "qwen_camera_results": QwenCameraEditResultsDataset,
    "qwen_camera_edit_control": QwenCameraEditResultsDataset,
    "qwen_camera": QwenCameraEditResultsDataset,

    # Loaders with overlay generation (datasets/overlay_loaders.py)
    "dresscode_overlay":    OverlayDressCodeDataset,
    "vitonhd_overlay":      OverlayVITONHDDataset,
    "laion_overlay":        OverlayLAIONDataset,
    "laion":              OverlayLAIONDataset,
    "laion_fashion":      OverlayLAIONDataset,
    "street_tryon_overlay": OverlayStreetTryOnDataset,
    "curvton_overlay":      OverlayCurvTONDataset,
}

_LAION_NAMES = {
    "laion", "laion_overlay", "laion_fashion",
}


def get_dataset(name: str, root: str, **kwargs):
    """
    Factory function; returns the appropriate dataset by name.

    Handles two kinds of constructors:
      1. BaseTryOnDataset subclasses   → (root=, split=, img_size=, ...)
      2. LAION streaming loaders       → (split=, limit=, img_size=, ...)
    """
    name = name.lower().replace("-", "_")
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
        )

    # ── LAION (no root needed) ────────────────────────────────────────────
    if name in _LAION_NAMES:
        return DATASET_REGISTRY[name](**kwargs)

    # ── Standard BaseTryOnDataset subclasses ──────────────────────────────
    # Translate 'dresscode_category' → 'category' for DressCode loaders and
    # strip any keys that BaseTryOnDataset.__init__ doesn't accept.
    _DRESSCODE_NAMES = {"dresscode", "dresscode_overlay"}
    if name in _DRESSCODE_NAMES and "dresscode_category" in kwargs:
        kwargs = dict(kwargs)
        kwargs["category"] = kwargs.pop("dresscode_category")

    # Keys that every BaseTryOnDataset accepts; drop everything else so
    # unrecognised kwargs (e.g. run_pose, use_overlay …) don't propagate.
    _BASE_KEYS = {"split", "img_size", "transform", "category"}
    clean_kwargs = {k: v for k, v in kwargs.items() if k in _BASE_KEYS}

    return DATASET_REGISTRY[name](root=root, **clean_kwargs)
