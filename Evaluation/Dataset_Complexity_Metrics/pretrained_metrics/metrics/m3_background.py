"""
metrics/m3_background.py
=========================
Metric 3 — Background Complexity
----------------------------------

3A. Background Texture Entropy
    H_bg = -Σ p_j log p_j  (histogram entropy over 256-bin grayscale)

3B. Background Object Density
    C_obj = E[n_i]  where n_i = # objects detected in the background region

Pretrained models
------------------
- DeepLabV3-ResNet101 (torchvision) → person/background segmentation
- DETR (facebook/detr-resnet-50)    → object detection for density
  Falls back to connected-components counting when DETR is unavailable.

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (via compute())
------------------------
dict with:
    bg_entropy_mean              : mean texture entropy across dataset
    bg_entropy_var               : variance of texture entropy
    bg_object_density_mean       : mean #objects in background per image
    bg_semantic_entropy_mean     : mean semantic class entropy (DETR classes)
    bg_semantic_entropy_var      : variance of semantic class entropy
    bg_semantic_unique_mean      : mean #unique semantic classes per image
    bg_semantic_unique_var       : variance of #unique semantic classes
    bg_complexity_3A             : bg_entropy_mean  (alias)
    bg_complexity_3B             : bg_object_density_mean  (alias)
    bg_complexity_semantic       : bg_semantic_entropy_mean (alias)
    bg_semantic_entropy_global   : global entropy over all detections
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# DeepLabV3 person-masker
# ─────────────────────────────────────────────────────────────────────────────

class _PersonSegmenter:
    """Returns a binary person mask (B, H, W) from DeepLabV3."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._load()

    def _load(self):
        try:
            import torchvision.models.segmentation as s
            self._model = s.deeplabv3_resnet101(
                weights=s.DeepLabV3_ResNet101_Weights.DEFAULT
            ).to(self.device).eval()
            print("[BackgroundMetric] DeepLabV3 loaded for person segmentation.")
        except Exception as e:
            raise RuntimeError(
                "[BackgroundMetric] DeepLabV3 is required but unavailable."
            ) from e

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Returns (B, H, W) bool tensor — True = person pixel."""
        # Accept either BCHW or BHWC and normalize to BCHW.
        if imgs.ndim != 4:
            raise RuntimeError(f"[BackgroundMetric] Expected 4D tensor, got shape={tuple(imgs.shape)}")
        if imgs.shape[1] != 3 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
        if imgs.shape[1] != 3:
            raise RuntimeError(f"[BackgroundMetric] Expected channel dimension C=3, got shape={tuple(imgs.shape)}")
        B, C, H, W = imgs.shape
        if self._model is not None:
            norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            x    = torch.stack([norm(im) for im in imgs]).to(self.device)
            out  = self._model(x)["out"]          # (B, 21, H, W)
            pred = out.argmax(1)                   # (B, H, W)
            return (pred == 15).cpu()              # class 15 = person

        # Proxy: centre of the image ≈ person
        mask = torch.zeros(B, H, W, dtype=torch.bool)
        ch, cw = H // 4, W // 4
        mask[:, ch: H - ch, cw: W - cw] = True
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Texture entropy
# ─────────────────────────────────────────────────────────────────────────────

def _texture_entropy(bg_rgb: torch.Tensor, person_mask: torch.Tensor) -> float:
    """
    bg_rgb      : (3, H, W) float [0,1]
    person_mask : (H, W) bool — True = person (exclude)
    Returns scalar entropy of background grayscale histogram.
    """
    # Grayscale
    gray = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]  # (H,W)
    bg_pixels = gray[~person_mask].numpy()

    if bg_pixels.size == 0:
        return float("nan")

    bg_pixels = np.clip(bg_pixels, 0.0, 1.0)
    counts, _ = np.histogram(bg_pixels, bins=256, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return float("nan")
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def _semantic_entropy(class_ids: List[int]) -> float:
    if not class_ids:
        return float("nan")
    ids = np.array(class_ids, dtype=int)
    uniq, counts = np.unique(ids, return_counts=True)
    p = counts.astype(float) / counts.sum()
    return float(-np.sum(p * np.log(p)))


# ─────────────────────────────────────────────────────────────────────────────
# Object detector
# ─────────────────────────────────────────────────────────────────────────────

class _ObjectDetector:
    """DETR → number of objects in background region."""

    CONF_THRESHOLD = 0.5

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self._model   = None
        self._feature = None
        self._backend = "stub"
        self._load()

    def _load(self):
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self._feature = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50"
            )
            self._model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50"
            ).to(self.device).eval()
            self._backend = "detr"
            print("[BackgroundMetric] DETR loaded for object density.")
        except Exception as e:
            raise RuntimeError(
                "[BackgroundMetric] DETR is required but unavailable."
            ) from e

    @torch.no_grad()
    def count_objects(
        self, imgs: torch.Tensor, person_masks: torch.Tensor
    ) -> List[int]:
        """
        imgs         : (B, 3, H, W) float [0,1]
        person_masks : (B, H, W  )  bool
        Returns List[int] — number of background objects per image.
        """
        B = imgs.shape[0]

        if self._backend == "detr":
            import torchvision.transforms.functional as TF

            # Batch person-masking on CPU
            imgs_masked = imgs.clone()
            bg_mask = (~person_masks).unsqueeze(1).float()  # (B, 1, H, W)
            imgs_masked = imgs_masked * bg_mask  # zero out person pixels

            # PIL conversion (all at once)
            pils = []
            for i in range(B):
                pil = TF.to_pil_image(imgs_masked[i].clamp(0, 1).cpu()).convert("RGB")
                if pil.width < 32 or pil.height < 32:
                    pil = pil.resize((224, 224))
                pils.append(pil)

            # Batched DETR forward (single pass)
            inputs = self._feature(
                images=pils,
                return_tensors="pt",
            ).to(self.device)
            outs = self._model(**inputs)

            # Per-image confidence thresholding (cheap CPU indexing)
            counts = []
            for i in range(B):
                probs = outs.logits.softmax(-1)[i, :, :-1]
                conf  = probs.max(-1).values
                n_obj = int((conf > self.CONF_THRESHOLD).sum().item())
                counts.append(n_obj)
            return counts
        raise RuntimeError("[BackgroundMetric] No valid object detector backend available.")

    @torch.no_grad()
    def detect_classes(
        self, imgs: torch.Tensor, person_masks: torch.Tensor
    ) -> List[List[int]]:
        """
        Returns per-image list of DETR class ids above confidence threshold.
        """
        if self._backend != "detr":
            raise RuntimeError("[BackgroundMetric] No valid object detector backend available.")

        import torchvision.transforms.functional as TF

        B = imgs.shape[0]
        imgs_masked = imgs.clone()
        bg_mask = (~person_masks).unsqueeze(1).float()  # (B, 1, H, W)
        imgs_masked = imgs_masked * bg_mask  # zero out person pixels

        pils = []
        for i in range(B):
            pil = TF.to_pil_image(imgs_masked[i].clamp(0, 1).cpu()).convert("RGB")
            if pil.width < 32 or pil.height < 32:
                pil = pil.resize((224, 224))
            pils.append(pil)

        inputs = self._feature(
            images=pils,
            return_tensors="pt",
        ).to(self.device)
        outs = self._model(**inputs)

        classes = []
        for i in range(B):
            probs = outs.logits.softmax(-1)[i, :, :-1]
            conf = probs.max(-1).values
            labels = probs.argmax(-1)
            keep = conf > self.CONF_THRESHOLD
            classes.append(labels[keep].cpu().tolist())
        return classes

    def _component_count(
        self, imgs: torch.Tensor, person_masks: torch.Tensor
    ) -> List[int]:
        """Count connected edge-components in background as object proxy."""
        try:
            from skimage.measure import label as sk_label
        except ImportError:
            return [0] * imgs.shape[0]

        counts = []
        for i in range(imgs.shape[0]):
            gray = (0.299 * imgs[i, 0] + 0.587 * imgs[i, 1]
                    + 0.114 * imgs[i, 2]).numpy()
            bg   = ~person_masks[i].numpy()
            gray_bg = gray * bg

            # Sobel gradient magnitude
            from scipy.ndimage import sobel
            gx = sobel(gray_bg, axis=0)
            gy = sobel(gray_bg, axis=1)
            mag = np.sqrt(gx ** 2 + gy ** 2)

            thr = np.percentile(mag[bg], 75) if bg.any() else 0.0
            edge_map = (mag > thr) & bg

            labeled  = sk_label(edge_map)
            n_comp   = int(labeled.max())
            counts.append(n_comp)
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# BackgroundMetrics
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundMetrics:
    """Accumulates background texture entropy and object density."""

    def __init__(self, device: str = "cpu"):
        self._segmenter = _PersonSegmenter(device)
        self._detector  = _ObjectDetector(device)
        self._entropies: List[float] = []
        self._obj_counts: List[int]  = []
        self._semantic_entropies: List[float] = []
        self._semantic_uniques: List[int] = []
        self._semantic_all: List[int] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        # Be robust to dataloaders that emit BHWC tensors.
        if person_imgs.ndim != 4:
            raise RuntimeError(f"[BackgroundMetric] Expected 4D tensor, got shape={tuple(person_imgs.shape)}")
        if person_imgs.shape[1] != 3 and person_imgs.shape[-1] == 3:
            person_imgs = person_imgs.permute(0, 3, 1, 2).contiguous()
        if person_imgs.shape[1] != 3:
            raise RuntimeError(f"[BackgroundMetric] Expected channel dimension C=3, got shape={tuple(person_imgs.shape)}")

        person_masks = self._segmenter(person_imgs)      # (B,H,W) bool
        obj_counts   = self._detector.count_objects(person_imgs, person_masks)
        class_lists  = self._detector.detect_classes(person_imgs, person_masks)

        for i in range(person_imgs.shape[0]):
            ent = _texture_entropy(person_imgs[i], person_masks[i])
            self._entropies.append(ent)

            sem_ent = _semantic_entropy(class_lists[i])
            self._semantic_entropies.append(sem_ent)
            self._semantic_uniques.append(len(set(class_lists[i])))
            self._semantic_all.extend(class_lists[i])

        self._obj_counts.extend(obj_counts)

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        ent = np.array([v for v in self._entropies if not math.isnan(v)])
        obj = np.array(self._obj_counts, dtype=float)
        sem_ent = np.array([v for v in self._semantic_entropies if not math.isnan(v)])
        sem_uniq = np.array(self._semantic_uniques, dtype=float)
        sem_global = _semantic_entropy(self._semantic_all)

        # ── Sub-component means ───────────────────────────────────────────
        ent_mean  = float(ent.mean())  if len(ent)  else float("nan")
        ent_var   = float(ent.var())   if len(ent)  else float("nan")
        obj_mean  = float(obj.mean())  if len(obj)  else float("nan")
        sem_mean  = float(sem_ent.mean()) if len(sem_ent) else float("nan")
        sem_var   = float(sem_ent.var())  if len(sem_ent) else float("nan")
        uniq_mean = float(sem_uniq.mean()) if len(sem_uniq) else float("nan")
        uniq_var  = float(sem_uniq.var())  if len(sem_uniq) else float("nan")

        # ── Overall background complexity score (0-1) ─────────────────────
        #   Three normalised pillars combined with weights:
        #     1. Texture entropy  (max ~ 8 for 8-bit grayscale patches)
        #     2. Object density   (clamped at 20 objects)
        #     3. Semantic entropy  (max = log2(91 COCO classes) ~ 6.5)
        #
        #   Higher -> more complex / cluttered background.
        MAX_TEXTURE_ENT = 8.0
        MAX_OBJ_COUNT   = 20.0
        MAX_SEM_ENT     = math.log2(91)  # COCO has 91 class ids

        parts, weights = [], []
        if not math.isnan(ent_mean):
            parts.append(min(ent_mean / MAX_TEXTURE_ENT, 1.0))
            weights.append(0.35)
        if not math.isnan(obj_mean):
            parts.append(min(obj_mean / MAX_OBJ_COUNT, 1.0))
            weights.append(0.35)
        if not math.isnan(sem_mean):
            parts.append(min(sem_mean / MAX_SEM_ENT, 1.0))
            weights.append(0.30)

        if parts:
            w_sum = sum(weights)
            overall = sum(p * w for p, w in zip(parts, weights)) / w_sum
        else:
            overall = float("nan")

        # Human-readable difficulty label
        if math.isnan(overall):
            label_val = float("nan")
        elif overall < 0.30:
            label_val = 1.0   # simple
        elif overall < 0.60:
            label_val = 2.0   # moderate
        else:
            label_val = 3.0   # complex

        return {
            # ── Overall score ─────────────────────────────────────────────
            "bg_overall_complexity":       overall,        # 0-1 composite
            "bg_difficulty_label":         label_val,      # 1=simple 2=moderate 3=complex

            # ── Texture (3A) ──────────────────────────────────────────────
            "bg_entropy_mean":             ent_mean,
            "bg_entropy_var":              ent_var,
            "bg_complexity_3A":            ent_mean,

            # ── Object density (3B) ───────────────────────────────────────
            "bg_object_density_mean":      obj_mean,
            "bg_complexity_3B":            obj_mean,

            # ── Semantic diversity ────────────────────────────────────────
            "bg_semantic_entropy_mean":    sem_mean,
            "bg_semantic_entropy_var":     sem_var,
            "bg_semantic_unique_mean":     uniq_mean,
            "bg_semantic_unique_var":      uniq_var,
            "bg_complexity_semantic":      sem_mean,
            "bg_semantic_entropy_global":  float(sem_global),
        }

    def reset(self):
        self._entropies.clear()
        self._semantic_entropies.clear()
        self._semantic_uniques.clear()
        self._semantic_all.clear()
        self._obj_counts.clear()
