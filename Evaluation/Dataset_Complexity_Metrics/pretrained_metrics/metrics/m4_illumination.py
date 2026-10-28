"""
metrics/m4_illumination.py
===========================
Metric 4 — Illumination Complexity (improved)
---------------------------------------------

Improvements over the original implementation:
- Use linearized sRGB luminance (Rec.709 Y) instead of uint8 LAB conversion to avoid quantisation.
- Optional person masking: compute statistics only over the person region to avoid background contamination.
- Gaussian smoothing before Sobel to reduce texture responses.
- Normalise gradient variance by mean gradient (scale-robust) and support MAD for robust luminance spread.

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]
person_masks (optional) : torch.Tensor (B, H, W) or (B,1,H,W) bool or {0,1}

Returns (compute())
--------------------
dict with:
    luminance_mean_global       : mean of per-image mean-L
    luminance_var_global        : robust spread (MAD) or variance of per-image mean-L
    illumination_gradient_mean  : mean of per-image (normalised) gradient-variance
    illumination_richness_score : lighting/shadow richness (gradient-driven, brightness-decoupled)
    illumination_complexity     : luminance_var_global + illumination_gradient_mean
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0,1] to linear RGB (per-channel)."""
    c = np.clip(c, 0.0, 1.0)
    mask = c <= 0.04045
    out = np.empty_like(c, dtype=np.float32)
    out[mask] = c[mask] / 12.92
    out[~mask] = ((c[~mask] + 0.055) / 1.055) ** 2.4
    return out


def _rgb_to_luminance(imgs: torch.Tensor) -> (np.ndarray, List[np.ndarray]):
    """
    imgs : (B, 3, H, W)  float32  [0,1]
    Returns per-image mean luminance (Rec.709 Y) and list of luminance maps (H,W) float32.
    """
    B = imgs.shape[0]
    mean_L = np.zeros(B, dtype=np.float32)
    L_maps = []

    for i in range(B):
        rgb = imgs[i].permute(1, 2, 0).numpy().astype(np.float32)  # (H,W,3)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_lin = _srgb_to_linear(rgb)
        # Rec.709 luminance (Y) from linear RGB
        Y = 0.2126 * rgb_lin[:, :, 0] + 0.7152 * rgb_lin[:, :, 1] + 0.0722 * rgb_lin[:, :, 2]
        mean_L[i] = float(Y.mean())
        L_maps.append(Y.astype(np.float32))

    return mean_L, L_maps


def _sobel_gradient_stats(L_map: np.ndarray, blur_sigma: float = 1.0, eps: float = 1e-8) -> (float, float):
    """
    L_map : (H, W) float
    Returns (var, mean) of gradient magnitude after gaussian smoothing.
    """
    if blur_sigma > 0:
        k = max(3, int(6 * blur_sigma + 1))
        if k % 2 == 0:
            k += 1
        L_blur = cv2.GaussianBlur(L_map, (k, k), blur_sigma)
    else:
        L_blur = L_map

    gx = cv2.Sobel(L_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag.var()), float(mag.mean() + eps)


def _mad(arr: np.ndarray) -> float:
    """Median absolute deviation (scalar)."""
    arr = np.asarray(arr)
    return float(np.median(np.abs(arr - np.median(arr))))


# ─────────────────────────────────────────────────────────────────────────────
# IlluminationMetrics
# ─────────────────────────────────────────────────────────────────────────────

class IlluminationMetrics:

    def __init__(self):
        # configuration for improved metric
        self.blur_sigma = 1.0
        self.use_mad = True
        self.normalize_grad = True
        self.eps = 1e-8

        self._mean_L: List[float] = []
        self._grad_var: List[float] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor, person_masks: torch.Tensor | None = None):
        """
        person_imgs : (B, 3, H, W)  float32  [0,1]
        person_masks : optional (B, H, W) or (B,1,H,W) boolean mask where True=person
        """
        mean_L, L_maps = _rgb_to_luminance(person_imgs.cpu())

        B = person_imgs.shape[0]
        masks = None
        if person_masks is not None:
            pm = person_masks.cpu().numpy()
            if pm.ndim == 4:
                pm = pm[:, 0]
            masks = (pm > 0.5)

        for i in range(len(mean_L)):
            L = L_maps[i]
            if masks is not None:
                m = masks[i]
                if m.sum() >= 10:
                    mean_L_i = float(L[m].mean())
                else:
                    mean_L_i = float(L.mean())
            else:
                mean_L_i = float(mean_L[i])

            gv_var, gv_mean = _sobel_gradient_stats(L, blur_sigma=self.blur_sigma, eps=self.eps)

            if masks is not None and masks[i].sum() >= 10:
                # compute stats only inside mask for gradient
                mag_var = _compute_masked_grad_var(L, masks[i], blur_sigma=self.blur_sigma, eps=self.eps)
                # if masked mean gradient returned, normalise by its mean
                g_mean = _compute_masked_grad_mean(L, masks[i], blur_sigma=self.blur_sigma, eps=self.eps)
                gv_var = mag_var
                gv_mean = g_mean + self.eps

            if self.normalize_grad:
                grad_metric = gv_var / (gv_mean + self.eps)
            else:
                grad_metric = gv_var

            self._mean_L.append(mean_L_i)
            self._grad_var.append(float(grad_metric))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        if not self._mean_L:
            return {k: float("nan") for k in [
                "luminance_mean_global", "luminance_var_global",
                "illumination_gradient_mean", "illumination_richness_score",
                "illumination_complexity",
            ]}
        arr_L = np.array(self._mean_L)
        arr_gv = np.array(self._grad_var)

        lum_mean = float(arr_L.mean())
        if self.use_mad:
            lum_var = float(_mad(arr_L))
        else:
            lum_var = float(arr_L.var())
        grad_mean = float(arr_gv.mean())

        return {
            "luminance_mean_global": lum_mean,
            "luminance_var_global": lum_var,
            "illumination_gradient_mean": grad_mean,
            "illumination_richness_score": grad_mean,
            "illumination_complexity": lum_var + grad_mean,
        }

    def reset(self):
        self._mean_L.clear()
        self._grad_var.clear()


def _compute_masked_grad_var(L_map: np.ndarray, mask: np.ndarray, blur_sigma: float = 1.0, eps: float = 1e-8) -> float:
    # compute gradient mag and return variance over masked pixels
    if blur_sigma > 0:
        k = max(3, int(6 * blur_sigma + 1))
        if k % 2 == 0:
            k += 1
        L_blur = cv2.GaussianBlur(L_map, (k, k), blur_sigma)
    else:
        L_blur = L_map
    gx = cv2.Sobel(L_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    m = mask.astype(bool)
    if m.sum() == 0:
        return float(mag.var())
    return float(mag[m].var())


def _compute_masked_grad_mean(L_map: np.ndarray, mask: np.ndarray, blur_sigma: float = 1.0, eps: float = 1e-8) -> float:
    # compute gradient mag and return mean over masked pixels
    if blur_sigma > 0:
        k = max(3, int(6 * blur_sigma + 1))
        if k % 2 == 0:
            k += 1
        L_blur = cv2.GaussianBlur(L_map, (k, k), blur_sigma)
    else:
        L_blur = L_map
    gx = cv2.Sobel(L_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    m = mask.astype(bool)
    if m.sum() == 0:
        return float(mag.mean() + eps)
    return float(mag[m].mean() + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible aliases used by test.py and EDA code
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_lab_l(imgs: torch.Tensor):
    """Alias for _rgb_to_luminance (legacy name used by EDA code)."""
    return _rgb_to_luminance(imgs)


def _sobel_gradient_variance(L_map: np.ndarray) -> float:
    """Alias: returns the variance component from _sobel_gradient_stats."""
    var, _ = _sobel_gradient_stats(L_map)
    return var
