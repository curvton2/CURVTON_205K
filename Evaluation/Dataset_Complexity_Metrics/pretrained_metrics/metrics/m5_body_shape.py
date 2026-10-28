"""
metrics/m5_body_shape.py
=========================
Metric 5 - Body Shape Diversity
---------------------------------
Uses SMPL shape coefficients beta in R^10 extracted from HMR2.0 or a proxy.

D_shape = log det(Cov(beta) + eps*I)

Pretrained model
-----------------
HMR2.0 (4D-Humans) - direct successor to SPIN from the same research lineage.
Weights are downloaded automatically on first use.
  pip install git+https://github.com/shubham-goel/4D-Humans.git

Backend priority:
  Level 1 - HMR2.0:       Genuine beta in R^10 SMPL shape coefficients.
             Requires:  pip install git+https://github.com/shubham-goel/4D-Humans.git
             Weights:   auto-downloaded on first use, cached to ~/.cache/4DHumans/
             SMPL file: place basicModel_neutral_lbs_10_207_0_v1.0.0.pkl at
                        ~/.cache/4DHumans/data/  (download from smplify.is.tue.mpg.de)
  Level 2 - ViT-B/16 proxy:  10-D projection of ViT CLS embedding.
             NOT equivalent to SMPL beta, but captures body-shape variety.
             Requires:  pip install timm
  Level 3 - Random stub (smoke-test only).

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (compute())
--------------------
dict with:
    shape_diversity_logdet   : log det(Cov(beta) + eps*I)
    shape_variance_total     : sum of eigenvalues of Cov(beta)  (total variance)
    shape_dims               : number of shape coefficients (10)
    backend                  : "hmr2", "vit_proxy", or "stub"
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from pretrained_metrics.cache_setup import configure_model_caches, ensure_hmr2_smpl_model


# -----------------------------------------------------------------------------
# Shape extractor - backend priority: HMR2.0 -> ViT proxy -> stub
# -----------------------------------------------------------------------------

class _ShapeExtractor:
    SHAPE_DIM = 10

    _NORMALIZE = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self._backend = "stub"

        # HMR2.0
        self._hmr2_model = None
        self._hmr2_cfg   = None

        # ViT proxy
        self._vit      = None
        self._vit_proj: Optional[nn.Linear] = None

        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        if self._try_hmr2():
            return
        if self._try_vit():
            return
        raise RuntimeError(
            "[BodyShapeMetric] No valid backend available. "
            "Install HMR2.0 (4D-Humans) or timm for ViT proxy."
        )

    # ------------------------------------------------------------------ #
    def _try_hmr2(self) -> bool:
        try:
            cache_info = configure_model_caches(set_home_for_hmr2=True)
            smpl_info = ensure_hmr2_smpl_model(cache_info["base_path"])
            from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT

            # PyTorch 2.6 changed torch.load default to weights_only=True.
            # HMR2.0 checkpoints embed omegaconf objects which are not in the
            # default allowlist. Register them before loading.
            import torch.serialization as _ts
            from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig
            _ts.add_safe_globals([_DictConfig, _ListConfig])

            # download_models() fetches the checkpoint if not already cached.
            # Returns None - the path is exposed via DEFAULT_CHECKPOINT.
            download_models()
            self._hmr2_model, self._hmr2_cfg = load_hmr2(DEFAULT_CHECKPOINT)
            self._hmr2_model = self._hmr2_model.to(self.device).eval()

            self._backend = "hmr2"
            print(f"[BodyShapeMetric] HMR2.0 loaded (cache: {cache_info['fourdhumans_cache']}).")
            print(f"[BodyShapeMetric] SMPL path status: {smpl_info['status']} -> {smpl_info['dst']}")
            return True

        except ImportError:
            print("[BodyShapeMetric] HMR2.0 not installed. "
                  "Run: pip install git+https://github.com/shubham-goel/4D-Humans.git  "
                  "Falling back to ViT proxy.")
            return False

        except Exception as e:
            print(f"[BodyShapeMetric] HMR2.0 initialisation failed: {e}. "
                  "Falling back to ViT proxy.")
            self._hmr2_model = None
            return False

    # ------------------------------------------------------------------ #
    def _try_vit(self) -> bool:
        try:
            import timm
            self._vit = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            ).to(self.device).eval()
            torch.manual_seed(0)
            self._vit_proj = nn.Linear(768, self.SHAPE_DIM, bias=False).to(self.device)
            nn.init.orthogonal_(self._vit_proj.weight)
            self._vit_proj.eval()
            self._backend = "vit_proxy"
            print("[BodyShapeMetric] Using ViT-B/16 proxy for body shape (no HMR2.0).")
            return True
        except Exception as e:
            print(f"[BodyShapeMetric] ViT unavailable ({e}).")
            return False

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns (B, 10) numpy float32 - SMPL beta or proxy.
        """
        B = imgs.shape[0]

        if self._backend == "hmr2":
            return self._hmr2_forward(imgs)

        if self._backend == "vit_proxy":
            x = TF.resize(imgs, [224, 224]).to(self.device)
            x = torch.stack([self._NORMALIZE(im) for im in x])
            feats = self._vit(x)              # (B, 768)
            betas = self._vit_proj(feats)     # (B, 10)
            return betas.cpu().numpy()
        raise RuntimeError("[BodyShapeMetric] No valid body-shape backend available.")

    # ------------------------------------------------------------------ #
    def _hmr2_forward(self, imgs: torch.Tensor) -> np.ndarray:
        """
        HMR2.0 ViT backbone internally crops 32 px from each side of the width:
            x[:, :, :, 32:-32]
        So the model needs input (B, 3, 256, 256); after the crop the effective
        size is 192 wide x 256 tall  ->  12 x 16 = 192 patches, which matches
        the pretrained pos_embed of size 192.
        Passing [192, 256] instead produced a 192x192 crop (144 patches) and
        caused the RuntimeError: size 144 != 192.
        """
        # Resize to 256x256 so the internal 32-px crop leaves 192 wide x 256 tall
        x = TF.resize(imgs, [256, 256]).to(self.device)
        x = torch.stack([self._NORMALIZE(im) for im in x])

        batch = {"img": x}
        out   = self._hmr2_model(batch)

        # 'pred_smpl_params' is always present; betas shape is (B, 10)
        betas = out["pred_smpl_params"]["betas"]   # (B, 10)
        return betas.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# BodyShapeMetrics  (public API - unchanged)
# -----------------------------------------------------------------------------

class BodyShapeMetrics:

    def __init__(self, device: str = "cpu", eps: float = 1e-6):
        self._extractor = _ShapeExtractor(device)
        self.eps = eps
        self._betas: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """person_imgs : (B, 3, H, W)  float32  [0,1]"""
        betas = self._extractor(person_imgs)    # (B, 10)
        for b in betas:
            self._betas.append(b)

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        if len(self._betas) < 2:
            return {k: float("nan") for k in [
                "shape_diversity_logdet", "shape_variance_total",
                "shape_dims", "backend",
            ]}

        B_mat = np.stack(self._betas, axis=0)       # (N, 10)
        mu    = B_mat.mean(axis=0, keepdims=True)
        Bc    = B_mat - mu
        cov   = (Bc.T @ Bc) / max(len(B_mat) - 1, 1)   # (10, 10)
        reg   = cov + self.eps * np.eye(10)

        sign, log_det = np.linalg.slogdet(reg)
        d_shape   = float(log_det) if sign > 0 else float("nan")
        d_shape_norm = (d_shape / 10.0) if (not np.isnan(d_shape)) else float("nan")
        total_var = float(np.linalg.eigvalsh(reg).sum())

        return {
            "shape_diversity_logdet": d_shape,
            "shape_diversity_logdet_raw": d_shape,
            "shape_diversity_logdet_normalized": d_shape_norm,
            "shape_variance_total":   total_var,
            "shape_dims":             float(self._extractor.SHAPE_DIM),
            "backend":                self._extractor._backend,
        }

    def reset(self):
        self._betas.clear()
