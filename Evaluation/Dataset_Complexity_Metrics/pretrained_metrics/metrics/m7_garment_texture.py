"""
metrics/m7_garment_texture.py
==============================
Metric 7 - Garment Texture Diversity (ensemble).

This version uses two garment encoders when available:
  1) FashionCLIP (patrickjohncyh/fashion-clip)
  2) DINOv2 (facebook/dinov2-base)

Final garment metrics are weighted averages over backend-specific metrics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF


class _GarmentEncoder:
    """
    Multi-backend garment encoder.
    Returns dict[backend_name] -> embeddings (B, D) as numpy arrays.
    """

    def __init__(self, device: str = "cpu", w_fashion_clip: float = 0.5, w_dinov2: float = 0.5):
        self.device = device
        self._models: Dict[str, object] = {}
        self._processors: Dict[str, object] = {}
        self.weights = {
            "fashion_clip": float(w_fashion_clip),
            "dinov2": float(w_dinov2),
        }
        self._load()

    def _load(self):
        self._try_load_fashion_clip()
        self._try_load_dinov2()
        if not self._models:
            raise RuntimeError(
                "[GarmentMetric] No garment encoder available. "
                "Install transformers and ensure model downloads are allowed."
            )
        loaded = ", ".join(self._models.keys())
        print(f"[GarmentMetric] Active garment backends: {loaded}")

    def _try_load_fashion_clip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device).eval()
            processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            self._models["fashion_clip"] = model
            self._processors["fashion_clip"] = processor
            print("[GarmentMetric] FashionCLIP loaded.")
        except Exception as e:
            print(f"[GarmentMetric] FashionCLIP unavailable ({e}).")

    def _try_load_dinov2(self):
        try:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
            self._models["dinov2"] = model
            self._processors["dinov2"] = processor
            print("[GarmentMetric] DINOv2 (facebook/dinov2-base) loaded.")
        except Exception as e:
            print(f"[GarmentMetric] DINOv2 unavailable ({e}).")

    @torch.no_grad()
    def __call__(self, cloth_imgs: torch.Tensor) -> Dict[str, np.ndarray]:
        outputs: Dict[str, np.ndarray] = {}
        if "fashion_clip" in self._models:
            outputs["fashion_clip"] = self._fashion_clip_embed(cloth_imgs)
        if "dinov2" in self._models:
            outputs["dinov2"] = self._dinov2_embed(cloth_imgs)
        return outputs

    def _to_pils(self, imgs: torch.Tensor):
        return [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in imgs]

    def _fashion_clip_embed(self, imgs: torch.Tensor) -> np.ndarray:
        processor = self._processors["fashion_clip"]
        model = self._models["fashion_clip"]
        pils = self._to_pils(imgs)
        inputs = processor(images=pils, return_tensors="pt").to(self.device)
        emb = model.get_image_features(**inputs)
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        elif hasattr(emb, "last_hidden_state"):
            emb = emb.last_hidden_state[:, 0]
        return emb.float().cpu().numpy()

    def _dinov2_embed(self, imgs: torch.Tensor) -> np.ndarray:
        processor = self._processors["dinov2"]
        model = self._models["dinov2"]
        pils = self._to_pils(imgs)
        inputs = processor(images=pils, return_tensors="pt").to(self.device)
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0]
        else:
            raise RuntimeError("DINOv2 output missing both pooler_output and last_hidden_state")
        return emb.float().cpu().numpy()


class GarmentTextureMetrics:
    def __init__(
        self,
        device: str = "cpu",
        eps: float = 1e-6,
        n_components: int = 128,
        w_fashion_clip: float = 0.5,
        w_dinov2: float = 0.5,
    ):
        self._encoder = _GarmentEncoder(
            device=device,
            w_fashion_clip=w_fashion_clip,
            w_dinov2=w_dinov2,
        )
        self.eps = eps
        self.n_components = n_components
        self._embeddings: Dict[str, List[np.ndarray]] = {k: [] for k in self._encoder._models.keys()}

    def update(self, cloth_imgs: torch.Tensor):
        enc = self._encoder(cloth_imgs)
        for backend, embs in enc.items():
            for emb in embs:
                self._embeddings[backend].append(emb)

    def _compute_backend_stats(self, embeddings: List[np.ndarray]) -> Dict[str, float]:
        if len(embeddings) < 2:
            return {
                "garment_diversity_logdet": float("nan"),
                "garment_diversity_logdet_normalized": float("nan"),
                "garment_variance_total": float("nan"),
                "garment_embed_dim": float("nan"),
                "garment_effective_rank": float("nan"),
            }

        e = np.stack(embeddings, axis=0)
        n, d = e.shape
        mu = e.mean(axis=0, keepdims=True)
        ec = e - mu

        k_max = min(n - 1, d, self.n_components)
        _, s, _ = np.linalg.svd(ec, full_matrices=False)
        s = s[:k_max]
        eigvals = (s ** 2) / max(n - 1, 1)
        total_var = float(eigvals.sum())

        if total_var > 0:
            cvar = np.cumsum(eigvals) / total_var
            effective_rank = int(np.searchsorted(cvar, 0.95)) + 1
        else:
            effective_rank = 0

        if effective_rank == 0:
            return {
                "garment_diversity_logdet": float("-inf"),
                "garment_diversity_logdet_normalized": float("-inf"),
                "garment_variance_total": total_var,
                "garment_embed_dim": float(k_max),
                "garment_effective_rank": 0.0,
            }

        sig = eigvals[:effective_rank] + self.eps
        log_det = float(np.sum(np.log(sig)))
        log_det_norm = log_det / effective_rank

        return {
            "garment_diversity_logdet": log_det,
            "garment_diversity_logdet_normalized": log_det_norm,
            "garment_variance_total": total_var,
            "garment_embed_dim": float(effective_rank),
            "garment_effective_rank": float(effective_rank),
        }

    def compute(self) -> Dict[str, float]:
        per_backend: Dict[str, Dict[str, float]] = {}
        for backend, embs in self._embeddings.items():
            per_backend[backend] = self._compute_backend_stats(embs)

        keys = [
            "garment_diversity_logdet",
            "garment_diversity_logdet_normalized",
            "garment_variance_total",
            "garment_embed_dim",
            "garment_effective_rank",
        ]

        weighted: Dict[str, float] = {}
        for key in keys:
            vals = []
            ws = []
            for backend, stats in per_backend.items():
                v = stats.get(key, float("nan"))
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    continue
                vals.append(float(v))
                ws.append(float(self._encoder.weights.get(backend, 0.0)))
            if not vals:
                weighted[key] = float("nan")
            else:
                wsum = sum(ws)
                if wsum <= 0:
                    ws = [1.0 for _ in ws]
                    wsum = float(len(ws))
                weighted[key] = float(sum(v * w for v, w in zip(vals, ws)) / wsum)

        out = dict(weighted)
        out["garment_diversity_logdet_raw"] = weighted.get("garment_diversity_logdet", float("nan"))

        def _norm_score_from_logdet(v: float) -> float:
            # Higher normalized log-det indicates larger spread/diversity.
            if np.isnan(v) or np.isinf(v):
                return float("nan")
            return float(1.0 / (1.0 + np.exp(-v)))

        for backend, stats in per_backend.items():
            out[f"garment_{backend}_diversity_logdet_raw"] = stats["garment_diversity_logdet"]
            out[f"garment_{backend}_diversity_logdet"] = stats["garment_diversity_logdet"]
            out[f"garment_{backend}_diversity_logdet_normalized"] = stats[
                "garment_diversity_logdet_normalized"
            ]
            out[f"garment_{backend}_score_0_1"] = _norm_score_from_logdet(
                stats["garment_diversity_logdet_normalized"]
            )
            out[f"garment_{backend}_variance_total"] = stats["garment_variance_total"]
            out[f"garment_{backend}_effective_rank"] = stats["garment_effective_rank"]
            out[f"garment_weight_{backend}"] = float(self._encoder.weights.get(backend, 0.0))

        out["garment_ensemble_score_0_1"] = _norm_score_from_logdet(
            out["garment_diversity_logdet_normalized"]
        )
        out["garment_backends_active"] = float(len(self._encoder._models))
        return out

    def reset(self):
        for key in self._embeddings.keys():
            self._embeddings[key].clear()
