"""
metrics/m1_pose.py
==================
Metric 1 — Pose Diversity & Pose Articulation Complexity
----------------------------------------------------------

1A. Pose Diversity (D_pose)
    log det(Cov(v_i) + ε·I)
    where v_i ∈ R^34 is the flattened, normalised pose vector for image i.

1B. Pose Articulation Complexity (C_artic)
    Sum of joint-angle variances across a predefined set of limb triplets.

1C. Pose Uncertainty (U_pose)
    Mean missing-keypoint ratio per image.
    Images with fewer reliably detected keypoints contribute slightly to
    combined diversity/complexity.

Pretrained model
-----------------
OpenMMLab MMPose ViTPose + person detector (primary),
HuggingFace ViTPose fallback. COCO-17 keypoints.

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns
--------
dict with keys:
    "pose_diversity"           float  (dataset-level, call compute() at end)
    "pose_artic_complexity"    float  (dataset-level)
    "per_image_artic"          list[float]   (per batch / accumulated)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# COCO 17-joint skeleton
# ─────────────────────────────────────────────────────────────────────────────
# Joint index → name
COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
J2I = {n: i for i, n in enumerate(COCO_JOINTS)}

# Limb triplets (a, b, c): b is the vertex joint (angle measured at b)
LIMB_TRIPLETS = [
    ("left_shoulder",  "left_elbow",   "left_wrist"),    # left elbow
    ("right_shoulder", "right_elbow",  "right_wrist"),   # right elbow
    ("left_hip",       "left_knee",    "left_ankle"),    # left knee
    ("right_hip",      "right_knee",   "right_ankle"),   # right knee
    ("left_elbow",     "left_shoulder","left_hip"),      # left shoulder tilt
    ("right_elbow",    "right_shoulder","right_hip"),    # right shoulder tilt
    ("left_shoulder",  "left_hip",     "left_knee"),     # left torso-hip
    ("right_shoulder", "right_hip",    "right_knee"),    # right torso-hip
]
TRIPLET_IDX = [(J2I[a], J2I[b], J2I[c]) for a, b, c in LIMB_TRIPLETS]

# SPIN model neck proxy: midpoint of left/right shoulder
IDX_L_SHOULDER = J2I["left_shoulder"]
IDX_R_SHOULDER = J2I["right_shoulder"]
IDX_L_HIP      = J2I["left_hip"]
IDX_R_HIP      = J2I["right_hip"]

# End-effectors used for position-based pose components
END_EFFECTOR_IDXS = [
    J2I["left_wrist"],
    J2I["right_wrist"],
    J2I["left_ankle"],
    J2I["right_ankle"],
]


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained keypoint extractor
# ─────────────────────────────────────────────────────────────────────────────

class _KeypointExtractor:
    """COCO-17 keypoint extractor using MMPose ViTPose (fallback: HF ViTPose)."""

    INPUT_SIZE = (256, 192)   # H×W for most top-down pose models

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend = "none"
        self._mmpose_inferencer = None
        self._model = None
        self._processor = None
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        try:
            from mmpose.apis import MMPoseInferencer

            self._mmpose_inferencer = MMPoseInferencer(
                pose2d="vitpose-b",
                det_model="rtmdet_m",
                det_cat_ids=[0],
                device=self.device,
            )
            self._backend = "mmpose_vitpose"
            print("[PoseMetric] Using OpenMMLab MMPose ViTPose + detector.")
            return
        except Exception as e:
            print(f"[PoseMetric] MMPose ViTPose unavailable ({e}); trying HF ViTPose fallback.")

        try:
            from transformers import AutoProcessor, VitPoseForPoseEstimation

            self._processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
            self._model = VitPoseForPoseEstimation.from_pretrained(
                "usyd-community/vitpose-base-simple",
                use_safetensors=True,
            ).to(self.device).eval()
            self._backend = "vitpose_hf_fallback"
            print("[PoseMetric] Using HF ViTPose fallback for keypoint extraction.")
            return
        except Exception as e2:
            raise RuntimeError(
                "[PoseMetric] No ViTPose backend available. "
                "Install MMPose(+mmdet) or transformers ViTPose dependencies."
            ) from e2

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns : (B, 17, 2) numpy array of (x, y) pixel coordinates
        """
        if imgs.ndim != 4:
            raise RuntimeError(f"[PoseMetric] Expected 4D tensor, got {tuple(imgs.shape)}")
        if imgs.shape[1] != 3 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
        if imgs.shape[1] != 3:
            raise RuntimeError(f"[PoseMetric] Expected C=3, got {tuple(imgs.shape)}")

        if self._backend == "mmpose_vitpose":
            return self._extract_with_mmpose(imgs)
        return self._extract_with_hf(imgs)

    def _extract_with_mmpose(self, imgs: torch.Tensor) -> np.ndarray:
        b = imgs.shape[0]
        all_kps = np.zeros((b, 17, 2), dtype=np.float32)
        for i in range(b):
            img = imgs[i].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
            try:
                res = next(self._mmpose_inferencer(img, return_vis=False))
                preds = res.get("predictions", [])
                persons = preds[0] if preds and len(preds) > 0 else []
                if not persons:
                    continue

                def _person_score(p):
                    sc = p.get("keypoint_scores", None)
                    if sc is None:
                        return 0.0
                    arr = np.asarray(sc, dtype=np.float32).reshape(-1)
                    return float(np.mean(arr)) if arr.size else 0.0

                best = max(persons, key=_person_score)
                kps = np.asarray(best.get("keypoints", []), dtype=np.float32)
                if kps.ndim == 2 and kps.shape[1] >= 2:
                    n = min(17, kps.shape[0])
                    all_kps[i, :n, :] = kps[:n, :2]
            except Exception:
                continue
        return all_kps

    def _extract_with_hf(self, imgs: torch.Tensor) -> np.ndarray:
        b, _c, h, w = imgs.shape
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in imgs]
        boxes = [np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32) for _ in range(b)]

        inputs = self._processor(images=pils, boxes=boxes, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        pose_results = self._processor.post_process_pose_estimation(
            outputs, boxes=boxes, threshold=0.1
        )

        all_kps = np.zeros((b, 17, 2), dtype=np.float32)
        for i in range(b):
            if i >= len(pose_results) or not pose_results[i]:
                continue
            person = pose_results[i][0]
            keypoints = person.get("keypoints", [])
            labels = person.get("labels", [])
            for kp, label in zip(keypoints, labels):
                idx = int(label.item() if torch.is_tensor(label) else label)
                if 0 <= idx < 17:
                    x = float(kp[0].item() if torch.is_tensor(kp[0]) else kp[0])
                    y = float(kp[1].item() if torch.is_tensor(kp[1]) else kp[1])
                    all_kps[i, idx, 0] = x
                    all_kps[i, idx, 1] = y
        return all_kps

# ─────────────────────────────────────────────────────────────────────────────
# Normalise pose
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_pose(kps: np.ndarray) -> np.ndarray:
    """
    kps : (B, 17, 2)
    Returns normalised (B, 17, 2) and keypoint-validity mask (B, 17) bool.
    Translation: subtract hip centre if available, else valid-keypoint centroid.
    Scale:       torso length (neck→hip) if available, else valid-keypoint bbox diagonal.
    """
    B = kps.shape[0]
    out = np.zeros_like(kps, dtype=np.float32)
    valid_kp = np.isfinite(kps).all(axis=2) & (np.linalg.norm(kps, axis=2) > 1e-6)

    for i in range(B):
        p = kps[i]
        vk = valid_kp[i]
        if vk.sum() == 0:
            continue

        has_hips = bool(vk[IDX_L_HIP] and vk[IDX_R_HIP])
        has_shoulders = bool(vk[IDX_L_SHOULDER] and vk[IDX_R_SHOULDER])

        if has_hips:
            center = (p[IDX_L_HIP] + p[IDX_R_HIP]) / 2.0
        else:
            center = p[vk].mean(axis=0)

        if has_hips and has_shoulders:
            neck = (p[IDX_L_SHOULDER] + p[IDX_R_SHOULDER]) / 2.0
            scale = float(np.linalg.norm(neck - center))
        else:
            pts = p[vk]
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            scale = float(np.linalg.norm(mx - mn))

        if scale < 1e-6:
            continue

        out[i, vk] = (p[vk] - center) / scale

    return out, valid_kp


# ─────────────────────────────────────────────────────────────────────────────
# Joint-angle computation
# ─────────────────────────────────────────────────────────────────────────────

def _joint_angle(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    """Angle at joint b (vertex), in radians."""
    va = pa - pb
    vc = pc - pb
    n_a = np.linalg.norm(va)
    n_c = np.linalg.norm(vc)
    if n_a < 1e-8 or n_c < 1e-8:
        return float("nan")
    cos_t = np.dot(va, vc) / (n_a * n_c)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.arccos(cos_t))


# ─────────────────────────────────────────────────────────────────────────────
# PoseMetrics class
# ─────────────────────────────────────────────────────────────────────────────

class PoseMetrics:
    """
    Accumulates per-image pose features, then computes:
      - D_pose  = log det(Cov(v_i) + ε·I)
      - C_artic = Σ_limbs Var(θ_limb)
    """

    def __init__(
        self,
        device: str = "cpu",
        eps: float = 1e-6,
        diversity_w_angle: float = 0.6,
        diversity_w_end_effector: float = 0.3,
        diversity_w_uncertainty: float = 0.1,
        complexity_w_angle: float = 0.6,
        complexity_w_end_effector: float = 0.3,
        complexity_w_uncertainty: float = 0.1,
    ):
        self.extractor = _KeypointExtractor(device=device)
        self.eps = eps
        self.diversity_w_angle = float(diversity_w_angle)
        self.diversity_w_end_effector = float(diversity_w_end_effector)
        self.diversity_w_uncertainty = float(diversity_w_uncertainty)
        self.complexity_w_angle = float(complexity_w_angle)
        self.complexity_w_end_effector = float(complexity_w_end_effector)
        self.complexity_w_uncertainty = float(complexity_w_uncertainty)
        self._pose_vecs: List[np.ndarray]     = []   # (34,) vectors
        self._all_angles: Dict[int, List[float]] = {t: [] for t in range(len(TRIPLET_IDX))}
        self._angle_vecs: List[np.ndarray] = []      # (num_triplets,) per image, NaN for missing
        self._end_eff_vecs: List[np.ndarray] = []    # (len(END_EFFECTOR_IDXS)*2,) per image
        self._per_image_artic: List[float]    = []
        self._per_image_end_eff_complexity: List[float] = []
        self._per_image_uncertainty: List[float] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """
        person_imgs : (B, 3, H, W)  float32  [0,1]
        Call once per batch.
        """
        kps_raw  = self.extractor(person_imgs)         # (B,17,2) numpy
        kps_norm, valid_kp = _normalise_pose(kps_raw)  # (B,17,2), (B,17)

        for i in range(kps_raw.shape[0]):
            pn = kps_norm[i]   # (17, 2)
            vk = valid_kp[i]   # (17,)

            # Uncertainty score in [0, 1]: higher means less reliable detection.
            valid_ratio = float(np.mean(vk.astype(np.float32)))
            uncertainty = 1.0 - valid_ratio
            self._per_image_uncertainty.append(float(np.clip(uncertainty, 0.0, 1.0)))

            # ── Pose vector ─────────────────────────────────────────────
            # Keep fixed-size representation; invalid keypoints remain zeros.
            if int(vk.sum()) > 0:
                self._pose_vecs.append(pn.flatten())   # (34,)

            # ── Joint angles ─────────────────────────────────────────────
            img_angles = []
            angle_vec = np.full((len(TRIPLET_IDX),), np.nan, dtype=np.float32)
            for t_idx, (ia, ib, ic) in enumerate(TRIPLET_IDX):
                if not (vk[ia] and vk[ib] and vk[ic]):
                    continue
                ang = _joint_angle(pn[ia], pn[ib], pn[ic])
                if not math.isnan(ang):
                    bend = float(np.abs(ang - math.pi))
                    self._all_angles[t_idx].append(bend)
                    img_angles.append(bend)
                    angle_vec[t_idx] = bend

            self._angle_vecs.append(angle_vec)

            # ── End-effector positions ───────────────────────────────────
            end_pts = []
            for idx in END_EFFECTOR_IDXS:
                if vk[idx]:
                    end_pts.append(pn[idx])
                else:
                    end_pts.append(np.array([0.0, 0.0], dtype=np.float32))
            end_arr = np.stack(end_pts, axis=0).astype(np.float32)  # (4,2)
            self._end_eff_vecs.append(end_arr.reshape(-1))

            valid_end_pts = [pn[idx] for idx in END_EFFECTOR_IDXS if vk[idx]]
            if len(valid_end_pts) >= 2:
                ve = np.stack(valid_end_pts, axis=0)
                # Position-based articulation: spread among end-effectors in an image.
                self._per_image_end_eff_complexity.append(float(np.sqrt(np.var(ve[:, 0]) + np.var(ve[:, 1]))))
            else:
                self._per_image_end_eff_complexity.append(float("nan"))

            # Per-image articulation = std of all valid angles
            if img_angles:
                self._per_image_artic.append(float(np.std(img_angles)))
            else:
                self._per_image_artic.append(float("nan"))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        """
          Returns dict with formulas:

             1) Pose Diversity
                 D_pose = log det( Cov(v) + eps * I )
                 where v in R^34 is flattened normalized keypoints.

             2) Pose Articulation Complexity
                 C_artic = sum_limb Var(theta_limb)
                 where theta_limb is the joint angle at each predefined triplet.

             3) Mean per-image articulation
                 mean_i std_j(theta_{i,j}) over valid angles in each image.
        """
        if len(self._pose_vecs) < 2:
            return {
                "pose_diversity": float("nan"),
                "pose_diversity_angle": float("nan"),
                "pose_diversity_end_effector": float("nan"),
                "pose_diversity_uncertainty": float("nan"),
                "pose_diversity_weight_angle": self.diversity_w_angle,
                "pose_diversity_weight_end_effector": self.diversity_w_end_effector,
                "pose_diversity_weight_uncertainty": self.diversity_w_uncertainty,
                "pose_artic_complexity": float("nan"),
                "pose_artic_complexity_angle": float("nan"),
                "pose_artic_complexity_end_effector": float("nan"),
                "pose_artic_complexity_uncertainty": float("nan"),
                "pose_artic_weight_angle": self.complexity_w_angle,
                "pose_artic_weight_end_effector": self.complexity_w_end_effector,
                "pose_artic_weight_uncertainty": self.complexity_w_uncertainty,
                "pose_uncertainty_mean": float("nan"),
                "pose_artic_mean_per_image": float("nan"),
            }

        def _weighted_average(parts: List[tuple[float, float]]) -> float:
            vals = [(v, w) for v, w in parts if not math.isnan(v) and w > 0.0]
            if not vals:
                return float("nan")
            wsum = float(sum(w for _, w in vals))
            if wsum <= 0.0:
                return float("nan")
            return float(sum(v * w for v, w in vals) / wsum)

        # 1A — Diversity (existing pose-vector metric)
        V   = np.stack(self._pose_vecs, axis=0).astype(np.float64)    # (N, D)
        D   = V.shape[1]
        mu  = V.mean(axis=0, keepdims=True)
        Vc  = V - mu
        cov = (Vc.T @ Vc) / max(len(V) - 1, 1)    # (D, D)
        reg = cov + self.eps * np.eye(D)
        sign, log_det = np.linalg.slogdet(reg)
        d_pose = float(log_det) if sign > 0 else float("nan")
        d_pose_norm = (d_pose / D) if (not math.isnan(d_pose) and D > 0) else float("nan")

        # 1A-angle — Diversity from joint-angle vectors
        A = np.stack(self._angle_vecs, axis=0).astype(np.float64)  # (N, T)
        for j in range(A.shape[1]):
            col = A[:, j]
            mask = np.isfinite(col)
            if np.any(mask):
                fill = float(np.mean(col[mask]))
                col[~mask] = fill
            else:
                col[:] = 0.0
            A[:, j] = col

        Tdim = A.shape[1]
        Ac = A - A.mean(axis=0, keepdims=True)
        cov_a = (Ac.T @ Ac) / max(len(A) - 1, 1)
        reg_a = cov_a + self.eps * np.eye(Tdim)
        sign_a, log_det_a = np.linalg.slogdet(reg_a)
        d_pose_angle = float(log_det_a / Tdim) if (sign_a > 0 and Tdim > 0) else float("nan")

        # 1A-eff — Diversity from end-effector positions
        E = np.stack(self._end_eff_vecs, axis=0).astype(np.float64)  # (N, 8)
        Edim = E.shape[1]
        Ec = E - E.mean(axis=0, keepdims=True)
        cov_e = (Ec.T @ Ec) / max(len(E) - 1, 1)
        reg_e = cov_e + self.eps * np.eye(Edim)
        sign_e, log_det_e = np.linalg.slogdet(reg_e)
        d_pose_end_eff = float(log_det_e / Edim) if (sign_e > 0 and Edim > 0) else float("nan")

        unc_arr = np.array(self._per_image_uncertainty, dtype=np.float64)
        d_pose_uncertainty = float(np.mean(unc_arr)) if unc_arr.size > 0 else float("nan")

        d_pose_combined = _weighted_average([
            (d_pose_angle, self.diversity_w_angle),
            (d_pose_end_eff, self.diversity_w_end_effector),
            (d_pose_uncertainty, self.diversity_w_uncertainty),
        ])

        # 1B-angle — Complexity from joint angles
        c_artic_angle = 0.0
        valid_limbs = 0
        for t_idx in range(len(TRIPLET_IDX)):
            angles = self._all_angles[t_idx]
            if len(angles) > 0:
                # Angles are already stored as absolute bend magnitudes |theta - pi|.
                c_artic_angle += float(np.mean(np.array(angles)))
                valid_limbs += 1
        
        if valid_limbs > 0:
            c_artic_angle = c_artic_angle / valid_limbs

        # 1B-eff — Complexity from end-effector spatial spread
        eff_per_image = [v for v in self._per_image_end_eff_complexity if not math.isnan(v)]
        c_artic_end_eff = float(np.mean(eff_per_image)) if eff_per_image else float("nan")

        c_artic_uncertainty = d_pose_uncertainty

        c_artic_combined = _weighted_average([
            (c_artic_angle, self.complexity_w_angle),
            (c_artic_end_eff, self.complexity_w_end_effector),
            (c_artic_uncertainty, self.complexity_w_uncertainty),
        ])

        artic_per_image = [v for v in self._per_image_artic if not math.isnan(v)]

        return {
            # Backward-compatible main keys now represent weighted combined variants.
            "pose_diversity":            d_pose_combined,
            "pose_diversity_logdet_raw": d_pose,
            "pose_diversity_logdet_normalized": d_pose_norm,
            "pose_diversity_angle":      d_pose_angle,
            "pose_diversity_end_effector": d_pose_end_eff,
            "pose_diversity_uncertainty": d_pose_uncertainty,
            "pose_diversity_weight_angle": self.diversity_w_angle,
            "pose_diversity_weight_end_effector": self.diversity_w_end_effector,
            "pose_diversity_weight_uncertainty": self.diversity_w_uncertainty,
            "pose_artic_complexity":     c_artic_combined,
            "pose_artic_complexity_angle": c_artic_angle,
            "pose_artic_complexity_end_effector": c_artic_end_eff,
            "pose_artic_complexity_uncertainty": c_artic_uncertainty,
            "pose_artic_weight_angle": self.complexity_w_angle,
            "pose_artic_weight_end_effector": self.complexity_w_end_effector,
            "pose_artic_weight_uncertainty": self.complexity_w_uncertainty,
            "pose_uncertainty_mean": d_pose_uncertainty,
            "pose_artic_mean_per_image": float(np.mean(artic_per_image)) if artic_per_image else float("nan"),
        }

    def reset(self):
        self._pose_vecs.clear()
        for k in self._all_angles:
            self._all_angles[k].clear()
        self._angle_vecs.clear()
        self._end_eff_vecs.clear()
        self._per_image_artic.clear()
        self._per_image_end_eff_complexity.clear()
        self._per_image_uncertainty.clear()
