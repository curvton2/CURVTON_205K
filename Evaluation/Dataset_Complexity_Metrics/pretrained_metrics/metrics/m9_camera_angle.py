"""
metrics/m9_camera_angle.py
===========================
Metric 9 — Camera Angle Diversity
----------------------------------
Measures the diversity of camera viewpoints/angles across a dataset.
Important for evaluating how well a model handles different viewing angles.

Camera Angle Components
------------------------
  1. Azimuth (Horizontal angle):
     - Frontal (0°), Side (±90°), Back (180°)
     - Left vs Right distinction

  2. Elevation (Vertical angle):
     - Eye-level, High-angle (bird's eye), Low-angle (worm's eye)

  3. Body Orientation:
     - Person facing camera vs turned away
     - 3/4 view, profile view

Formula
--------
    Diversity metrics:
    - angle_mean, angle_std: Statistics of azimuth distribution
    - elevation_mean, elevation_std: Statistics of elevation
    - angle_entropy: Shannon entropy of binned angle distribution
    - camera_diversity_score: Combined diversity metric

Pretrained Models
------------------
Primary:  HMR2.0 (4DHumans) - Provides camera parameters and body rotation
Fallback: ViTPose - Body orientation from shoulder/hip keypoints
Fallback: DINOv2 - Viewpoint-aware features for clustering

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (via compute())
------------------------
dict with:
    azimuth_mean            : Mean horizontal angle (degrees)
    azimuth_std             : Std of horizontal angles
    elevation_mean          : Mean vertical angle (degrees)
    elevation_std           : Std of vertical angles
    azimuth_entropy         : Entropy of angle distribution (higher = more diverse)
    camera_diversity_score  : Combined diversity score
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pretrained_metrics.cache_setup import configure_model_caches, ensure_hmr2_smpl_model


# ─────────────────────────────────────────────────────────────────────────────
# Camera Angle Backend
# ─────────────────────────────────────────────────────────────────────────────

class _CameraAngleBackend:
    """
    Extracts camera/viewpoint angles from person images.

    Backend priority:
      1. HMR2.0 (4DHumans) - Full camera + body rotation parameters
      2. ViTPose - Body orientation from pose keypoints
      3. DINOv2 - Viewpoint-aware feature clustering
    """

    # ── HMR2.0 body rotation indices ──────────────────────────────────────────
    # Global orientation is the first 3 values of body_pose (axis-angle)

    # ── ViTPose keypoint indices (COCO format) ────────────────────────────────
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_EAR = 3
    RIGHT_EAR = 4

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend: Optional[str] = None

        # Model references
        self._hmr_model = None
        self._hmr_cfg = None
        self._vitpose_model = None
        self._vitpose_processor = None
        self._dino_model = None
        self._dino_processor = None

        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        """Load camera angle estimation models with fallback chain."""

        # ── Try HMR2.0 first ──────────────────────────────────────────────────
        try:
            cache_info = configure_model_caches(set_home_for_hmr2=True)
            smpl_info = ensure_hmr2_smpl_model(cache_info["base_path"])
            from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
            import torch.serialization as _ts
            from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig

            _ts.add_safe_globals([_DictConfig, _ListConfig])
            download_models()
            self._hmr_model, self._hmr_cfg = load_hmr2(DEFAULT_CHECKPOINT)
            self._hmr_model = self._hmr_model.to(self.device).eval()
            self._backend = "hmr2"
            print(f"[CameraAngle] Using HMR2.0 for camera estimation (cache: {cache_info['fourdhumans_cache']}).")
            print(f"[CameraAngle] SMPL path status: {smpl_info['status']} -> {smpl_info['dst']}")
            return
        except Exception as e:
            print(f"[CameraAngle] HMR2.0 not available: {e}")

        # ── Try ViTPose ───────────────────────────────────────────────────────
        try:
            from transformers import AutoProcessor, VitPoseForPoseEstimation
            self._vitpose_processor = AutoProcessor.from_pretrained(
                "usyd-community/vitpose-base-simple"
            )
            self._vitpose_model = VitPoseForPoseEstimation.from_pretrained(
                "usyd-community/vitpose-base-simple",
                use_safetensors=True,
            ).to(self.device).eval()
            self._backend = "vitpose"
            print("[CameraAngle] Using ViTPose for body orientation estimation.")
            return
        except Exception as e:
            print(f"[CameraAngle] ViTPose not available: {e}")

        # Try alternative pose model
        try:
            import torchvision
            weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            self._vitpose_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                weights=weights
            ).to(self.device).eval()
            self._backend = "keypointrcnn"
            print("[CameraAngle] Using KeypointRCNN for body orientation estimation.")
            return
        except Exception as e:
            print(f"[CameraAngle] Pose models not available: {e}")

        # ── Try DINOv2 ────────────────────────────────────────────────────────
        try:
            from transformers import AutoModel, AutoImageProcessor
            self._dino_model = AutoModel.from_pretrained(
                "facebook/dinov2-base"
            ).to(self.device).eval()
            self._dino_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base"
            )
            self._backend = "dino"
            print("[CameraAngle] Using DINOv2 for viewpoint feature extraction.")
            return
        except Exception as e:
            print(f"[CameraAngle] DINOv2 not available: {e}")
        raise RuntimeError(
            "[CameraAngle] No valid camera-angle backend available. "
            "Install HMR2.0, ViTPose/KeypointRCNN, or DINOv2."
        )

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def estimate_angles(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate camera angles for a batch of images.

        Args:
            imgs: (B, 3, H, W) float32 [0, 1]

        Returns:
            dict with:
                azimuth: (B,) horizontal angle in degrees [0, 360)
                elevation: (B,) vertical angle in degrees [-90, 90]
                confidence: (B,) confidence score [0, 1]
        """
        imgs = imgs.to(self.device)
        B, C, H, W = imgs.shape

        angles: Dict[str, torch.Tensor]
        if self._backend == "hmr2":
            angles = self._hmr2_angles(imgs, H, W)
        elif self._backend == "vitpose":
            angles = self._vitpose_angles(imgs, H, W)
        elif self._backend == "keypointrcnn":
            angles = self._keypointrcnn_angles(imgs, H, W)
        elif self._backend == "dino":
            angles = self._dino_angles(imgs, H, W)
        else:
            raise RuntimeError("[CameraAngle] No valid camera-angle backend available.")

        # Use canonical azimuth representation for all downstream metrics/EDA.
        if "azimuth" in angles:
            angles["azimuth"] = torch.remainder(angles["azimuth"], 360.0)
        return angles

    # --------------------------------------------------------------------- #
    def _hmr2_angles(self, imgs: torch.Tensor, H: int, W: int):
        """
        Extract camera angles from HMR2.0 predictions.
        HMR2.0 outputs global body orientation as axis-angle rotation.
        """
        B = imgs.shape[0]

        # Normalize for HMR2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        imgs_norm = (imgs - mean) / std

        # Resize to HMR2.0 expected size (256x256)
        imgs_resized = F.interpolate(imgs_norm, size=(256, 256), mode="bilinear", align_corners=False)

        try:
            output = self._hmr_model({"img": imgs_resized})

            # ── Extract global orientation ─────────────────────────────────
            # HMR2 returns pred_smpl_params["global_orient"] which can be:
            #   - (B, 1, 3, 3) rotation matrix (most common)
            #   - (B, 3, 3) rotation matrix
            #   - (B, 3) axis-angle
            global_orient = None
            rot_matrix = None  # Will be set if we get a rotation matrix

            if isinstance(output, dict):
                go = output.get("global_orient", None)
                if go is None and isinstance(output.get("pred_smpl_params"), dict):
                    pred_smpl = output["pred_smpl_params"]
                    go = pred_smpl.get("global_orient", None)

                if go is not None:
                    if go.ndim == 4 and go.shape[-2:] == (3, 3):
                        # (B, 1, 3, 3) or (B, K, 3, 3) rotation matrix
                        rot_matrix = go[:, 0]  # -> (B, 3, 3)
                    elif go.ndim == 3 and go.shape[-2:] == (3, 3):
                        # (B, 3, 3) rotation matrix
                        rot_matrix = go
                    elif go.ndim == 3 and go.shape[-1] == 3:
                        # (B, 1, 3) axis-angle -> squeeze
                        global_orient = go[:, 0, :3]
                    elif go.ndim == 2 and go.shape[1] >= 3:
                        # (B, 3) axis-angle
                        global_orient = go[:, :3]

            # ── Convert to azimuth / elevation ─────────────────────────────
            if rot_matrix is not None:
                # Extract Euler angles directly from rotation matrix
                azimuth, elevation = self._rotmat_to_euler(rot_matrix)
                confidence = torch.ones(B, device=self.device)
            elif global_orient is not None:
                # Convert axis-angle to euler angles
                azimuth, elevation = self._axis_angle_to_euler(global_orient)
                confidence = torch.ones(B, device=self.device)
            else:
                # Fallback to camera parameters
                pred_cam = output.get("pred_cam", None) if isinstance(output, dict) else None
                if pred_cam is not None:
                    azimuth = torch.zeros(B, device=self.device)
                    elevation = torch.zeros(B, device=self.device)
                    confidence = torch.ones(B, device=self.device) * 0.5
                else:
                    azimuth = torch.zeros(B, device=self.device)
                    elevation = torch.zeros(B, device=self.device)
                    confidence = torch.zeros(B, device=self.device)

        except Exception as e:
            raise RuntimeError("[CameraAngle] HMR2.0 inference failed.") from e

        return {
            "azimuth": azimuth.cpu(),
            "elevation": elevation.cpu(),
            "confidence": confidence.cpu(),
        }

    # --------------------------------------------------------------------- #
    def _axis_angle_to_euler(self, axis_angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert axis-angle rotation to azimuth (yaw) and elevation (pitch).

        Args:
            axis_angle: (B, 3) axis-angle rotation

        Returns:
            azimuth: (B,) in degrees
            elevation: (B,) in degrees
        """
        B = axis_angle.shape[0]

        # Compute rotation angle
        angle = torch.norm(axis_angle, dim=1, keepdim=True).clamp(min=1e-8)
        axis = axis_angle / angle

        # Convert to rotation matrix
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Rodrigues formula components
        K = torch.zeros(B, 3, 3, device=axis_angle.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(B, -1, -1)
        R = I + sin_a.unsqueeze(-1) * K + (1 - cos_a.unsqueeze(-1)) * torch.bmm(K, K)

        # Extract Euler angles (Y-X-Z convention)
        # Azimuth (yaw) = rotation around Y axis
        # Elevation (pitch) = rotation around X axis
        azimuth = torch.atan2(R[:, 0, 2], R[:, 2, 2]) * 180 / math.pi
        elevation = torch.asin(-R[:, 1, 2].clamp(-1, 1)) * 180 / math.pi

        # Keep 1-D tensors even when batch size is 1.
        return azimuth.reshape(-1), elevation.reshape(-1)

    # --------------------------------------------------------------------- #
    def _rotmat_to_euler(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract azimuth (yaw) and elevation (pitch) from rotation matrix.

        Args:
            R: (B, 3, 3) rotation matrix

        Returns:
            azimuth: (B,) in degrees [0, 360)
            elevation: (B,) in degrees [-90, 90]
        """
        # Extract Euler angles (Y-X-Z convention)
        # Azimuth (yaw) = rotation around Y axis
        # Elevation (pitch) = rotation around X axis
        azimuth = torch.atan2(R[:, 0, 2], R[:, 2, 2]) * 180 / math.pi
        elevation = torch.asin(-R[:, 1, 2].clamp(-1, 1)) * 180 / math.pi
        return azimuth.reshape(-1), elevation.reshape(-1)

    # --------------------------------------------------------------------- #
    def _vitpose_angles(self, imgs: torch.Tensor, H: int, W: int):
        """Estimate body orientation from ViTPose keypoints."""
        B = imgs.shape[0]

        try:
            import torchvision.transforms.functional as TF

            if self._vitpose_processor is None:
                raise RuntimeError("ViTPose processor is not initialized.")

            pils = [TF.to_pil_image(imgs[i].cpu().clamp(0, 1)) for i in range(B)]

            # ViTPose is top-down in transformers; use one full-image person box.
            # Processor expects COCO xywh boxes.
            boxes = [np.array([[0.0, 0.0, float(W), float(H)]], dtype=np.float32) for _ in range(B)]

            inputs = self._vitpose_processor(images=pils, boxes=boxes, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._vitpose_model(**inputs)
            pose_results = self._vitpose_processor.post_process_pose_estimation(
                outputs, boxes=boxes, threshold=0.3
            )

            azimuth_list = []
            elevation_list = []
            confidence_list = []

            for i in range(B):
                img_res = pose_results[i] if i < len(pose_results) else []
                if not img_res:
                    azimuth_list.append(0.0)
                    elevation_list.append(0.0)
                    confidence_list.append(0.0)
                    continue

                person_pose = img_res[0]
                kpts = torch.zeros(17, 3)
                for kp, score, label in zip(
                    person_pose.get("keypoints", []),
                    person_pose.get("scores", []),
                    person_pose.get("labels", []),
                ):
                    idx = int(label.item() if torch.is_tensor(label) else label)
                    if 0 <= idx < 17:
                        x = float(kp[0].item() if torch.is_tensor(kp[0]) else kp[0])
                        y = float(kp[1].item() if torch.is_tensor(kp[1]) else kp[1])
                        s = float(score.item() if torch.is_tensor(score) else score)
                        kpts[idx, 0] = x
                        kpts[idx, 1] = y
                        kpts[idx, 2] = s

                az, el, conf = self._single_keypoints_to_angle(kpts, H, W)
                azimuth_list.append(az)
                elevation_list.append(el)
                confidence_list.append(conf)

            azimuth = torch.tensor(azimuth_list)
            elevation = torch.tensor(elevation_list)
            confidence = torch.tensor(confidence_list)

        except Exception as e:
            raise RuntimeError("[CameraAngle] ViTPose inference failed.") from e

        return {
            "azimuth": azimuth.cpu(),
            "elevation": elevation.cpu(),
            "confidence": confidence.cpu(),
        }

    # --------------------------------------------------------------------- #
    def _keypointrcnn_angles(self, imgs: torch.Tensor, H: int, W: int):
        """Estimate body orientation from KeypointRCNN."""
        B = imgs.shape[0]

        azimuth_list = []
        elevation_list = []
        confidence_list = []

        # KeypointRCNN expects list of images
        imgs_list = [imgs[i] for i in range(B)]

        try:
            outputs = self._vitpose_model(imgs_list)

            for i, out in enumerate(outputs):
                if len(out["keypoints"]) > 0:
                    # Take the most confident detection
                    scores = out["scores"]
                    best_idx = scores.argmax()
                    kpts = out["keypoints"][best_idx]  # (17, 3) x, y, conf

                    az, el, conf = self._single_keypoints_to_angle(kpts, H, W)
                    azimuth_list.append(az)
                    elevation_list.append(el)
                    confidence_list.append(conf)
                else:
                    azimuth_list.append(0.0)
                    elevation_list.append(0.0)
                    confidence_list.append(0.0)

        except Exception as e:
            raise RuntimeError("[CameraAngle] KeypointRCNN inference failed.") from e

        return {
            "azimuth": torch.tensor(azimuth_list),
            "elevation": torch.tensor(elevation_list),
            "confidence": torch.tensor(confidence_list),
        }

    # --------------------------------------------------------------------- #
    def _single_keypoints_to_angle(
        self, kpts: torch.Tensor, H: int, W: int
    ) -> Tuple[float, float, float]:
        """
        Compute azimuth and elevation from a single person's keypoints.

        Body orientation estimation:
        - Azimuth: from shoulder width ratio (foreshortening indicates rotation)
        - Azimuth: from nose-ear visibility (frontal vs side)
        - Elevation: from vertical position of shoulders relative to hips
        """
        # Extract keypoints (x, y, confidence)
        nose = kpts[self.NOSE]
        l_shoulder = kpts[self.LEFT_SHOULDER]
        r_shoulder = kpts[self.RIGHT_SHOULDER]
        l_hip = kpts[self.LEFT_HIP]
        r_hip = kpts[self.RIGHT_HIP]
        l_ear = kpts[self.LEFT_EAR]
        r_ear = kpts[self.RIGHT_EAR]

        confidence = min(
            l_shoulder[2].item(), r_shoulder[2].item(),
            l_hip[2].item(), r_hip[2].item()
        )

        if confidence < 0.3:
            return 0.0, 0.0, 0.0

        # ── Azimuth estimation ────────────────────────────────────────────────
        # Method 1: Shoulder midpoint offset from nose
        shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
        nose_x = nose[0]
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0])

        if shoulder_width > 10:
            # Lateral offset indicates rotation
            offset = (nose_x - shoulder_mid_x) / (shoulder_width + 1e-6)
            azimuth_from_offset = offset * 45  # Scale to reasonable range

            # Method 2: Shoulder width foreshortening
            # Typical frontal shoulder width is ~0.3-0.4 of image width
            # Side view shows much less
            expected_width = W * 0.35
            width_ratio = shoulder_width / expected_width
            width_ratio = min(max(width_ratio, 0.3), 1.5)

            # Foreshortening indicates rotation
            if width_ratio < 0.7:
                # Significant foreshortening = side view
                azimuth_from_width = (1 - width_ratio) * 90
            else:
                azimuth_from_width = 0

            # Method 3: Ear visibility
            ear_azimuth = 0.0
            if l_ear[2] > 0.5 and r_ear[2] < 0.3:
                ear_azimuth = -45  # Facing left
            elif r_ear[2] > 0.5 and l_ear[2] < 0.3:
                ear_azimuth = 45   # Facing right

            # Combine methods
            azimuth = azimuth_from_offset * 0.4 + azimuth_from_width * 0.3 + ear_azimuth * 0.3

            # Determine sign from which shoulder is closer (lower y = higher in image = closer)
            if l_shoulder[0] > r_shoulder[0]:  # Right shoulder on left side of image = facing right
                if azimuth_from_width > 20:
                    azimuth = abs(azimuth)
            else:
                if azimuth_from_width > 20:
                    azimuth = -abs(azimuth)
        else:
            azimuth = 0.0

        # ── Elevation estimation ──────────────────────────────────────────────
        # Compare shoulder-hip vertical distance to expected
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2
        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
        torso_height = hip_mid_y - shoulder_mid_y

        if torso_height > 10:
            # Image center y
            center_y = H / 2
            # If person is above center = low angle shot, below center = high angle
            person_center_y = (shoulder_mid_y + hip_mid_y) / 2
            vertical_offset = (center_y - person_center_y) / H

            # Scale to elevation angle
            elevation = vertical_offset * 30  # ±30 degrees range
        else:
            elevation = 0.0

        return float(azimuth), float(elevation), float(confidence)

    # --------------------------------------------------------------------- #
    def _keypoints_to_angles(
        self, keypoints: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch version of keypoint-to-angle conversion."""
        B = keypoints.shape[0]
        azimuth = torch.zeros(B)
        elevation = torch.zeros(B)
        confidence = torch.zeros(B)

        for i in range(B):
            az, el, conf = self._single_keypoints_to_angle(keypoints[i], H, W)
            azimuth[i] = az
            elevation[i] = el
            confidence[i] = conf

        return azimuth, elevation, confidence

    # --------------------------------------------------------------------- #
    def _dino_angles(self, imgs: torch.Tensor, H: int, W: int):
        """
        Use DINOv2 features to estimate viewpoint via feature analysis.
        DINOv2 learns viewpoint-aware representations.
        """
        B = imgs.shape[0]

        try:
            # Prepare images for DINO
            from PIL import Image
            import torchvision.transforms.functional as TF

            # Process each image
            features_list = []
            for i in range(B):
                img_pil = TF.to_pil_image(imgs[i].cpu())
                inputs = self._dino_processor(images=img_pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self._dino_model(**inputs)
                cls_token = outputs.last_hidden_state[:, 0]  # (1, D)
                features_list.append(cls_token)

            features = torch.cat(features_list, dim=0)  # (B, D)

            # Analyze features for viewpoint estimation
            # Use PCA-like analysis: first few principal directions often encode viewpoint
            features_centered = features - features.mean(dim=0, keepdim=True)

            # Project onto principal axes (approximation)
            # The first component often correlates with azimuth
            U, S, V = torch.svd(features_centered)

            # Scale projections to angle range
            proj_1 = U[:, 0] * S[0] if len(S) > 0 else torch.zeros(B, device=self.device)
            proj_2 = U[:, 1] * S[1] if len(S) > 1 else torch.zeros(B, device=self.device)

            # Normalize to angle range
            azimuth = proj_1 / (proj_1.abs().max() + 1e-6) * 90  # [-90, 90]
            elevation = proj_2 / (proj_2.abs().max() + 1e-6) * 30  # [-30, 30]

            confidence = torch.ones(B, device=self.device) * 0.5  # Lower confidence for DINO

        except Exception as e:
            raise RuntimeError("[CameraAngle] DINO inference failed.") from e

        return {
            "azimuth": azimuth.cpu(),
            "elevation": elevation.cpu(),
            "confidence": confidence.cpu(),
        }

    # --------------------------------------------------------------------- #
    def _heuristic_angles(self, imgs: torch.Tensor, H: int, W: int):
        """
        Heuristic-based viewpoint estimation using image symmetry.
        Frontal views tend to be more symmetric than side views.
        """
        B = imgs.shape[0]

        azimuth_list = []
        elevation_list = []

        for i in range(B):
            img = imgs[i]  # (3, H, W)

            # Compute horizontal symmetry
            left_half = img[:, :, :W // 2]
            right_half = torch.flip(img[:, :, W // 2:], dims=[2])

            # Resize to same size if needed
            min_w = min(left_half.shape[2], right_half.shape[2])
            left_half = left_half[:, :, :min_w]
            right_half = right_half[:, :, :min_w]

            symmetry = 1 - (left_half - right_half).abs().mean().item()

            # High symmetry = frontal, low symmetry = side view
            azimuth = (1 - symmetry) * 90  # 0-90 degrees

            # Determine direction from intensity distribution
            left_intensity = img[:, :, :W // 2].mean().item()
            right_intensity = img[:, :, W // 2:].mean().item()
            if left_intensity > right_intensity:
                azimuth = -azimuth  # Facing left

            # Elevation from vertical intensity gradient
            top_intensity = img[:, :H // 2, :].mean().item()
            bottom_intensity = img[:, H // 2:, :].mean().item()
            elevation = (bottom_intensity - top_intensity) * 20

            azimuth_list.append(float(azimuth))
            elevation_list.append(float(elevation))

        return {
            "azimuth": torch.tensor(azimuth_list),
            "elevation": torch.tensor(elevation_list),
            "confidence": torch.ones(B) * 0.3,  # Low confidence for heuristic
        }


# ─────────────────────────────────────────────────────────────────────────────
# CameraAngleMetrics
# ─────────────────────────────────────────────────────────────────────────────

class CameraAngleMetrics:
    """
    Accumulates camera angle estimates and computes diversity statistics.

    Metrics computed:
    - Azimuth (horizontal): mean, std, entropy
    - Elevation (vertical): mean, std
    - Combined diversity score
    """

    def __init__(self, device: str = "cpu", n_bins: int = 12):
        """
        Args:
            device: Computation device
            n_bins: Number of bins for entropy calculation (default 12 = 30° bins)
        """
        self._backend = _CameraAngleBackend(device=device)
        self._n_bins = n_bins

        # Accumulators
        self._azimuths: List[float] = []
        self._elevations: List[float] = []
        self._confidences: List[float] = []

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """
        Update with a batch of person images.

        Args:
            person_imgs: (B, 3, H, W) float32 [0, 1]
        """
        angles = self._backend.estimate_angles(person_imgs)

        azimuth = angles["azimuth"]
        elevation = angles["elevation"]
        confidence = angles["confidence"]

        B = azimuth.shape[0]
        for i in range(B):
            self._azimuths.append(float(azimuth[i].item()))
            self._elevations.append(float(elevation[i].item()))
            self._confidences.append(float(confidence[i].item()))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        """
        Compute camera angle diversity metrics.

        Returns:
            dict with azimuth/elevation statistics and diversity scores
        """
        if not self._azimuths:
            return {
                "azimuth_mean": float("nan"),
                "azimuth_std": float("nan"),
                "azimuth_entropy": float("nan"),
                "elevation_mean": float("nan"),
                "elevation_std": float("nan"),
                "camera_diversity_score": float("nan"),
                "frontal_ratio": float("nan"),
                "side_ratio": float("nan"),
            }

        az_arr = np.mod(np.array(self._azimuths), 360.0)
        el_arr = np.array(self._elevations)
        conf_arr = np.array(self._confidences)

        # Signed view of azimuths is used for symmetric category thresholds.
        az_signed = ((az_arr + 180.0) % 360.0) - 180.0

        # ── Basic statistics ──────────────────────────────────────────────────
        az_mean = float(np.mean(az_arr))
        az_std = float(np.std(az_arr))
        el_mean = float(np.mean(el_arr))
        el_std = float(np.std(el_arr))

        # ── Azimuth entropy (diversity measure) ───────────────────────────────
        # Bin azimuths into n_bins sectors
        bin_edges = np.linspace(0, 360, self._n_bins + 1)
        hist, _ = np.histogram(az_arr, bins=bin_edges, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()  # Normalize

        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(self._n_bins)  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy  # [0, 1]

        # ── View categorization ───────────────────────────────────────────────
        # Use signed equivalent azimuth in [-180, 180) for view categories:
        # Frontal: |azimuth| < 30°
        # Side: 60° < |azimuth| < 120°
        # Back: |azimuth| > 150°
        frontal_mask = np.abs(az_signed) < 30
        side_mask = (np.abs(az_signed) > 60) & (np.abs(az_signed) < 120)
        back_mask = np.abs(az_signed) > 150

        n_total = len(az_arr)
        frontal_ratio = float(frontal_mask.sum() / n_total)
        side_ratio = float(side_mask.sum() / n_total)
        back_ratio = float(back_mask.sum() / n_total)

        # ── Combined diversity score ──────────────────────────────────────────
        # Higher is more diverse
        # Components:
        #   - Normalized entropy (0-1): measures spread
        #   - Azimuth std normalized: measures variance
        #   - View balance: penalize if one view dominates
        max_az_std = 90  # Maximum reasonable std
        normalized_az_std = min(az_std / max_az_std, 1.0)

        # View balance score (higher if views are balanced)
        view_probs = np.array([frontal_ratio, side_ratio, back_ratio, 
                               1 - frontal_ratio - side_ratio - back_ratio])
        view_probs = view_probs + 1e-10
        view_probs = view_probs / view_probs.sum()
        view_entropy = -np.sum(view_probs * np.log2(view_probs))
        view_balance = view_entropy / 2.0  # Normalize to ~[0, 1]

        diversity_score = (
            normalized_entropy * 0.4 +
            normalized_az_std * 0.3 +
            view_balance * 0.3
        )

        # ── Confidence-weighted statistics ────────────────────────────────────
        if conf_arr.sum() > 0:
            weighted_az_mean = float(np.average(az_arr, weights=conf_arr))
            avg_confidence = float(np.mean(conf_arr))
        else:
            weighted_az_mean = az_mean
            avg_confidence = 0.0

        return {
            # Azimuth statistics
            "azimuth_mean": az_mean,
            "azimuth_std": az_std,
            "azimuth_entropy": float(normalized_entropy),
            "azimuth_weighted_mean": weighted_az_mean,

            # Elevation statistics
            "elevation_mean": el_mean,
            "elevation_std": el_std,

            # View categorization
            "frontal_ratio": frontal_ratio,
            "side_ratio": side_ratio,
            "back_ratio": back_ratio,

            # Overall diversity
            "camera_diversity_score": float(diversity_score),
            "estimation_confidence": avg_confidence,
        }

    # ------------------------------------------------------------------ #
    def reset(self):
        """Clear accumulated data."""
        self._azimuths.clear()
        self._elevations.clear()
        self._confidences.clear()

    # ------------------------------------------------------------------ #
    def get_angle_distribution(self) -> Dict[str, np.ndarray]:
        """
        Get the raw angle distributions for visualization.

        Returns:
            dict with azimuth and elevation arrays
        """
        return {
            "azimuths": np.array(self._azimuths),
            "elevations": np.array(self._elevations),
            "confidences": np.array(self._confidences),
        }
