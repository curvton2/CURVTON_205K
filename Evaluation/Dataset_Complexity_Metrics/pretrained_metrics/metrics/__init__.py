"""
pretrained_metrics/metrics/__init__.py
======================================
Metric classes behind the dataset-complexity numbers reported in the paper.
M6 (appearance) and M8 (VAE latent) are not reported and live in ../../extras/.
"""

from .m1_pose import PoseMetrics
from .m2_occlusion import OcclusionMetrics
from .m3_background import BackgroundMetrics
from .m4_illumination import IlluminationMetrics
from .m5_body_shape import BodyShapeMetrics
from .m7_garment_texture import GarmentTextureMetrics
from .m9_camera_angle import CameraAngleMetrics

__all__ = [
    "PoseMetrics",
    "OcclusionMetrics",
    "BackgroundMetrics",
    "IlluminationMetrics",
    "BodyShapeMetrics",
    "GarmentTextureMetrics",
    "CameraAngleMetrics",
]
