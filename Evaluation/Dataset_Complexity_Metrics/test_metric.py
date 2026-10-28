"""
test_metric.py — Run one metric (M1–M9) on every dataset in a config and print its outputs.

    python test_metric.py --metric m1 --config configs/pretrained_metrics_datasets.yaml --device cuda
    python test_metric.py --metric m9 --curvton_root /path/to/dataset_ultimate

M6 (appearance) and M8 (VAE latent) are not reported in the paper; their code is in extras/.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import Dict, List, Tuple

import torch

from metric_test_common import add_common_args, run_metric_on_all_datasets


MetricBatch = List[Dict[str, torch.Tensor]]


def _backend_attr(path: str):
    parts = path.split(".")

    def _get(obj):
        cur = obj
        for p in parts:
            cur = getattr(cur, p, None)
            if cur is None:
                return "unknown"
        return cur

    return _get


def _m7_backend(obj) -> str:
    return ",".join(sorted(getattr(obj._encoder, "_models", {}).keys())) or "unknown"


SPECS = {
    "m1": {
        "module": "pretrained_metrics.metrics.m1_pose",
        "class": "PoseMetrics",
        "title": "M1 Pose",
        "summary_key": "pose_diversity",
        "summary_label": "Pose Diversity",
        "input_key": "person",
        "backend": _backend_attr("extractor._backend"),
        "set_home_for_hmr2": False,
        "set_defaults": {"continue_on_error": False, "error_log": "logs/test_m1_pose_errors.log"},
    },
    "m2": {
        "module": "pretrained_metrics.metrics.m2_occlusion",
        "class": "OcclusionMetrics",
        "title": "M2 Occlusion",
        "summary_key": "occlusion_complexity",
        "summary_label": "Occlusion Complexity",
        "input_key": "person",
        "backend": _backend_attr("_seg._backend"),
        "set_home_for_hmr2": False,
    },
    "m3": {
        "module": "pretrained_metrics.metrics.m3_background",
        "class": "BackgroundMetrics",
        "title": "M3 Background",
        "summary_key": "bg_overall_complexity",
        "summary_label": "Background Overall Complexity",
        "input_key": "person",
        "backend": lambda obj: f"seg={getattr(getattr(obj, '_segmenter', None), '_backend', 'unknown')},"
        f"det={getattr(getattr(obj, '_detector', None), '_backend', 'unknown')}",
        "set_home_for_hmr2": False,
    },
    "m4": {
        "module": "pretrained_metrics.metrics.m4_illumination",
        "class": "IlluminationMetrics",
        "title": "M4 Illumination",
        "summary_key": "illumination_richness_score",
        "summary_label": "Illumination Richness Score",
        "input_key": "person",
        "backend": lambda _obj: "signal_processing",
        "set_home_for_hmr2": False,
    },
    "m5": {
        "module": "pretrained_metrics.metrics.m5_body_shape",
        "class": "BodyShapeMetrics",
        "title": "M5 Body Shape",
        "summary_key": "shape_diversity_logdet",
        "summary_label": "Body Shape Diversity (logdet)",
        "input_key": "person",
        "backend": _backend_attr("_extractor._backend"),
        "set_home_for_hmr2": True,
    },
    "m6": {
        "module": "extras.m6_appearance",
        "class": "AppearanceMetrics",
        "title": "M6 Appearance",
        "summary_key": "appearance_diversity_mean",
        "summary_label": "Appearance Diversity Mean",
        "input_key": "person",
        "backend": _backend_attr("_embedder._backend"),
        "set_home_for_hmr2": False,
    },
    "m7": {
        "module": "pretrained_metrics.metrics.m7_garment_texture",
        "class": "GarmentTextureMetrics",
        "title": "M7 Garment Texture",
        "summary_key": "garment_ensemble_score_0_1",
        "summary_label": "Garment Texture Ultimate Ensemble Score (0-1)",
        "input_key": "cloth",
        "backend": _m7_backend,
        "set_home_for_hmr2": False,
    },
    "m8": {
        "module": "extras.m8_vae_latent",
        "class": "VAELatentMetric",
        "title": "M8 VAE Latent",
        "summary_key": "vae_diversity_logdet",
        "summary_label": "VAE Latent Diversity (logdet)",
        "input_key": "person",
        "backend": _backend_attr("_encoder._backend"),
        "set_home_for_hmr2": False,
    },
    "m9": {
        "module": "pretrained_metrics.metrics.m9_camera_angle",
        "class": "CameraAngleMetrics",
        "title": "M9 Camera Angle",
        "summary_key": "camera_diversity_score",
        "summary_label": "Camera Angle Diversity Score",
        "input_key": "person",
        "backend": _backend_attr("_backend._backend"),
        "set_home_for_hmr2": True,
        "set_defaults": {"continue_on_error": False, "error_log": "logs/test_m9_camera_errors.log"},
    },
}


def _build_obj(metric_id: str, device: str, args):
    spec = SPECS[metric_id]
    mod = importlib.import_module(spec["module"])
    cls = getattr(mod, spec["class"])
    if metric_id in {"m1", "m2", "m3", "m5", "m7", "m8", "m9"}:
        return cls(device=device)
    if metric_id == "m4":
        return cls()
    if metric_id == "m6":
        return cls(
            device=device,
            face_detector_backend=args.m6_face_detector_backend,
            retinaface_repo_dir=args.m6_retinaface_repo_dir,
            retinaface_weights=args.m6_retinaface_weights,
            retinaface_backbone=args.m6_retinaface_backbone,
            retinaface_device=args.m6_retinaface_device,
        )
    raise ValueError(f"Unsupported metric id: {metric_id}")


def _probe(metric_id: str, device: str, batches: MetricBatch, args) -> Tuple[Dict, str, float]:
    spec = SPECS[metric_id]
    obj = _build_obj(metric_id, device, args)
    t0 = time.time()
    for batch in batches:
        obj.update(batch[spec["input_key"]])
    result = obj.compute()
    backend = spec["backend"](obj)
    return result, str(backend), time.time() - t0


def run(metric_id: str, argv: List[str] | None = None) -> int:
    if metric_id not in SPECS:
        raise ValueError(f"Unknown metric id: {metric_id}")
    spec = SPECS[metric_id]

    parser = argparse.ArgumentParser(description=f"{spec['title']} metric test across all configured datasets")
    add_common_args(parser)
    parser.add_argument("--m6_face_detector_backend", type=str, default="retinaface_pytorch",
                        choices=["auto", "retinaface_pytorch", "insightface", "haar"])
    parser.add_argument("--m6_retinaface_repo_dir", type=str, default=None)
    parser.add_argument("--m6_retinaface_weights", type=str, default=None)
    parser.add_argument("--m6_retinaface_backbone", type=str, default="mobilenetv1_0.25")
    parser.add_argument("--m6_retinaface_device", type=str, default="cpu")

    if "set_defaults" in spec:
        parser.set_defaults(**spec["set_defaults"])

    args = parser.parse_args(argv)
    if metric_id == "m2":
        if not args.only_datasets:
            args.only_datasets = ["dresscode", "street_tryon"]
        if not args.only_dresscode_categories:
            args.only_dresscode_categories = ["dresses"]

    return run_metric_on_all_datasets(
        args=args,
        metric_title=spec["title"],
        probe_fn=lambda d, b, a: _probe(metric_id, d, b, a),
        paper_score_key=spec["summary_key"],
        paper_score_label=spec["summary_label"],
        set_home_for_hmr2=spec["set_home_for_hmr2"],
    )


def main() -> int:
    parser = argparse.ArgumentParser("Run M1-M9 metric test")
    parser.add_argument("--metric", required=True, choices=sorted(SPECS.keys()))
    args, rest = parser.parse_known_args()
    return run(args.metric, rest)


if __name__ == "__main__":
    sys.exit(main())

