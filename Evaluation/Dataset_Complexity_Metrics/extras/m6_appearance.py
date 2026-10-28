"""
extras/m6_appearance.py
=========================
Metric 6 - Parametric Face Appearance using MediaPipe Face Landmarker.

Face localization backends (in order):
1. PyTorch RetinaFace from yakhyo/retinaface-pytorch (optional)
2. InsightFace detector (optional)
3. OpenCV Haar cascade
4. Top-center fallback crop
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

RETINAFACE_REPO_URL = "https://github.com/yakhyo/retinaface-pytorch.git"
RETINAFACE_WEIGHTS_URL = (
    "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.pth"
)


class _RetinaFaceTorchDetector:
    """Loads yakhyo/retinaface-pytorch dynamically and runs pure PyTorch inference."""

    def __init__(
        self,
        repo_dir: str,
        weights_path: str,
        network: str = "mobilenetv1_0.25",
        device: str = "cpu",
        conf_threshold: float = 0.60,
        nms_threshold: float = 0.40,
        topk: int = 200,
    ):
        self.repo_dir = str(repo_dir)
        self.weights_path = str(weights_path)
        self.network = str(network)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.topk = int(topk)
        self._load_runtime()

    def _load_runtime(self):
        repo = Path(self.repo_dir)
        if not repo.exists():
            raise FileNotFoundError(f"RetinaFace repo_dir not found: {repo}")
        weights = Path(self.weights_path)
        if not weights.exists():
            raise FileNotFoundError(f"RetinaFace weights not found: {weights}")

        names = ["config", "layers", "models", "utils", "utils.box_utils"]
        backups = {n: sys.modules.get(n) for n in names}

        sys.path.insert(0, str(repo))
        try:
            from config import get_config
            from layers import PriorBox
            from models import RetinaFace
            from utils.box_utils import decode, nms
        finally:
            sys.path.pop(0)
            for n, m in backups.items():
                if m is not None:
                    sys.modules[n] = m
                elif n in sys.modules:
                    del sys.modules[n]

        self._get_config = get_config
        self._PriorBox = PriorBox
        self._decode = decode
        self._nms = nms

        self.cfg = self._get_config(self.network)
        if self.cfg is None:
            raise RuntimeError(f"Unsupported RetinaFace backbone/network: {self.network}")

        self.model = RetinaFace(cfg=self.cfg).to(self.device).eval()
        state_dict = torch.load(str(weights), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def detect_largest(self, rgb_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if rgb_image is None or rgb_image.size == 0:
            return None

        h, w = rgb_image.shape[:2]
        if h < 8 or w < 8:
            return None

        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        img = np.float32(bgr)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        tens = torch.from_numpy(img).unsqueeze(0).to(self.device)

        loc, conf, _ = self.model(tens)
        loc = loc.squeeze(0)
        conf = conf.squeeze(0)

        priors = self._PriorBox(self.cfg, image_size=(h, w)).generate_anchors().to(self.device)
        boxes = self._decode(loc, priors, self.cfg["variance"])
        scale = torch.tensor([w, h, w, h], device=self.device)
        boxes = (boxes * scale).cpu().numpy()
        scores = conf[:, 1].cpu().numpy()

        keep = scores > self.conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        if boxes.shape[0] == 0:
            return None

        order = scores.argsort()[::-1][: self.topk]
        boxes = boxes[order]
        scores = scores[order]
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep_idx = self._nms(dets, self.nms_threshold)
        if len(keep_idx) == 0:
            return None

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        idx = int(np.argmax(scores))

        x1, y1, x2, y2 = boxes[idx].tolist()
        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(x1 + 1, min(w, int(round(x2))))
        y2 = max(y1 + 1, min(h, int(round(y2))))
        return x1, y1, x2, y2


class ParametricFaceMetric:
    """Tracks parametric face states (52 blendshapes + 16 transform = 68 dims)."""

    def __init__(
        self,
        device: str = "cpu",
        face_detector_backend: str = "auto",
        retinaface_repo_dir: Optional[str] = None,
        retinaface_weights: Optional[str] = None,
        retinaface_backbone: str = "mobilenetv1_0.25",
        retinaface_device: Optional[str] = None,
    ):
        self.device = device
        self._blendshapes_list: List[np.ndarray] = []
        self._embeddings = self._blendshapes_list

        self._face_detector_backend = str(face_detector_backend or "auto").lower()
        self._retinaface_repo_dir = retinaface_repo_dir or os.environ.get("RETINAFACE_PYTORCH_DIR")
        self._retinaface_weights = retinaface_weights or os.environ.get("RETINAFACE_PYTORCH_WEIGHTS")
        self._retinaface_backbone = retinaface_backbone or os.environ.get(
            "RETINAFACE_PYTORCH_BACKBONE", "mobilenetv1_0.25"
        )
        self._retinaface_device = retinaface_device or os.environ.get("RETINAFACE_PYTORCH_DEVICE", "cpu")

        self._retinaface_torch = None
        self._face_app = None
        self._face_cascade = None
        self._available = False

        self._load()

    @staticmethod
    def _default_retinaface_paths() -> Tuple[Path, Path]:
        root = Path(os.environ.get("RETINAFACE_PYTORCH_HOME", Path.home() / ".cache" / "retinaface_pytorch"))
        return root / "retinaface-pytorch", root / "retinaface_mv1_0.25.pth"

    def _ensure_retinaface_artifacts(self):
        if self._face_detector_backend not in ("auto", "retinaface_pytorch"):
            return

        repo_path = Path(self._retinaface_repo_dir) if self._retinaface_repo_dir else None
        weights_path = Path(self._retinaface_weights) if self._retinaface_weights else None
        default_repo, default_weights = self._default_retinaface_paths()

        if repo_path is None:
            repo_path = default_repo
            self._retinaface_repo_dir = str(repo_path)
        if weights_path is None:
            weights_path = default_weights
            self._retinaface_weights = str(weights_path)

        repo_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        if not repo_path.exists():
            print(f"[FaceParametric] Cloning RetinaFace repo to {repo_path} ...")
            subprocess.run(["git", "clone", RETINAFACE_REPO_URL, str(repo_path)], check=True)
        else:
            print(f"[FaceParametric] RetinaFace repo already present at {repo_path}, skipping clone.")

        if not weights_path.exists():
            print(f"[FaceParametric] Downloading RetinaFace weights to {weights_path} ...")
            urllib.request.urlretrieve(RETINAFACE_WEIGHTS_URL, str(weights_path))
        else:
            print(f"[FaceParametric] RetinaFace weights already present at {weights_path}, skipping download.")

    @staticmethod
    def _square_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))

        raw = img[y1:y2, x1:x2]
        if raw.size == 0:
            return None

        ch, cw = raw.shape[:2]
        side = max(ch, cw)
        sq = np.zeros((side, side, 3), dtype=np.uint8)
        oy = (side - ch) // 2
        ox = (side - cw) // 2
        sq[oy : oy + ch, ox : ox + cw] = raw
        return sq

    def _load(self):
        try:
            self._ensure_retinaface_artifacts()
        except Exception as e:
            print(f"[FaceParametric] RetinaFace auto-setup skipped: {e}")

        try_retina = self._face_detector_backend in ("auto", "retinaface_pytorch")
        if try_retina and self._retinaface_repo_dir and self._retinaface_weights:
            try:
                self._retinaface_torch = _RetinaFaceTorchDetector(
                    repo_dir=self._retinaface_repo_dir,
                    weights_path=self._retinaface_weights,
                    network=self._retinaface_backbone,
                    device=self._retinaface_device,
                )
                print(
                    f"[FaceParametric] Loaded PyTorch RetinaFace ({self._retinaface_backbone}) "
                    f"from {self._retinaface_weights}"
                )
            except Exception as e:
                print(f"[FaceParametric] PyTorch RetinaFace unavailable: {e}")

        try_insight = self._face_detector_backend in ("auto", "insightface")
        if try_insight:
            try:
                from insightface.app import FaceAnalysis

                quiet_out = io.StringIO()
                warnings.filterwarnings("ignore")
                with contextlib.redirect_stdout(quiet_out), contextlib.redirect_stderr(quiet_out):
                    self._face_app = FaceAnalysis(
                        providers=["CPUExecutionProvider"],
                        allowed_modules=["detection"],
                    )
                    self._face_app.prepare(ctx_id=-1, det_size=(320, 320))
                print("[FaceParametric] Loaded InsightFace detector.")
            except Exception as e:
                print(f"[FaceParametric] InsightFace detector unavailable: {e}")
                self._face_app = None

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
            if not os.path.exists(model_path):
                print("[FaceParametric] Downloading MediaPipe face_landmarker.task...")
                url = (
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                    "face_landmarker/float16/1/face_landmarker.task"
                )
                urllib.request.urlretrieve(url, model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=0.1,
                min_face_presence_confidence=0.1,
                min_tracking_confidence=0.1,
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
            self._mp_image = mp.Image
            self._mp_format = mp.ImageFormat.SRGB
            self._available = True
            print("[FaceParametric] Loaded MediaPipe parametric face model (68-D).")
        except Exception as e:
            print(f"[FaceParametric] Could not load MediaPipe face model: {e}")
            self._available = False

    def _crop_face(self, img_full: np.ndarray) -> np.ndarray:
        h, w, _ = img_full.shape

        if self._retinaface_torch is not None:
            try:
                box = self._retinaface_torch.detect_largest(img_full)
                if box is not None:
                    sq = self._square_crop(img_full, *box)
                    if sq is not None:
                        return sq
            except Exception:
                pass

        if self._face_app is not None:
            try:
                faces = self._face_app.get(cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR))
                if faces:
                    faces = sorted(
                        faces,
                        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
                        reverse=True,
                    )
                    b = faces[0].bbox
                    sq = self._square_crop(img_full, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    if sq is not None:
                        return sq
            except Exception:
                pass

        if self._face_cascade is None:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self._face_cascade = cv2.CascadeClassifier(cascade_path)

        try:
            gray = cv2.cvtColor(img_full, cv2.COLOR_RGB2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(15, 15),
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, fw, fh = faces[0]
                mx = int(fw * 0.35)
                my = int(fh * 0.35)
                sq = self._square_crop(img_full, x - mx, y - my, x + fw + mx, y + fh + my)
                if sq is not None:
                    return sq
        except Exception:
            pass

        # Final top-center fallback.
        crop_h = max(1, int(h * 0.25))
        crop_w0 = max(0, int(w * 0.25))
        crop_w1 = min(w, int(w * 0.75))
        sq = self._square_crop(img_full, crop_w0, 0, crop_w1, crop_h)
        if sq is None:
            return img_full
        return sq

    @torch.no_grad()
    def update(self, person_imgs: torch.Tensor):
        if not self._available:
            return

        imgs = (person_imgs.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        for img_full in imgs:
            img_crop = self._crop_face(img_full)
            mp_img = self._mp_image(image_format=self._mp_format, data=np.ascontiguousarray(img_crop))

            try:
                res = self._detector.detect(mp_img)
                if not getattr(res, "face_blendshapes", None):
                    continue

                bs = res.face_blendshapes[0]
                b_scores = np.array([cat.score for cat in bs], dtype=np.float32)

                if getattr(res, "facial_transformation_matrixes", None):
                    t_matrix = res.facial_transformation_matrixes[0].flatten().astype(np.float32)
                else:
                    t_matrix = np.zeros(16, dtype=np.float32)

                params = np.concatenate([b_scores, t_matrix])
                self._blendshapes_list.append(params)
            except Exception:
                continue

    def compute(self) -> Dict[str, float]:
        if not self._blendshapes_list:
            return {
                "appearance_complexity": float("nan"),
                "appearance_diversity_mean": float("nan"),
                "appearance_diversity_std": float("nan"),
                "appearance_diversity_cov_norm": float("nan"),
                "appearance_diversity_logdet": float("nan"),
                "n_faces": 0,
            }

        e = np.stack(self._blendshapes_list, axis=0)
        n, d = e.shape
        if n < 2:
            return {
                "appearance_complexity": float("nan"),
                "appearance_diversity_mean": float("nan"),
                "appearance_diversity_std": float("nan"),
                "appearance_diversity_cov_norm": float("nan"),
                "appearance_diversity_logdet": float("nan"),
                "n_faces": n,
            }

        mu = e.mean(axis=0, keepdims=True)
        cov = (e - mu).T @ (e - mu) / (n - 1)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        var_total = float(np.sum(eigvals))

        if var_total > 0:
            cvar = np.cumsum(eigvals) / var_total
            erank = min(int(np.searchsorted(cvar, 0.95)) + 1, n - 1)
        else:
            erank = 0

        if erank > 0:
            sig = eigvals[:erank] + 1e-6
            log_det = float(np.sum(np.log(sig)))
            cov_norm = -log_det / erank
            log_det_norm = log_det / erank
        else:
            log_det = float("nan")
            cov_norm = float("nan")
            log_det_norm = float("nan")

        # Pairwise cosine distance over parametric embeddings.
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        cdist = 1.0 - (en @ en.T)
        tri = np.triu_indices(n, k=1)
        mean_div = float(np.mean(cdist[tri])) if tri[0].size > 0 else float("nan")
        std_div = float(np.std(cdist[tri])) if tri[0].size > 0 else float("nan")

        return {
            "appearance_complexity": float(np.mean(np.linalg.norm(e, axis=1))),
            "appearance_diversity_mean": mean_div,
            "appearance_diversity_std": std_div,
            "appearance_diversity_cov_norm": cov_norm,
            "appearance_diversity_logdet": log_det,
            "appearance_diversity_logdet_raw": log_det,
            "appearance_diversity_logdet_normalized": log_det_norm,
            "n_faces": n,
        }

    def reset(self):
        self._blendshapes_list.clear()


AppearanceMetrics = ParametricFaceMetric
