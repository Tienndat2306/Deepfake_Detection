"""Single-video inference service used by the Flask app."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

from data.augmentation import get_val_clip_transform
from evaluation.evaluate import infer_feat_dim, resolve_checkpoint_path
from inference.predict import preprocess_frame_for_inference
from models.deepfake_model import DeepfakeDetector
from preprocess.face_detector import FaceDetector
from preprocess.video_utils import get_video_info, read_frames_by_indices, sample_frame_indices


@dataclass(frozen=True)
class AppPaths:
    root: Path
    upload_dir: Path
    keyframe_dir: Path


class DeepfakeInferenceService:
    """Lazy-loaded model and face detector for web inference."""

    def __init__(
        self,
        root_dir: Path,
        config_path: Path | None = None,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.config_path = config_path or (self.root_dir / "configs" / "train_config.yaml")
        self.checkpoint_arg = str(checkpoint_path) if checkpoint_path else None
        self.device = self._resolve_device(device)
        self.config = self._load_config(self.config_path)
        self.model: DeepfakeDetector | None = None
        self.face_detector: FaceDetector | None = None
        self.checkpoint_path: Path | None = None
        self.checkpoint_meta: dict[str, Any] = {}

    def _resolve_device(self, requested: str | None) -> torch.device:
        if requested and requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay config: {path}")
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if "model" not in config:
            model_config_path = path.parent / "model_config.yaml"
            if model_config_path.exists():
                with model_config_path.open("r", encoding="utf-8") as f:
                    model_config = yaml.safe_load(f) or {}
                if "model" in model_config:
                    config["model"] = model_config["model"]
        checkpoint_cfg = config.setdefault("checkpoint", {})
        save_dir = Path(str(checkpoint_cfg.get("save_dir", "checkpoints")))
        if not save_dir.exists() and (self.root_dir / "checkpoints").exists():
            checkpoint_cfg["save_dir"] = str(self.root_dir / "checkpoints")
        return config

    def _load_model(self) -> DeepfakeDetector:
        if self.model is not None:
            return self.model

        model_cfg = self.config.get("model", {})
        input_cfg = model_cfg.get("input", {})
        eff_cfg = model_cfg.get("efficientnet", {})
        transformer_cfg = model_cfg.get("transformer", {})
        dropout = float(eff_cfg.get("dropout", transformer_cfg.get("dropout", 0.3)))

        model = DeepfakeDetector(
            feat_dim=infer_feat_dim(model_cfg),
            d_model=int(transformer_cfg.get("d_model", 512)),
            nhead=int(transformer_cfg.get("nhead", 8)),
            num_layers=int(transformer_cfg.get("num_layers", 4)),
            dropout=dropout,
            num_frames=int(input_cfg.get("num_frames", model_cfg.get("num_frames", 10))),
            pretrained_backbone=False,
        )

        self.checkpoint_path = resolve_checkpoint_path(self.checkpoint_arg, self.config)
        self.checkpoint_meta = model.load_checkpoint(str(self.checkpoint_path))
        model.to(self.device)
        model.eval()
        self.model = model
        return model

    def _load_face_detector(self) -> FaceDetector:
        if self.face_detector is not None:
            return self.face_detector
        aug_path = self.root_dir / "configs" / "aug_config.yaml"
        with aug_path.open("r", encoding="utf-8") as f:
            aug_cfg = yaml.safe_load(f) or {}
        preprocess_cfg = aug_cfg.get("preprocess", {})
        self.face_detector = FaceDetector(
            model_path=self.root_dir / "preprocess" / "models" / "blaze_face_short_range.tflite",
            min_detection_confidence=float(preprocess_cfg.get("min_face_confidence", 0.6)),
            min_face_ratio=float(preprocess_cfg.get("min_face_size_ratio", 0.05)),
        )
        return self.face_detector

    def analyze_video(self, video_path: Path, output_dir: Path) -> dict[str, Any]:
        started = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        detector = self._load_face_detector()
        model_cfg = self.config.get("model", {})
        input_cfg = model_cfg.get("input", {})
        num_frames = int(input_cfg.get("num_frames", model_cfg.get("num_frames", 10)))
        img_size = int(input_cfg.get("img_size", model_cfg.get("image_size", 256)))

        video_info = get_video_info(video_path)
        total_frames = int(video_info.get("total_frames", 0))
        if total_frames <= 0:
            raise RuntimeError("Video khong doc duoc frame hop le.")

        candidate_count = max(num_frames * 3, num_frames)
        frame_indices = sample_frame_indices(total_frames, candidate_count, mode="uniform")
        sampled = read_frames_by_indices(video_path, frame_indices)

        crops: list[np.ndarray] = []
        keyframes: list[dict[str, Any]] = []
        for frame_idx, frame_bgr in sampled:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detection = detector.detect(rgb)
            if detection is None:
                continue
            crop = detector.align_and_crop(
                bgr_frame=frame_bgr,
                detection=detection,
                target_size=(img_size, img_size),
                crop_margin=0.3,
            )
            if crop is None:
                continue
            crops.append(crop)
            thumb_name = f"frame_{len(crops):03d}_{int(frame_idx):06d}.jpg"
            thumb_path = output_dir / thumb_name
            cv2.imwrite(str(thumb_path), crop)
            keyframes.append(
                {
                    "id": len(crops),
                    "frame_number": int(frame_idx),
                    "thumbnail_url": f"/static/uploads/{output_dir.name}/{thumb_name}",
                    "confidence": round(detector.get_detection_score(detection) * 100, 1),
                    "has_artifact": False,
                }
            )
            if len(crops) >= num_frames:
                break

        if not crops:
            raise RuntimeError("Khong detect duoc khuon mat nao trong video.")
        while len(crops) < num_frames:
            crops.append(crops[len(crops) % len(keyframes)])

        clip = get_val_clip_transform(img_size=img_size)(crops[:num_frames]).unsqueeze(0)
        clip = clip.to(self.device, non_blocking=True)
        with torch.no_grad():
            with autocast(enabled=self.device.type == "cuda"):
                logit = model(clip).view(-1)[0]
            fake_probability = float(torch.sigmoid(logit.float()).cpu().item())

        verdict = "Fake" if fake_probability >= 0.5 else "Real"
        confidence = fake_probability if verdict == "Fake" else 1.0 - fake_probability
        for frame in keyframes:
            frame["has_artifact"] = verdict == "Fake" and fake_probability >= 0.65

        file_hash = self._sha256(video_path)
        result = {
            "session_id": output_dir.name,
            "verdict": verdict,
            "fake_probability": round(fake_probability, 6),
            "real_probability": round(1.0 - fake_probability, 6),
            "confidence": round(confidence, 6),
            "threshold": 0.5,
            "video_url": "",
            "keyframes": keyframes,
            "metadata": {
                "sha256": file_hash,
                "filename": video_path.name,
                "total_frames": total_frames,
                "fps": float(video_info.get("fps", 0.0)),
                "width": int(video_info.get("width", 0)),
                "height": int(video_info.get("height", 0)),
                "duration_seconds": float(video_info.get("duration_seconds", 0.0)),
                "faces_used": min(len(crops), num_frames),
                "device": str(self.device),
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_epoch": self.checkpoint_meta.get("epoch"),
                "checkpoint_val_auc": self.checkpoint_meta.get("val_auc"),
                "elapsed_seconds": round(time.time() - started, 3),
            },
        }
        with (output_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
