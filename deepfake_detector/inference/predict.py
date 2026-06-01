"""Inference-time frame preprocessing helpers.

Detect on the original BGR frame, crop the selected face, then resize the
face crop to the model input size. Callers must not resize the full frame
before detection because detection boxes are tied to the frame shape.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from preprocess.face_detector import FaceDetector
from preprocess.preprocess import _detect_and_crop


def preprocess_frame_for_inference(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    target_size: int = 256,
    crop_margin: float = 0.3,
) -> Optional[np.ndarray]:
    """Return a cropped face resized to target_size from an unresized BGR frame."""
    face, _bbox = _detect_and_crop(
        frame_bgr=frame_bgr,
        detector=detector,
        target_size=int(target_size),
        crop_margin=float(crop_margin),
    )
    return face
