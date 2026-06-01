"""Face detection + alignment module based on MediaPipe Tasks API.

Cải tiến so với bản gốc (tăng tỉ lệ detect mặt):
  1. Multi-scale detect: nếu miss ở full-res → upscale frame nhỏ trước khi detect.
  2. Confidence fallback: nếu miss ở confidence cao → tự thử lại ở confidence thấp hơn.
  3. Sửa bbox clamp sau rotation: clamp FLOAT trước khi cast int để tránh crop rỗng.
  4. Kiểm tra face_size_ratio: loại face giả (bbox quá nhỏ so với frame).
  5. detect() trả về list thay vì object để caller dễ lọc / retry.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

BoundingBox = Tuple[int, int, int, int]
Point2D = Tuple[float, float]
LOGGER = logging.getLogger(__name__)

# Kích thước tối thiểu một cạnh ảnh để BlazeFace hoạt động tốt.
# Nếu frame nhỏ hơn ngưỡng này, upscale trước khi detect rồi chiếu ngược bbox.
_MIN_DETECT_SIZE = 320


class FaceDetector:
    """
    Face detector dùng MediaPipe BlazeFace Tasks API.

    Mỗi instance giữ MỘT detector ở confidence chính (primary_confidence).
    Khi detect thất bại, tự thử lại với fallback_confidence (thấp hơn).
    Hai mức confidence dùng chung model asset → không tốn thêm bộ nhớ model.
    """

    DEFAULT_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    )

    _LINUX_LIB_HINTS = {
        "libGLESv2.so.2": "libgles2",
        "libEGL.so.1":    "libegl1",
        "libGL.so.1":     "libgl1",
        "libglib-2.0.so.0": "libglib2.0-0",
    }

    def __init__(
        self,
        model_path: str | Path = "preprocess/models/blaze_face_short_range.tflite",
        min_detection_confidence: float = 0.5,
        fallback_confidence: float = 0.3,
        min_face_ratio: float = 0.03,
    ) -> None:
        """
        Args:
            min_detection_confidence: Confidence chính — dùng trước tiên.
                Giảm từ 0.6 → 0.5 để không bỏ sót mặt hơi nghiêng/mờ.
            fallback_confidence: Nếu detect miss với confidence chính,
                thử lại với mức này (mặc định 0.3).
                Giá trị thấp hơn → bắt được nhiều hơn nhưng có thể có false positive.
            min_face_ratio: Diện tích bbox / diện tích frame tối thiểu để chấp nhận.
                Lọc các bbox quá nhỏ (noise, background face xa).
                Mặc định 0.03 (bbox chiếm ≥3% diện tích frame).
        """
        if str(model_path) == "preprocess/models/blaze_face_short_range.tflite":
            model_path = Path(__file__).parent / "models" / "blaze_face_short_range.tflite"
        self.model_path = Path(model_path)
        self.primary_confidence  = float(min_detection_confidence)
        self.fallback_confidence = float(fallback_confidence)
        self.min_face_ratio      = float(min_face_ratio)

        # Backward-compat: attribute cũ dùng bởi clean_processed_dataset
        self.min_detection_confidence = self.primary_confidence

        self.download_model_if_needed(self.model_path)

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:
            raise ImportError(
                "Can cai mediapipe. Chay: pip install mediapipe"
            ) from exc

        self._mp        = mp
        self._mp_python = mp_python
        self._mp_vision = mp_vision

        self._detector_primary  = self._build_detector(self.primary_confidence)
        # Chỉ tạo fallback detector khi khác primary để tiết kiệm RAM.
        if abs(self.fallback_confidence - self.primary_confidence) > 1e-6:
            self._detector_fallback = self._build_detector(self.fallback_confidence)
        else:
            self._detector_fallback = self._detector_primary

    def _build_detector(self, confidence: float) -> Any:
        base = self._mp_python.BaseOptions(model_asset_path=str(self.model_path))
        opts = self._mp_vision.FaceDetectorOptions(
            base_options=base,
            running_mode=self._mp_vision.RunningMode.IMAGE,
            min_detection_confidence=confidence,
        )
        try:
            return self._mp_vision.FaceDetector.create_from_options(opts)
        except OSError as exc:
            raise RuntimeError(self._build_mediapipe_runtime_error(exc)) from exc

    # ── Runtime error helper ──────────────────────────────────────────────────

    @classmethod
    def _build_mediapipe_runtime_error(cls, exc: OSError) -> str:
        exc_text = str(exc)
        missing_lib = next((l for l in cls._LINUX_LIB_HINTS if l in exc_text), None)
        lines = [
            "Khong khoi tao duoc MediaPipe FaceDetector Tasks API.",
            f"Chi tiet: {exc_text}",
        ]
        if os.name == "posix":
            if missing_lib:
                pkg = cls._LINUX_LIB_HINTS[missing_lib]
                lines += [
                    f"Thu vien bi thieu: {missing_lib}",
                    f"  sudo apt-get install -y {pkg}",
                ]
            else:
                lines.append(
                    "  sudo apt-get install -y libgles2 libegl1 libgl1 libglib2.0-0"
                )
        lines.append("Restart process Python sau khi cai xong.")
        return "\n".join(lines)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        for attr in ("_detector_primary", "_detector_fallback"):
            d = getattr(self, attr, None)
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
            setattr(self, attr, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ── Model download ────────────────────────────────────────────────────────

    @classmethod
    def download_model_if_needed(cls, model_path: str | Path) -> Path:
        model_file = Path(model_path)
        if model_file.exists():
            return model_file
        model_file.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(cls.DEFAULT_MODEL_URL, model_file)
        return model_file

    # ── Internal detect helpers ───────────────────────────────────────────────

    @staticmethod
    def _bbox_from_detection(detection: Any) -> BoundingBox:
        bbox = detection.bounding_box
        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(bbox.origin_x + bbox.width)
        y2 = int(bbox.origin_y + bbox.height)
        return x1, y1, x2, y2

    @staticmethod
    def _keypoint_to_pixel(keypoint: Any, frame_w: int, frame_h: int) -> Point2D:
        x, y = float(keypoint.x), float(keypoint.y)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * frame_w, y * frame_h
        return x, y

    def _run_detector(self, rgb_frame: np.ndarray, use_fallback: bool = False) -> List[Any]:
        """Chạy detector trên rgb_frame, trả về list detections (có thể rỗng)."""
        detector = self._detector_fallback if use_fallback else self._detector_primary
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)
        return result.detections if result else []

    def _best_detection(
        self,
        detections: List[Any],
        frame_h: int,
        frame_w: int,
    ) -> Optional[Any]:
        """
        Lọc + chọn khuôn mặt tốt nhất:
        - Loại bbox quá nhỏ (noise / mặt nền quá xa).
        - Ưu tiên mặt đủ lớn và nằm gần tâm khung hình.
        """
        frame_area = float(frame_h * frame_w)
        if frame_area <= 0.0:
            return None

        alpha = 0.5
        frame_cx = frame_w * 0.5
        frame_cy = frame_h * 0.5
        max_center_dist = float(np.hypot(frame_cx, frame_cy)) or 1.0
        valid = []
        for d in detections:
            bbox = d.bounding_box
            area = float(bbox.width) * float(bbox.height)
            area_ratio = area / frame_area
            if area_ratio >= self.min_face_ratio:
                face_cx = float(bbox.origin_x) + float(bbox.width) * 0.5
                face_cy = float(bbox.origin_y) + float(bbox.height) * 0.5
                dist_to_center = float(np.hypot(face_cx - frame_cx, face_cy - frame_cy))
                dist_norm = dist_to_center / max_center_dist
                score = area_ratio - alpha * dist_norm
                valid.append((score, d))
        if not valid:
            return None
        return max(valid, key=lambda x: x[0])[1]

    # ── Multi-scale + fallback detect ─────────────────────────────────────────

    def _detect_with_upscale(self, rgb_frame: np.ndarray) -> Optional[Any]:
        """
        Detect với multi-scale strategy:
        1. Full-res, primary confidence.
        2. Nếu miss → full-res, fallback confidence.
        3. Nếu miss → upscale cạnh ngắn lên _MIN_DETECT_SIZE, primary confidence.
        4. Nếu miss → upscale + fallback confidence.
        Bbox từ upscale được chiếu ngược về kích thước gốc.
        """
        h, w = rgb_frame.shape[:2]

        # Bước 1 & 2: full-res
        for use_fallback in (False, True):
            dets = self._run_detector(rgb_frame, use_fallback=use_fallback)
            best = self._best_detection(dets, h, w)
            if best is not None:
                return best

        # Bước 3 & 4: upscale nếu frame nhỏ
        short_side = min(h, w)
        if short_side >= _MIN_DETECT_SIZE:
            return None  # Frame đã đủ lớn, không cần upscale

        scale = _MIN_DETECT_SIZE / short_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        upscaled = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        for use_fallback in (False, True):
            dets = self._run_detector(upscaled, use_fallback=use_fallback)
            best = self._best_detection(dets, new_h, new_w)
            if best is None:
                continue
            # Chiếu bbox + keypoints về kích thước gốc
            return self._rescale_detection(best, scale, is_normalized=True)

        return None

    @staticmethod
    def _rescale_detection(
        detection: Any,
        scale: float,
        is_normalized: bool | None = True,
    ) -> Any:
        """
        Tạo proxy object chứa bbox và keypoints đã chia lại theo scale.
        Dùng SimpleNamespace để giữ tương thích với phần code dùng attribute access.
        """
        from types import SimpleNamespace

        bbox = detection.bounding_box
        new_bbox = SimpleNamespace(
            origin_x = bbox.origin_x / scale,
            origin_y = bbox.origin_y / scale,
            width    = bbox.width    / scale,
            height   = bbox.height   / scale,
        )

        new_keypoints = []
        for kp in (getattr(detection, "keypoints", None) or []):
            x, y = float(kp.x), float(kp.y)
            if is_normalized is True:
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    new_keypoints.append(SimpleNamespace(x=x, y=y))
                else:
                    LOGGER.warning(
                        "Expected normalized face keypoint while rescaling, "
                        "but got pixel-like coordinates; applying pixel rescale."
                    )
                    new_keypoints.append(SimpleNamespace(x=x / scale, y=y / scale))
            elif is_normalized is False:
                new_keypoints.append(SimpleNamespace(x=x / scale, y=y / scale))
            else:
                # Backward-compatible heuristic for legacy callers that cannot
                # state the coordinate space explicitly.
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    LOGGER.warning(
                        "Ambiguous face keypoint in [0, 1] while rescaling; "
                        "assuming normalized coordinates. Pass is_normalized "
                        "explicitly to avoid heuristic behavior."
                    )
                    new_keypoints.append(SimpleNamespace(x=x, y=y))
                else:
                    new_keypoints.append(SimpleNamespace(x=x / scale, y=y / scale))

        categories = getattr(detection, "categories", None) or []

        return SimpleNamespace(
            bounding_box = new_bbox,
            keypoints    = new_keypoints,
            categories   = categories,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, rgb_frame: np.ndarray) -> Optional[Any]:
        """
        Detect khuôn mặt lớn nhất từ frame RGB.
        Tự động thử multi-scale + fallback confidence nếu miss.
        Trả về detection object hoặc None.
        """
        if rgb_frame is None or rgb_frame.ndim != 3:
            return None
        result = self._detect_with_upscale(rgb_frame)
        if result is not None:
            return result

        h, w = rgb_frame.shape[:2]
        if h > 480 and w > 640:
            y1 = max(0, h//2 - int(h*0.3))
            y2 = min(h, h//2 + int(h*0.3))
            x1 = max(0, w//2 - int(w*0.3))
            x2 = min(w, w//2 + int(w*0.3))
            center_crop = rgb_frame[y1:y2, x1:x2]
            result = self._detect_with_upscale(center_crop)
            if result is not None:
                bbox = result.bounding_box
                bbox.origin_x += x1
                bbox.origin_y += y1
                # rescale keypoints nếu có
                for kp in (getattr(result, 'keypoints', None) or []):
                    if not (0.0 <= kp.x <= 1.0):
                        kp.x += x1
                        kp.y += y1
                return result

        return None

    @staticmethod
    def get_detection_score(detection: Any) -> float:
        """Lấy score của detection; trả 0.0 nếu không có categories/score."""
        if detection is None:
            return 0.0
        cats = getattr(detection, "categories", None) or []
        if not cats or not hasattr(cats[0], "score"):
            return 0.0
        try:
            return float(cats[0].score)
        except Exception:
            return 0.0

    @staticmethod
    def get_detection_bbox(detection: Any) -> BoundingBox:
        """Lấy bbox (x1, y1, x2, y2) từ detection theo hệ tọa độ pixel."""
        if detection is None:
            return 0, 0, 0, 0
        return FaceDetector._bbox_from_detection(detection)

    def crop_from_bbox(
        self,
        bgr_frame: np.ndarray,
        bbox: BoundingBox,
        target_size: Tuple[int, int] = (256, 256),
        crop_margin: float = 0.3,
    ) -> Optional[np.ndarray]:
        """
        Crop từ bbox có sẵn (không detect lại), dùng cùng logic clamp/margin như align_and_crop.
        Hữu ích cho fallback dùng bbox frame trước.
        """
        if bgr_frame is None or bgr_frame.ndim != 3:
            return None
        if bbox is None or len(bbox) != 4:
            return None

        h, w = bgr_frame.shape[:2]
        x1_raw, y1_raw, x2_raw, y2_raw = [float(v) for v in bbox]

        # Chuẩn hóa bbox theo thứ tự trái-phải, trên-dưới trước khi clamp.
        x_left = min(x1_raw, x2_raw)
        y_top = min(y1_raw, y2_raw)
        x_right = max(x1_raw, x2_raw)
        y_bottom = max(y1_raw, y2_raw)

        # Clamp float trước khi cast int để tránh crop rỗng do rounding.
        x1f = float(np.clip(x_left, 0.0, w - 1.0))
        y1f = float(np.clip(y_top, 0.0, h - 1.0))
        x2f = float(np.clip(x_right, x1f + 1.0, float(w)))
        y2f = float(np.clip(y_bottom, y1f + 1.0, float(h)))

        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        if x2 <= x1 or y2 <= y1:
            return None

        bw, bh = x2 - x1, y2 - y1
        mx = int(round(bw * float(crop_margin)))
        my = int(round(bh * float(crop_margin)))

        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = bgr_frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        tw, th = int(target_size[0]), int(target_size[1])
        return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    def _extract_eye_points(
        self, detection: Any, frame_w: int, frame_h: int
    ) -> Tuple[Optional[Point2D], Optional[Point2D]]:
        keypoints = getattr(detection, "keypoints", None) or []
        if len(keypoints) < 2:
            return None, None
        right_eye = self._keypoint_to_pixel(keypoints[0], frame_w, frame_h)
        left_eye  = self._keypoint_to_pixel(keypoints[1], frame_w, frame_h)
        return right_eye, left_eye

    def align_and_crop(
        self,
        bgr_frame: np.ndarray,
        detection: Any,
        target_size: Tuple[int, int] = (256, 256),
        crop_margin: float = 0.3,
    ) -> Optional[np.ndarray]:
        """
        Align + crop khuôn mặt.

        Sửa so với bản gốc:
        - Clamp bbox bằng float TRƯỚC khi cast int → tránh x2==x1 do rounding.
        - Clamp riêng cx2/cy2 thành min(w, ...) thay vì min(w-1, ...) để
          cho phép slice đến hết edge.
        - Kiểm tra kích thước crop sau khi clip, trước khi resize.
        """
        if bgr_frame is None or bgr_frame.ndim != 3 or detection is None:
            return None

        h, w = bgr_frame.shape[:2]
        bbox = detection.bounding_box

        # ── Lấy bbox float trước, clamp float, rồi mới cast int ──────────────
        x1f = float(np.clip(bbox.origin_x,               0.0, w - 1.0))
        y1f = float(np.clip(bbox.origin_y,               0.0, h - 1.0))
        x2f = float(np.clip(bbox.origin_x + bbox.width,  x1f + 1.0, float(w)))
        y2f = float(np.clip(bbox.origin_y + bbox.height, y1f + 1.0, float(h)))

        right_eye, left_eye = self._extract_eye_points(detection, frame_w=w, frame_h=h)
        rotated = bgr_frame

        if right_eye is not None and left_eye is not None:
            dx = left_eye[0] - right_eye[0]
            dy = left_eye[1] - right_eye[1]
            angle = float(np.clip(np.degrees(np.arctan2(dy, dx)), -45.0, 45.0))
            eye_center = (
                (right_eye[0] + left_eye[0]) * 0.5,
                (right_eye[1] + left_eye[1]) * 0.5,
            )
            rot_m = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            rotated = cv2.warpAffine(
                bgr_frame, rot_m, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Biến đổi 4 góc bbox theo ma trận xoay
            corners = np.array(
                [[x1f, y1f, 1.0], [x2f, y1f, 1.0],
                 [x1f, y2f, 1.0], [x2f, y2f, 1.0]],
                dtype=np.float32,
            )
            transformed = corners @ rot_m.T

            # Clamp float TRƯỚC khi cast int (fix bug bản gốc)
            x1f = float(np.clip(np.min(transformed[:, 0]), 0.0, float(w)))
            y1f = float(np.clip(np.min(transformed[:, 1]), 0.0, float(h)))
            x2f = float(np.clip(np.max(transformed[:, 0]), x1f + 1.0, float(w)))
            y2f = float(np.clip(np.max(transformed[:, 1]), y1f + 1.0, float(h)))

        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        if x2 <= x1 or y2 <= y1:
            return None

        # ── Mở rộng bbox với margin ───────────────────────────────────────────
        bw, bh = x2 - x1, y2 - y1
        mx = int(round(bw * float(crop_margin)))
        my = int(round(bh * float(crop_margin)))

        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w,  x2 + mx)
        cy2 = min(h,  y2 + my)

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = rotated[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        tw, th = int(target_size[0]), int(target_size[1])
        return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    def visualize_detection(self, frame: np.ndarray, detection: Any) -> np.ndarray:
        """Vẽ bbox + keypoints lên frame (dùng cho EDA)."""
        vis = frame.copy()
        if detection is None:
            return vis

        h, w = vis.shape[:2]
        bbox = detection.bounding_box
        x1 = int(np.clip(bbox.origin_x, 0, w - 1))
        y1 = int(np.clip(bbox.origin_y, 0, h - 1))
        x2 = int(np.clip(bbox.origin_x + bbox.width,  x1 + 1, w))
        y2 = int(np.clip(bbox.origin_y + bbox.height, y1 + 1, h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for kp in (getattr(detection, "keypoints", None) or []):
            px, py = self._keypoint_to_pixel(kp, frame_w=w, frame_h=h)
            cv2.circle(vis, (int(np.clip(px, 0, w-1)), int(np.clip(py, 0, h-1))), 2, (0, 0, 255), -1)

        cats = getattr(detection, "categories", None) or []
        score = float(cats[0].score) if cats and hasattr(cats[0], "score") else 0.0
        cv2.putText(
            vis, f"face {score:.2f}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
        return vis


# ── Backward-compatible wrapper ───────────────────────────────────────────────

def align_and_crop(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    target_size: int = 256,
    crop_margin: float = 0.3,
) -> Optional[np.ndarray]:
    """Wrapper giữ tương thích ngược với code preprocess cũ."""
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detection = detector.detect(rgb)
    if detection is None:
        return None
    return detector.align_and_crop(
        bgr_frame=frame_bgr,
        detection=detection,
        target_size=(target_size, target_size),
        crop_margin=crop_margin,
    )
