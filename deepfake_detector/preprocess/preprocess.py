"""Main preprocessing orchestration script for deepfake dataset creation.

Cải tiến so với bản gốc:
  1. Frame retry: khi slot detect miss, thử frame kế cận trong cửa sổ ±window_frames.
  2. Adaptive sampling: dùng mode="uniform" thay vì "stride" để lấy frame đều hơn,
     tránh bỏ sót cụm frame có mặt rõ.
  3. Oversample: lấy samples_per_video * oversample_factor frame, chỉ giữ đủ slot.
     Tăng cơ hội detect đủ mặt mà không đọc toàn bộ video.
  4. Log chi tiết: ghi rõ bao nhiêu frame miss / retry success per video.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

try:
    from .face_detector import FaceDetector
    from .video_utils import list_video_files, sample_frame_indices, get_video_info
except ImportError:
    from preprocess.face_detector import FaceDetector
    from preprocess.video_utils import list_video_files, sample_frame_indices, get_video_info

_WORKER_DETECTOR: FaceDetector | None = None
_WORKER_DETECTOR_CONF: float | None = None

# Số frame kế cận thử lại mỗi phía khi miss (retry window)
DEFAULT_RETRY_WINDOW = 4
# Hệ số oversample: lấy samples * factor frame → tăng khả năng có đủ slot
DEFAULT_OVERSAMPLE = 2
DONE_META_FILENAME = ".done.json"
DONE_META_VERSION = 1


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw videos to face clips.")
    parser.add_argument("--input_dir",          type=str, required=True)
    parser.add_argument("--output_dir",         type=str, required=True)
    parser.add_argument("--label",              type=str, required=True, choices=["Real", "Fake"])
    parser.add_argument("--samples_per_video",  type=int, default=10)
    parser.add_argument("--target_size",        type=int, default=None)
    parser.add_argument("--min_face_confidence",type=float, default=None)
    parser.add_argument("--config",             type=str, required=True)
    parser.add_argument("--num_workers",        type=int, default=50)
    parser.add_argument(
        "--retry_window", type=int, default=DEFAULT_RETRY_WINDOW,
        help="So frame ke can thu lai moi phia khi miss (default 4).",
    )
    parser.add_argument(
        "--oversample_factor", type=int, default=DEFAULT_OVERSAMPLE,
        help="He so lay them frame de tang co hoi detect du mat (default 2).",
    )
    parser.add_argument(
        "--enable_prev_bbox_fallback",
        action="store_true",
        help="Neu miss detect thi thu crop 1 lan bang bbox cua slot truoc (co guard, mac dinh tat).",
    )
    return parser.parse_args()


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ── Config loader ─────────────────────────────────────────────────────────────

def load_preprocess_settings(config_path: str) -> Dict[str, float]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Khong tim thay config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {"crop_margin": 0.2, "target_size": 256.0, "min_face_confidence": 0.5}

    pre = cfg.get("preprocess", {}) if isinstance(cfg.get("preprocess"), dict) else {}
    trn = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    return {
        "crop_margin":          float(pre.get("crop_margin",          cfg.get("crop_margin",          trn.get("crop_margin",          0.2)))),
        "target_size":          float(pre.get("target_size",          cfg.get("target_size",          256))),
        "min_face_confidence":  float(pre.get("min_face_confidence",  cfg.get("min_face_confidence",  0.5))),
    }


# ── Resume check ──────────────────────────────────────────────────────────────

def _build_done_metadata(
    samples_per_video: int,
    target_size: int,
    crop_margin: float,
    min_face_confidence: float,
    retry_window: int,
    oversample_factor: int,
) -> Dict[str, Any]:
    return {
        "version": DONE_META_VERSION,
        "samples_per_video": int(samples_per_video),
        "target_size": int(target_size),
        "crop_margin": float(crop_margin),
        "min_face_confidence": float(min_face_confidence),
        "retry_window": int(retry_window),
        "oversample_factor": int(oversample_factor),
    }


def _expected_frame_names(video_stem: str, expected_frames: int) -> Set[str]:
    return {f"{video_stem}_f{i:04d}.jpg" for i in range(expected_frames)}


def _metadata_matches(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    for key, expected_val in expected.items():
        if key not in actual:
            return False
        actual_val = actual[key]
        if isinstance(expected_val, float):
            try:
                if abs(float(actual_val) - expected_val) > 1e-9:
                    return False
            except Exception:
                return False
        else:
            if actual_val != expected_val:
                return False
    return True


def has_completed_output(
    output_dir: Path,
    label: str,
    video_stem: str,
    expected_frames: int,
    expected_meta: Dict[str, Any],
) -> Tuple[bool, str]:
    video_out_dir = output_dir / label / video_stem
    if not video_out_dir.exists():
        return False, "missing output folder"

    expected_names = _expected_frame_names(video_stem, expected_frames)
    existing_names = {p.name for p in video_out_dir.glob(f"{video_stem}_f*.jpg")}
    missing = expected_names - existing_names
    if missing:
        return False, f"missing {len(missing)} expected frame(s)"

    done_file = video_out_dir / DONE_META_FILENAME
    if not done_file.exists():
        return False, "missing done metadata"

    try:
        meta = json.loads(done_file.read_text(encoding="utf-8"))
    except Exception:
        return False, "invalid done metadata"
    if not isinstance(meta, dict):
        return False, "invalid done metadata"
    if not _metadata_matches(meta, expected_meta):
        return False, "done metadata mismatch"
    return True, "complete output"


def has_enough_processed_frames(
    output_dir: Path, label: str, video_stem: str, expected_frames: int
) -> bool:
    """Backward-compatible wrapper: chi check so luong frame."""
    video_out_dir = output_dir / label / video_stem
    if not video_out_dir.exists():
        return False
    existing = list(video_out_dir.glob(f"{video_stem}_f*.jpg"))
    return len(existing) >= expected_frames


def _cleanup_stale_temp_dirs(label_dir: Path, stem: str) -> None:
    for tmp_dir in label_dir.glob(f".{stem}.tmp-*"):
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_done_metadata(out_dir: Path, meta: Dict[str, Any]) -> None:
    done_file = out_dir / DONE_META_FILENAME
    done_file.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


# ── Worker detector (per-process singleton) ───────────────────────────────────

def _get_worker_detector(min_face_confidence: float) -> FaceDetector:
    global _WORKER_DETECTOR, _WORKER_DETECTOR_CONF
    if (
        _WORKER_DETECTOR is None
        or _WORKER_DETECTOR_CONF is None
        or abs(float(min_face_confidence) - float(_WORKER_DETECTOR_CONF)) > 1e-12
    ):
        if _WORKER_DETECTOR is not None:
            _WORKER_DETECTOR.close()
        _WORKER_DETECTOR = FaceDetector(
            min_detection_confidence=float(min_face_confidence),
            # fallback confidence thấp hơn 0.2 so với primary
            fallback_confidence=max(0.1, float(min_face_confidence) - 0.2),
        )
        _WORKER_DETECTOR_CONF = float(min_face_confidence)
    return _WORKER_DETECTOR


# ── Frame-level crop với retry ────────────────────────────────────────────────

def _crop_with_retry(
    frame_pool: List[Tuple[int, Any]],   # list (frame_idx, frame_bgr) đã load
    slot_idx: int,                        # vị trí nominal trong frame_pool
    detector: FaceDetector,
    target_size: int,
    crop_margin: float,
    retry_window: int,
) -> Tuple[Optional[Any], bool, Optional[Tuple[int, int, int, int]]]:
    """
    Thử crop mặt ở slot_idx.
    Nếu miss → quét các frame trong cửa sổ ±retry_window quanh slot_idx.
    Trả về (face_image hoặc None, is_retry_success, bbox_detected hoặc None).
    """
    total = len(frame_pool)

    # Thứ tự thử: slot gốc trước, rồi mở rộng ra hai phía theo khoảng cách
    candidates = [slot_idx]
    for delta in range(1, retry_window + 1):
        if slot_idx - delta >= 0:
            candidates.append(slot_idx - delta)
        if slot_idx + delta < total:
            candidates.append(slot_idx + delta)

    for i, cand_idx in enumerate(candidates):
        _, frame_bgr = frame_pool[cand_idx]
        face, bbox = _detect_and_crop(
            frame_bgr=frame_bgr,
            detector=detector,
            target_size=target_size,
            crop_margin=crop_margin,
        )
        if face is not None:
            return face, (i > 0), bbox  # i>0 nghĩa là phải retry

    return None, False, None


def _detect_and_crop(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    target_size: int,
    crop_margin: float,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect + align/crop 1 frame.
    Trả về (face, bbox từ detection) nếu thành công.
    """
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None, None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detection = detector.detect(rgb)
    if detection is None:
        return None, None

    face = detector.align_and_crop(
        bgr_frame=frame_bgr,
        detection=detection,
        target_size=(target_size, target_size),
        crop_margin=crop_margin,
    )
    if face is None:
        return None, None

    bbox = detector.get_detection_bbox(detection)
    return face, bbox


# ── Single-video worker ───────────────────────────────────────────────────────

def _process_single_video(
    video_path: str,
    output_dir: str,
    label: str,
    samples_per_video: int,
    target_size: int,
    crop_margin: float,
    min_face_confidence: float,
    retry_window: int,
    oversample_factor: int,
    enable_prev_bbox_fallback: bool,
) -> Dict[str, Any]:
    """
    Worker xử lý 1 video với resume an toàn + ghi output atomic.
    """
    video = Path(video_path)
    out_root = Path(output_dir)
    stem = video.stem
    label_dir = out_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    final_out_dir = label_dir / stem

    done_meta = _build_done_metadata(
        samples_per_video=samples_per_video,
        target_size=target_size,
        crop_margin=crop_margin,
        min_face_confidence=min_face_confidence,
        retry_window=retry_window,
        oversample_factor=oversample_factor,
    )
    is_complete, skip_reason = has_completed_output(
        output_dir=out_root,
        label=label,
        video_stem=stem,
        expected_frames=samples_per_video,
        expected_meta=done_meta,
    )
    if is_complete:
        return {
            "video": video.name,
            "status": "skipped",
            "frames_saved": 0,
            "miss": 0,
            "retry_ok": 0,
            "prev_bbox_ok": 0,
            "message": f"Da hoan tat truoc do ({skip_reason}).",
        }

    _cleanup_stale_temp_dirs(label_dir, stem)
    temp_out_dir: Optional[Path] = None
    try:
        from preprocess.video_utils import _read_frames_with_indices

        info = get_video_info(video)
        total_frames = int(info.get("total_frames", 0))
        if total_frames <= 0:
            return {
                "video": video.name,
                "status": "error",
                "frames_saved": 0,
                "miss": 0,
                "retry_ok": 0,
                "prev_bbox_ok": 0,
                "message": "Khong doc duoc frame count.",
            }

        total_to_load = min(
            samples_per_video * max(1, oversample_factor),
            total_frames,
        )
        indices = sample_frame_indices(
            total_frames=total_frames,
            n_samples=total_to_load,
            mode="uniform",
        )
        frame_pool = _read_frames_with_indices(video, indices)
        if not frame_pool:
            return {
                "video": video.name,
                "status": "error",
                "frames_saved": 0,
                "miss": 0,
                "retry_ok": 0,
                "prev_bbox_ok": 0,
                "message": "Khong doc duoc frame nao.",
            }

        detector = _get_worker_detector(min_face_confidence)
        temp_out_dir = label_dir / f".{stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temp_out_dir.mkdir(parents=True, exist_ok=False)

        saved = 0
        miss_count = 0
        retry_ok = 0
        prev_bbox_ok = 0
        pool_size = len(frame_pool)
        last_good_bbox: Optional[Tuple[int, int, int, int]] = None
        allow_prev_bbox_fallback = True

        slot_indices = (
            np.linspace(0, pool_size - 1, num=samples_per_video).round().astype(int).tolist()
            if pool_size >= samples_per_video
            else list(range(pool_size))
        )

        for slot_pos, slot_idx in enumerate(slot_indices):
            face, is_retry, bbox = _crop_with_retry(
                frame_pool=frame_pool,
                slot_idx=slot_idx,
                detector=detector,
                target_size=target_size,
                crop_margin=crop_margin,
                retry_window=retry_window,
            )

            if face is None:
                miss_count += 1
                used_prev_bbox = False
                if (
                    enable_prev_bbox_fallback
                    and allow_prev_bbox_fallback
                    and last_good_bbox is not None
                ):
                    _, frame_bgr = frame_pool[slot_idx]
                    face = detector.crop_from_bbox(
                        bgr_frame=frame_bgr,
                        bbox=last_good_bbox,
                        target_size=(target_size, target_size),
                        crop_margin=crop_margin,
                    )
                    if face is not None:
                        prev_bbox_ok += 1
                        used_prev_bbox = True
                        allow_prev_bbox_fallback = False
                if face is None:
                    continue
                if not used_prev_bbox:
                    allow_prev_bbox_fallback = True
            else:
                if is_retry:
                    retry_ok += 1
                if bbox is not None and bbox != (0, 0, 0, 0):
                    last_good_bbox = bbox
                allow_prev_bbox_fallback = True

            out_path = temp_out_dir / f"{stem}_f{slot_pos:04d}.jpg"
            if cv2.imwrite(str(out_path), face):
                saved += 1

        if saved >= samples_per_video:
            _write_done_metadata(temp_out_dir, done_meta)
            if final_out_dir.exists():
                shutil.rmtree(final_out_dir, ignore_errors=True)
            temp_out_dir.replace(final_out_dir)
            temp_out_dir = None
            return {
                "video": video.name,
                "status": "success",
                "frames_saved": saved,
                "miss": miss_count,
                "retry_ok": retry_ok,
                "prev_bbox_ok": prev_bbox_ok,
                "message": (
                    f"OK (miss={miss_count}, retry_ok={retry_ok}, "
                    f"prev_bbox_ok={prev_bbox_ok})"
                ),
            }

        return {
            "video": video.name,
            "status": "error",
            "frames_saved": saved,
            "miss": miss_count,
            "retry_ok": retry_ok,
            "prev_bbox_ok": prev_bbox_ok,
            "message": (
                f"Khong du mat: saved={saved}/{samples_per_video} "
                f"(miss={miss_count}, retry_ok={retry_ok}, prev_bbox_ok={prev_bbox_ok})"
            ),
        }
    finally:
        if temp_out_dir is not None and temp_out_dir.exists():
            shutil.rmtree(temp_out_dir, ignore_errors=True)


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_preprocess_pipeline(
    input_dir: str,
    output_dir: str,
    label: str,
    samples_per_video: int,
    target_size: int,
    crop_margin: float,
    logger: logging.Logger,
    num_workers: int = 50,
    min_face_confidence: float = 0.5,
    retry_window: int = DEFAULT_RETRY_WINDOW,
    oversample_factor: int = DEFAULT_OVERSAMPLE,
    enable_prev_bbox_fallback: bool = False,
) -> Dict[str, int]:
    in_dir  = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / label).mkdir(parents=True, exist_ok=True)

    video_paths = list_video_files(in_dir)
    if not video_paths:
        logger.warning("Khong tim thay video trong %s", in_dir)
        return {"success": 0, "error": 0, "skipped": 0, "saved_frames": 0}

    logger.info("Tim thay %d video. Label=%s.", len(video_paths), label)

    expected_meta = _build_done_metadata(
        samples_per_video=samples_per_video,
        target_size=target_size,
        crop_margin=crop_margin,
        min_face_confidence=min_face_confidence,
        retry_window=retry_window,
        oversample_factor=oversample_factor,
    )

    to_process: List[Path] = []
    skipped = 0
    for vp in video_paths:
        is_complete, reason = has_completed_output(
            output_dir=out_dir,
            label=label,
            video_stem=vp.stem,
            expected_frames=samples_per_video,
            expected_meta=expected_meta,
        )
        if is_complete:
            skipped += 1
            logger.info("[SKIP] %s | %s", vp.name, reason)
        else:
            to_process.append(vp)

    if not to_process:
        logger.info("Tat ca video da du frame.")
        return {"success": 0, "error": 0, "skipped": skipped, "saved_frames": 0}

    cpu_count   = os.cpu_count() or 1
    max_workers = min(len(to_process), num_workers, cpu_count)
    logger.info(
        "Workers: requested=%d cpu=%d effective=%d | retry_window=%d oversample=%d prev_bbox_fallback=%s",
        num_workers, cpu_count, max_workers, retry_window, oversample_factor, enable_prev_bbox_fallback,
    )

    success = error = saved_frames = 0
    total_miss = total_retry_ok = total_prev_bbox_ok = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_single_video,
                str(vp), str(out_dir), label,
                samples_per_video, target_size, crop_margin,
                min_face_confidence, retry_window, oversample_factor, enable_prev_bbox_fallback,
            )
            for vp in to_process
        ]
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as exc:
                error += 1
                logger.exception("[ERROR] Worker: %s", exc)
                continue

            total_miss += int(r.get("miss", 0))
            total_retry_ok += int(r.get("retry_ok", 0))
            total_prev_bbox_ok += int(r.get("prev_bbox_ok", 0))

            if r["status"] == "success":
                success      += 1
                saved_frames += int(r["frames_saved"])
                logger.info("[OK]   %s | %s", r["video"], r["message"])
            elif r["status"] == "skipped":
                skipped += 1
                logger.info("[SKIP] %s | %s", r["video"], r["message"])
            else:
                error += 1
                logger.warning("[FAIL] %s | %s", r["video"], r["message"])

    summary = {
        "success": success,
        "error": error,
        "skipped": skipped,
        "saved_frames": saved_frames,
        "miss": total_miss,
        "retry_ok": total_retry_ok,
        "prev_bbox_ok": total_prev_bbox_ok,
    }
    logger.info(
        "Summary | success=%d error=%d skipped=%d total_frames=%d miss=%d retry_ok=%d prev_bbox_ok=%d",
        success, error, skipped, saved_frames, total_miss, total_retry_ok, total_prev_bbox_ok,
    )
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logger = setup_logger(Path(args.output_dir) / "preprocess.log")

    settings           = load_preprocess_settings(args.config)
    crop_margin        = float(settings["crop_margin"])
    target_size        = int(args.target_size) if args.target_size is not None else int(settings["target_size"])
    min_face_confidence = (
        float(args.min_face_confidence) if args.min_face_confidence is not None
        else float(settings["min_face_confidence"])
    )

    if not (0.0 <= min_face_confidence <= 1.0):
        raise ValueError("min_face_confidence phai trong [0, 1].")

    logger.info(
        "Config | input=%s label=%s samples=%d size=%d margin=%.3f conf=%.2f "
        "workers=%d retry=%d oversample=%d prev_bbox_fallback=%s",
        args.input_dir, args.label, args.samples_per_video, target_size,
        crop_margin, min_face_confidence, args.num_workers,
        args.retry_window, args.oversample_factor, args.enable_prev_bbox_fallback,
    )

    run_preprocess_pipeline(
        input_dir           = args.input_dir,
        output_dir          = args.output_dir,
        label               = args.label,
        samples_per_video   = args.samples_per_video,
        target_size         = target_size,
        crop_margin         = crop_margin,
        min_face_confidence = min_face_confidence,
        num_workers         = args.num_workers,
        logger              = logger,
        retry_window        = args.retry_window,
        oversample_factor   = args.oversample_factor,
        enable_prev_bbox_fallback = args.enable_prev_bbox_fallback,
    )


if __name__ == "__main__":
    main()
