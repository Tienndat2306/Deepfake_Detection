"""Clean corrupted or low-quality video folders in processed deepfake dataset.

Scan OUTPUT_DIR/{Real|Fake}/{video_stem}/ and flag folders for deletion when:
  Tier 1 - Empty folder   : no image files.
  Tier 2 - Too few frames : total frames < min_frames.
  Tier 3 - Black frame    : mean pixel < threshold.
  Tier 4 - Uniform frame  : std pixel < std_threshold.
  Tier 5 - Blurry/NoFace  : Laplacian variance < blur_threshold.
  Tier 6 - Face confidence: detect miss or score < face_conf_threshold (optional).

If bad frame ratio in a folder > bad_ratio, delete that whole folder.

OPTIMIZED for 50-core CPU:
  - Frame-level scanning parallelized with ProcessPoolExecutor (CPU-bound)
  - Folder-level audit batched across workers
  - Deletion parallelized with ThreadPoolExecutor (I/O-bound)
  - Resume support via checkpoint file

Usage:
  python clean_processed_dataset.py
  python clean_processed_dataset.py --delete
  python clean_processed_dataset.py --output-dir D:/processed_faces
  python clean_processed_dataset.py --delete --blur-threshold 100
  python clean_processed_dataset.py --workers 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR    = "./processed_faces"
BLACK_MEAN_THRESHOLD  = 10.0
STD_THRESHOLD         = 5.0
# [FIX B2] Dua nguong blur mac dinh ve 8.0 theo thong nhat pipeline.
BLUR_THRESHOLD        = 8.0
MIN_FRAMES            = 3
BAD_FRAME_RATIO       = 0.5
DEFAULT_WORKERS       = min(50, (os.cpu_count() or 1))
SUPPORTED_EXTS        = {".jpg", ".jpeg", ".png"}
CHECKPOINT_FILE       = ".clean_checkpoint.json"
CHECKPOINT_VERSION    = 2
DEFAULT_FACE_CONF_THRESHOLD = -1.0
DEFAULT_MAX_FRAMES_PER_VIDEO = 10

_FACE_DETECTOR_CLASS = None
_WORKER_FACE_DETECTOR = None
_WORKER_FACE_DETECTOR_KEY: float | None = None

# ── Frame-level helpers (must be top-level for pickling) ──────────────────────

def _iter_frames(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _resolve_face_conf_threshold(raw_value: float | None) -> float | None:
    """Gia tri < 0 hoac None se tat Tier 6."""
    if raw_value is None:
        return None
    v = float(raw_value)
    if v < 0.0:
        return None
    return v


def _import_face_detector_class():
    """Lazy import FaceDetector de khong bat buoc mediapipe khi Tier 6 tat."""
    global _FACE_DETECTOR_CLASS
    if _FACE_DETECTOR_CLASS is not None:
        return _FACE_DETECTOR_CLASS
    from preprocess.face_detector import FaceDetector  # pylint: disable=import-outside-toplevel
    _FACE_DETECTOR_CLASS = FaceDetector
    return _FACE_DETECTOR_CLASS


def _get_worker_face_detector(face_conf_threshold: float):
    """Khoi tao 1 detector/worker process, tai su dung cho moi frame."""
    global _WORKER_FACE_DETECTOR, _WORKER_FACE_DETECTOR_KEY
    if (
        _WORKER_FACE_DETECTOR is None
        or _WORKER_FACE_DETECTOR_KEY is None
        or abs(float(face_conf_threshold) - float(_WORKER_FACE_DETECTOR_KEY)) > 1e-12
    ):
        if _WORKER_FACE_DETECTOR is not None:
            try:
                _WORKER_FACE_DETECTOR.close()
            except Exception:
                pass
        detector_cls = _import_face_detector_class()
        # Dat min_detection_confidence thap de co the phan loai "low-face-confidence".
        primary_conf = min(0.3, max(0.1, float(face_conf_threshold)))
        _WORKER_FACE_DETECTOR = detector_cls(
            min_detection_confidence=primary_conf,
            fallback_confidence=0.1,
        )
        _WORKER_FACE_DETECTOR_KEY = float(face_conf_threshold)
    return _WORKER_FACE_DETECTOR


def _build_scan_signature(
    mean_thr: float,
    std_thr: float,
    blur_thr: float,
    min_frames: int,
    bad_ratio: float,
    face_conf_threshold: float | None,
) -> str:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "mean_thr": float(mean_thr),
        "std_thr": float(std_thr),
        "blur_thr": float(blur_thr),
        "min_frames": int(min_frames),
        "bad_ratio": float(bad_ratio),
        "face_conf_threshold": None if face_conf_threshold is None else float(face_conf_threshold),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_face_detector_runtime(face_conf_threshold: float) -> tuple[bool, str]:
    """
    Preflight khi bat Tier 6:
    - Xac minh import + khoi tao FaceDetector.
    - Neu loi (vd mediapipe chua cai), tra ve message ro rang.
    """
    try:
        detector_cls = _import_face_detector_class()
        probe = detector_cls(
            min_detection_confidence=min(0.3, max(0.1, float(face_conf_threshold))),
            fallback_confidence=0.1,
        )
        probe.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _check_single_frame(args: tuple) -> tuple[bool, str]:
    """Worker function: check one frame. Returns (is_bad, reason).
    Must be top-level for ProcessPoolExecutor pickling.
    """
    img_path_str, mean_thr, std_thr, blur_thr, face_conf_threshold = args
    img = cv2.imread(img_path_str)
    if img is None:
        return True, "unreadable"

    arr = img.astype(np.float32, copy=False)
    mean_val = float(arr.mean())
    std_val  = float(arr.std())

    if mean_val < mean_thr:
        return True, f"black (mean={mean_val:.1f})"
    if std_val < std_thr:
        return True, f"uniform (std={std_val:.1f})"

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = _laplacian_variance(gray)
    if lap_var < blur_thr:
        return True, f"blurry/no-face (lap_var={lap_var:.1f})"

    # Tier 6 (optional): face confidence
    if face_conf_threshold is not None:
        detector = _get_worker_face_detector(float(face_conf_threshold))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection = detector.detect(rgb)
        if detection is None:
            return True, "no-face-detect"
        score = detector.get_detection_score(detection)
        if score < float(face_conf_threshold):
            return True, f"low-face-confidence (score={score:.2f})"

    return False, "ok"


def _audit_folder_worker(args: tuple) -> tuple[str, str, bool, str, dict]:
    """Audit one folder synchronously (runs inside a worker process).
    Returns (label, folder_str, should_delete, reason, stats).
    """
    (
        label,
        folder_str,
        mean_thr,
        std_thr,
        blur_thr,
        min_frames,
        bad_ratio,
        face_conf_threshold,
    ) = args
    folder = Path(folder_str)

    frame_files = _iter_frames(folder)
    total = len(frame_files)
    stats: dict = {"total": total, "bad": 0, "reasons": {}}

    if total == 0:
        return label, folder_str, True, "empty folder (no image files)", stats
    if total < min_frames:
        return label, folder_str, True, f"too few frames ({total} < {min_frames})", stats

    # Check all frames in THIS process (no sub-pool to avoid nested spawn)
    bad_count = 0
    reason_counter: Counter[str] = Counter()
    for frame in frame_files:
        bad, reason = _check_single_frame(
            (str(frame), mean_thr, std_thr, blur_thr, face_conf_threshold)
        )
        if bad:
            bad_count += 1
            category = reason.split("(")[0].strip()
            reason_counter[category] += 1

    stats["bad"]     = bad_count
    stats["reasons"] = dict(reason_counter)
    ratio = bad_count / total

    if ratio > bad_ratio:
        top = ", ".join(f"{r} x{c}" for r, c in reason_counter.most_common(3))
        return (
            label, folder_str, True,
            f"bad frames {bad_count}/{total} ({ratio:.0%}) [{top}]",
            stats,
        )

    return label, folder_str, False, "ok", stats


# ── Folder collection ─────────────────────────────────────────────────────────

def collect_video_folders(output_dir: Path) -> list[tuple[str, Path]]:
    folders: list[tuple[str, Path]] = []
    for label in ("Real", "Fake"):
        label_dir = output_dir / label
        if not label_dir.exists():
            continue
        for child in sorted(label_dir.iterdir()):
            if child.is_dir():
                folders.append((label, child))
    return folders


def collect_excess_frames(
    folders: list[tuple[str, Path]],
    max_frames_per_video: int,
) -> list[tuple[str, Path, int, list[Path]]]:
    """
    Return list (label, folder, total_frames, extra_frames_to_delete)
    cho cac folder co so frame > max_frames_per_video.
    """
    items: list[tuple[str, Path, int, list[Path]]] = []
    max_keep = int(max_frames_per_video)
    for label, folder in folders:
        frame_files = _iter_frames(folder)
        total = len(frame_files)
        if total <= max_keep:
            continue
        extras = frame_files[max_keep:]
        items.append((label, folder, total, extras))
    return items


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint(output_dir: Path, signature: str) -> tuple[set[str], bool]:
    """
    Return (scanned_ok, signature_mismatch).
    signature_mismatch=True khi checkpoint ton tai nhung tham so scan da doi.
    """
    cp = output_dir / CHECKPOINT_FILE
    if cp.exists():
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return set(), False
            old_sig = str(data.get("signature", ""))
            if not old_sig:
                return set(), True
            if old_sig != signature:
                return set(), True
            return set(data.get("scanned_ok", [])), False
        except Exception:
            pass
    return set(), False


def _save_checkpoint(output_dir: Path, scanned_ok: set[str], signature: str) -> None:
    cp = output_dir / CHECKPOINT_FILE
    try:
        payload = {
            "version": CHECKPOINT_VERSION,
            "signature": signature,
            "scanned_ok": sorted(scanned_ok),
        }
        cp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clear_checkpoint(output_dir: Path) -> None:
    cp = output_dir / CHECKPOINT_FILE
    if cp.exists():
        cp.unlink(missing_ok=True)


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(items: list[tuple[str, Path, str]]) -> None:
    all_reasons: Counter[str] = Counter()
    for _, _, reason in items:
        key = reason.split("(")[0].split("[")[0].strip()
        all_reasons[key] += 1

    print("  Phan loai loi:")
    for reason_type, count in all_reasons.most_common():
        print(f"    {reason_type:<40s}: {count} folders")
    print()

    grouped: dict[str, list[tuple[Path, str]]] = {}
    for label, folder, reason in items:
        grouped.setdefault(label, []).append((folder, reason))

    for label in ("Real", "Fake"):
        rows = grouped.get(label, [])
        if not rows:
            continue
        print(f"  [{label}] {len(rows)} folder se bi xoa:")
        for folder, reason in rows[:25]:
            print(f"    {folder.name:44s} | {reason}")
        if len(rows) > 25:
            print(f"    ... va {len(rows) - 25} folder khac")
    print()


# ── Parallel scan ─────────────────────────────────────────────────────────────

def _print_trim_report(items: list[tuple[str, Path, int, list[Path]]], max_keep: int) -> None:
    total_files = sum(len(extra_files) for _, _, _, extra_files in items)
    print(f"  Video co frame du (> {max_keep}): {len(items)}")
    print(f"  Tong frame du can xoa          : {total_files}")
    print()

    grouped: dict[str, list[tuple[Path, int, int]]] = {}
    for label, folder, total, extra_files in items:
        grouped.setdefault(label, []).append((folder, total, len(extra_files)))

    for label in ("Real", "Fake"):
        rows = grouped.get(label, [])
        if not rows:
            continue
        print(f"  [{label}] {len(rows)} video co frame du:")
        for folder, total, extra_count in rows[:25]:
            print(
                f"    {folder.name:44s} | total={total:4d} "
                f"giu={max_keep:3d} xoa={extra_count:4d}"
            )
        if len(rows) > 25:
            print(f"    ... va {len(rows) - 25} video khac")
    print()


def parallel_scan(
    folders: list[tuple[str, Path]],
    mean_thr: float,
    std_thr: float,
    blur_thr: float,
    min_frames: int,
    bad_ratio: float,
    face_conf_threshold: float | None,
    workers: int,
    output_dir: Path,
    use_checkpoint: bool = True,
) -> list[tuple[str, Path, str]]:
    """Return list of (label, folder, reason) that should be deleted."""

    scan_signature = _build_scan_signature(
        mean_thr=mean_thr,
        std_thr=std_thr,
        blur_thr=blur_thr,
        min_frames=min_frames,
        bad_ratio=bad_ratio,
        face_conf_threshold=face_conf_threshold,
    )
    scanned_ok, signature_mismatch = (
        _load_checkpoint(output_dir, signature=scan_signature)
        if use_checkpoint
        else (set(), False)
    )
    if signature_mismatch:
        print("  [Resume] Checkpoint cu khong khop tham so scan, bo qua cache va quet lai.\n")

    skipped = sum(1 for _, f in folders if str(f) in scanned_ok)
    if skipped:
        print(f"  [Resume] Bo qua {skipped} folders da quet OK truoc do.\n")

    pending = [
        (
            label,
            str(folder),
            mean_thr,
            std_thr,
            blur_thr,
            min_frames,
            bad_ratio,
            face_conf_threshold,
        )
        for label, folder in folders
        if str(folder) not in scanned_ok
    ]

    to_delete: list[tuple[str, Path, str]] = []
    newly_ok: set[str] = set()

    # Each worker audits one complete folder (CPU-bound → ProcessPoolExecutor)
    effective_workers = min(workers, len(pending)) if pending else 1

    with ProcessPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(_audit_folder_worker, args): args for args in pending}

        with tqdm(total=len(pending), desc="Scanning", unit="video", ncols=78) as bar:
            for future in as_completed(futures):
                bar.update(1)
                try:
                    label, folder_str, should_delete, reason, _stats = future.result()
                    if should_delete:
                        to_delete.append((label, Path(folder_str), reason))
                    else:
                        newly_ok.add(folder_str)

                    # Periodically save checkpoint
                    if len(newly_ok) > 0 and (len(newly_ok) % 500 == 0) and use_checkpoint:
                        _save_checkpoint(output_dir, scanned_ok | newly_ok, signature=scan_signature)

                except Exception as exc:
                    args = futures[future]
                    print(f"\n  [WARN] Loi quet {Path(args[1]).name}: {exc}")
                    bar.update(0)

    if use_checkpoint:
        _save_checkpoint(output_dir, scanned_ok | newly_ok, signature=scan_signature)

    return to_delete


# ── Parallel delete ───────────────────────────────────────────────────────────

def parallel_delete(
    to_delete: list[tuple[str, Path, str]],
    workers: int,
) -> tuple[int, int]:
    """Delete folders in parallel using threads (I/O-bound). Returns (deleted, failed)."""
    deleted = 0
    failed  = 0

    def _remove(folder: Path) -> tuple[Path, Exception | None]:
        try:
            shutil.rmtree(folder)
            return folder, None
        except Exception as exc:
            return folder, exc

    thread_workers = min(workers, len(to_delete))
    with ThreadPoolExecutor(max_workers=thread_workers) as pool:
        futures = {pool.submit(_remove, folder): folder for _, folder, _ in to_delete}
        with tqdm(total=len(to_delete), desc="Deleting", unit="folder", ncols=78) as bar:
            for future in as_completed(futures):
                folder, exc = future.result()
                if exc is None:
                    deleted += 1
                else:
                    failed += 1
                    print(f"\n  [WARN] Loi xoa {folder.name}: {exc}")
                bar.update(1)

    return deleted, failed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parallel_delete_files(
    file_paths: list[Path],
    workers: int,
) -> tuple[int, int]:
    """Delete file paths in parallel (I/O-bound). Returns (deleted, failed)."""
    deleted = 0
    failed = 0
    if not file_paths:
        return deleted, failed

    def _remove_file(path: Path) -> tuple[Path, Exception | None]:
        try:
            path.unlink(missing_ok=True)
            return path, None
        except Exception as exc:
            return path, exc

    thread_workers = max(1, min(int(workers), len(file_paths)))
    with ThreadPoolExecutor(max_workers=thread_workers) as pool:
        futures = {pool.submit(_remove_file, path): path for path in file_paths}
        with tqdm(total=len(file_paths), desc="Deleting", unit="frame", ncols=78) as bar:
            for future in as_completed(futures):
                path, exc = future.result()
                if exc is None:
                    deleted += 1
                else:
                    failed += 1
                    print(f"\n  [WARN] Loi xoa frame {path.name}: {exc}")
                bar.update(1)

    return deleted, failed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean processed deepfake dataset (5-tier + optional Tier 6 face confidence).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delete",         action="store_true",
                        help="Xoa that su (mac dinh: dry-run).")
    parser.add_argument("--workers",        type=int,  default=DEFAULT_WORKERS,
                        help="So worker process (default = so CPU).")
    parser.add_argument("--threshold",      type=float, default=BLACK_MEAN_THRESHOLD,
                        help="Mean threshold cho black frame.")
    parser.add_argument("--std-threshold",  type=float, default=STD_THRESHOLD,
                        help="Std threshold cho uniform frame.")
    parser.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD,
                        help="Laplacian var threshold cho blurry/no-face.")
    parser.add_argument(
        "--face-conf-threshold",
        type=float,
        default=DEFAULT_FACE_CONF_THRESHOLD,
        help="Tier 6: score mat toi thieu. Dat < 0 de tat Tier 6.",
    )
    parser.add_argument("--min-frames",     type=int,   default=MIN_FRAMES,
                        help="So frame toi thieu moi folder.")
    parser.add_argument("--bad-ratio",      type=float, default=BAD_FRAME_RATIO,
                        help="Ty le frame loi toi da truoc khi xoa folder.")
    parser.add_argument("--no-checkpoint",  action="store_true",
                        help="Tat tinh nang resume checkpoint.")
    parser.add_argument("--clear-checkpoint", action="store_true",
                        help="Xoa checkpoint cu va quet lai tu dau.")
    parser.add_argument(
        "--trim-excess-only",
        action="store_true",
        help="Chi xoa frame du (> max-frames-per-video), bo qua toan bo tier scan.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=DEFAULT_MAX_FRAMES_PER_VIDEO,
        help="So frame toi da giu lai moi video khi dung --trim-excess-only.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    face_conf_threshold = _resolve_face_conf_threshold(args.face_conf_threshold)

    # Validate
    if args.max_frames_per_video < 1:
        parser.error("--max-frames-per-video phai >= 1")
    if args.min_frames < 1:
        parser.error("--min-frames phai >= 1")
    if not (0.0 <= args.bad_ratio <= 1.0):
        parser.error("--bad-ratio phai nam trong [0, 1]")
    if min(args.threshold, args.std_threshold, args.blur_threshold) < 0:
        parser.error("Cac threshold phai >= 0")
    if face_conf_threshold is not None and not (0.0 <= face_conf_threshold <= 1.0):
        parser.error("--face-conf-threshold phai nam trong [0, 1] hoac < 0 de tat.")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[ERROR] OUTPUT_DIR khong ton tai: {output_dir}")
        sys.exit(1)

    if (not args.trim_excess_only) and face_conf_threshold is not None:
        ok, err = _validate_face_detector_runtime(face_conf_threshold)
        if not ok:
            print("[ERROR] Khong the bat Tier 6 face confidence.")
            print("        Vui long cai mediapipe va dependencies truoc.")
            print("        Goi y: pip install mediapipe")
            print(f"        Chi tiet: {err}")
            sys.exit(1)

    if args.clear_checkpoint:
        _clear_checkpoint(output_dir)
        print("  [INFO] Da xoa checkpoint cu.\n")

    if args.trim_excess_only:
        print("=" * 70)
        print("  DEEPFAKE DATASET TRIMMER (chi xoa frame du)")
        print("=" * 70)
        print(f"  Dir         : {output_dir}")
        print(f"  Workers     : {args.workers} threads")
        print(f"  Keep/video  : {args.max_frames_per_video} frames")
        print(
            f"  Mode        : "
            f"{'>>> DELETE (THAT SU) <<<' if args.delete else 'DRY-RUN (them --delete de xoa that)'}"
        )
        print()

        folders = collect_video_folders(output_dir)
        if not folders:
            print("[WARN] Khong tim thay video folder nao.")
            sys.exit(0)

        real_total = sum(1 for lbl, _ in folders if lbl == "Real")
        fake_total = sum(1 for lbl, _ in folders if lbl == "Fake")
        print(f"  Tim thay {len(folders)} folders  (Real={real_total}, Fake={fake_total})\n")

        to_trim = collect_excess_frames(
            folders=folders,
            max_frames_per_video=args.max_frames_per_video,
        )
        print(f"{'=' * 70}")
        print("  KET QUA QUET FRAME DU")
        print(f"{'=' * 70}")
        print(f"  Tong so folders       : {len(folders)}")
        print(f"  Can trim (> max)      : {len(to_trim)}")
        print(f"  Da dung <= max frames : {len(folders) - len(to_trim)}")
        print()

        if to_trim:
            _print_trim_report(to_trim, max_keep=args.max_frames_per_video)

        if args.delete and to_trim:
            file_paths = [p for _, _, _, extra_files in to_trim for p in extra_files]
            print(f"[ACTION] Bat dau xoa {len(file_paths)} frame du voi {args.workers} threads...")
            deleted, failed = parallel_delete_files(file_paths=file_paths, workers=args.workers)
            print(f"\n  Ket qua xoa frame du: thanh cong={deleted}, that bai={failed}")
        elif not args.delete:
            if to_trim:
                print("  -> Day la dry-run. Them --delete de xoa frame du that su.")
            else:
                print("  [OK] Khong co frame du can xoa.")

        print()
        return

    print("=" * 70)
    title = "6-tier" if face_conf_threshold is not None else "5-tier"
    print(f"  DEEPFAKE DATASET CLEANER  ({title} | 50-core optimized)")
    print("=" * 70)
    print(f"  Dir     : {output_dir}")
    print(f"  Workers : {args.workers} processes")
    print(f"  Mode    : {'>>> DELETE (THAT SU) <<<' if args.delete else 'DRY-RUN (them --delete de xoa that)'}")
    print()
    print("  Tier 1  Empty folder     khong co anh")
    print(f"  Tier 2  Too few frames   < {args.min_frames} frames")
    print(f"  Tier 3  Black frame      mean < {args.threshold}")
    print(f"  Tier 4  Uniform frame    std  < {args.std_threshold}")
    print(f"  Tier 5  Blurry/NoFace    Laplacian var < {args.blur_threshold}")
    if face_conf_threshold is not None:
        print(f"  Tier 6  Face confidence  score < {face_conf_threshold}")
    print(f"  Xoa folder neu ty le frame loi > {args.bad_ratio:.0%}")
    print()

    folders = collect_video_folders(output_dir)
    if not folders:
        print("[WARN] Khong tim thay video folder nao.")
        sys.exit(0)

    real_total = sum(1 for lbl, _ in folders if lbl == "Real")
    fake_total = sum(1 for lbl, _ in folders if lbl == "Fake")
    print(f"  Tim thay {len(folders)} folders  (Real={real_total}, Fake={fake_total})\n")

    # ── SCAN ──────────────────────────────────────────────────────────────────
    to_delete = parallel_scan(
        folders      = folders,
        mean_thr     = args.threshold,
        std_thr      = args.std_threshold,
        blur_thr     = args.blur_threshold,
        min_frames   = args.min_frames,
        bad_ratio    = args.bad_ratio,
        face_conf_threshold = face_conf_threshold,
        workers      = args.workers,
        output_dir   = output_dir,
        use_checkpoint = not args.no_checkpoint,
    )

    ok_count = len(folders) - len(to_delete)
    print(f"\n{'=' * 70}")
    print("  KET QUA QUET")
    print(f"{'=' * 70}")
    print(f"  Tong so folders : {len(folders)}")
    print(f"  Hop le (OK)     : {ok_count}")
    print(f"  Can xoa (loi)   : {len(to_delete)}")
    print()

    if to_delete:
        _print_report(to_delete)

    # ── DELETE ─────────────────────────────────────────────────────────────────
    if args.delete and to_delete:
        print(f"[ACTION] Bat dau xoa {len(to_delete)} folders voi {args.workers} threads...")
        deleted, failed = parallel_delete(to_delete, workers=args.workers)
        print(f"\n  Ket qua xoa: thanh cong={deleted}, that bai={failed}")

        # Clear checkpoint after successful delete
        _clear_checkpoint(output_dir)

        remaining   = collect_video_folders(output_dir)
        real_count  = sum(1 for lbl, _ in remaining if lbl == "Real")
        fake_count  = sum(1 for lbl, _ in remaining if lbl == "Fake")
        imbalance   = max(real_count, fake_count) / max(min(real_count, fake_count), 1)

        print("\n  Dataset sau khi lam sach:")
        print(f"    Real  : {real_count:>6} videos")
        print(f"    Fake  : {fake_count:>6} videos")
        print(f"    Total : {real_count + fake_count:>6} videos")
        print(f"    Imbalance (Fake/Real): {imbalance:.2f}x")
        if imbalance > 3.0:
            print("\n  [WARN] Imbalance > 3x. Nen dung WeightedRandomSampler khi train.")

    elif not args.delete:
        if to_delete:
            print("  -> Day la dry-run. Them --delete de xoa that su.")
        else:
            print("  [OK] Dataset sach! Khong co folder nao can xoa.")

    print()


if __name__ == "__main__":
    main()
