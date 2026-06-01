"""Utilities xu ly video bang OpenCV cho preprocessing."""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mp4v", ".avi", ".mov", ".mkv", ".webm"}
FRAME_COUNT_SEEK_MAX = 1_000_000
FRAME_COUNT_SEQUENTIAL_LIMIT = 60_000


def list_video_files(root_dir: Path) -> List[Path]:
    """List tat ca file video duoi root_dir (de quy)."""
    return sorted(
        [
            p
            for p in root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
    )


def _can_read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> bool:
    """Thu seek/read tai frame_idx. False neu khong doc duoc frame hop le."""
    if frame_idx < 0:
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return False

    pos_after = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # Neu cap bi clamp ve qua xa truoc frame_idx, coi nhu seek khong dang tin.
    if pos_after > 0 and (pos_after + 5.0) < float(frame_idx):
        return False
    return True


def _probe_reported_frame_count(cap: cv2.VideoCapture, reported_count: int) -> bool:
    """
    Kiem tra nhanh metadata frame_count bang cac moc dau-giua-cuoi.
    Trả về True nếu reported_count co ve dang tin.
    """
    if reported_count <= 0:
        return False

    candidates = sorted(
        {
            0,
            max(0, reported_count // 4),
            max(0, reported_count // 2),
            max(0, (reported_count * 3) // 4),
            max(0, reported_count - 1),
        }
    )
    if not candidates:
        return False

    hits = 0
    for idx in candidates:
        if _can_read_frame_at(cap, idx):
            hits += 1

    # Yêu cầu đọc được mốc cuối và phần lớn mốc trung gian.
    return (_can_read_frame_at(cap, max(0, reported_count - 1))) and (hits >= max(3, len(candidates) - 1))


def _estimate_frame_count_by_seek(
    cap: cv2.VideoCapture,
    max_search: int = FRAME_COUNT_SEEK_MAX,
) -> int:
    """
    Uoc luong frame count bang seek/read:
    - Exponential search tim upper bound.
    - Binary search tim frame index cuoi cung doc duoc.
    """
    if not _can_read_frame_at(cap, 0):
        return 0

    low = 0
    high = 1
    while high <= max_search and _can_read_frame_at(cap, high):
        low = high
        high *= 2

    if high > max_search:
        high = max_search
        if _can_read_frame_at(cap, high):
            return high + 1

    left = low + 1
    right = max(low + 1, high - 1)
    best = low
    while left <= right:
        mid = (left + right) // 2
        if _can_read_frame_at(cap, mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best + 1


def _count_frames_sequential(
    cap: cv2.VideoCapture,
    max_frames: int = FRAME_COUNT_SEQUENTIAL_LIMIT,
) -> Tuple[int, bool]:
    """
    Dem tuan tu frame (fallback cuoi cung).
    Returns: (count, reached_limit).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        count += 1
    return count, (count >= max_frames)


def get_video_info(video_path: str | Path) -> Dict[str, float | int]:
    """
    Tra ve thong tin co ban cua video:
    - total_frames: frame count da validate/fallback
    - reported_frame_count: frame count tu metadata goc CAP_PROP_FRAME_COUNT
    - frame_count_source: metadata | seek_probe | sequential | sequential_limited | metadata_unverified | unavailable
    - fps, width, height, duration_seconds
    """
    path = Path(video_path)
    info: Dict[str, float | int] = {
        "total_frames": 0,
        "reported_frame_count": 0,
        "frame_count_source": "unknown",
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "duration_seconds": 0.0,
    }
    if not path.exists():
        return info

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return info

    total_frames = 0
    reported_frame_count = 0
    frame_count_source = "unknown"
    fps = 0.0
    width = 0
    height = 0

    try:
        reported_frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if reported_frame_count > 0 and _probe_reported_frame_count(cap, reported_frame_count):
            total_frames = reported_frame_count
            frame_count_source = "metadata"
        else:
            estimated_by_seek = _estimate_frame_count_by_seek(cap)
            if estimated_by_seek > 0:
                total_frames = int(estimated_by_seek)
                frame_count_source = "seek_probe"
            else:
                counted, reached_limit = _count_frames_sequential(cap)
                if counted > 0:
                    total_frames = int(counted)
                    frame_count_source = "sequential_limited" if reached_limit else "sequential"
                elif reported_frame_count > 0:
                    total_frames = int(reported_frame_count)
                    frame_count_source = "metadata_unverified"
                else:
                    total_frames = 0
                    frame_count_source = "unavailable"
    finally:
        cap.release()

    if fps <= 0:
        duration_seconds = 0.0
    else:
        duration_seconds = float(total_frames / fps)

    info["total_frames"] = total_frames
    info["reported_frame_count"] = reported_frame_count
    info["frame_count_source"] = frame_count_source
    info["fps"] = fps
    info["width"] = width
    info["height"] = height
    info["duration_seconds"] = duration_seconds
    return info


def sample_frame_indices(
    total_frames: int,
    n_samples: int,
    mode: str = "uniform",
) -> List[int]:
    """
    Sinh danh sach frame index can lay.

    mode:
    - uniform: lay deu (evenly spaced), phu hop val/test
    - jitter: random 1 window lien tuc [t, t+n_samples), phu hop train
    - stride: lay theo stride co dinh (logic preprocess goc)
    """
    if n_samples <= 0:
        raise ValueError("n_samples phai > 0.")
    if total_frames <= 0:
        return []

    mode = mode.lower()
    if mode not in {"uniform", "jitter", "stride"}:
        raise ValueError("mode phai la 'uniform', 'jitter' hoac 'stride'.")

    if mode == "uniform":
        if total_frames >= n_samples:
            return (
                np.linspace(0, total_frames - 1, num=n_samples)
                .round()
                .astype(np.int64)
                .tolist()
            )
        base = list(range(total_frames))
        repeats = math.ceil(n_samples / len(base))
        return (base * repeats)[:n_samples]

    if mode == "jitter":
        if total_frames >= n_samples:
            start = random.randint(0, total_frames - n_samples)
            return list(range(start, start + n_samples))
        base = list(range(total_frames))
        repeats = math.ceil(n_samples / len(base))
        return (base * repeats)[:n_samples]

    # mode == "stride"
    stride = max(1, total_frames // n_samples)
    indices = list(range(0, total_frames, stride))[:n_samples]
    if len(indices) < n_samples:
        base = list(range(total_frames))
        repeats = math.ceil(n_samples / len(base))
        indices = (base * repeats)[:n_samples]
    return indices


def _read_frames_with_indices(
    video_path: str | Path, frame_indices: Sequence[int]
) -> List[Tuple[int, np.ndarray]]:
    """
    Doc frame tai cac index va giu dung mapping index->frame.
    """
    path = Path(video_path)
    if len(frame_indices) == 0:
        return []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    pairs: List[Tuple[int, np.ndarray]] = []
    try:
        for idx in frame_indices:
            frame_idx = int(idx)
            if frame_idx < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            pairs.append((frame_idx, frame))
    finally:
        cap.release()

    return pairs


def read_frames_at_indices(video_path: str | Path, frame_indices: Sequence[int]) -> List[np.ndarray]:
    """
    Doc frame tai cac index cho truoc, tra ve list frame BGR.

    Dung CAP_PROP_POS_FRAMES hieu qua hon doc tuan tu khi can sparse frames:
    ta nhay truc tiep toi vi tri frame quan tam, tranh decode toan bo frame trung gian.
    """
    return [frame for _, frame in _read_frames_with_indices(video_path, frame_indices)]


def is_video_corrupted(video_path: str | Path) -> bool:
    """
    Kiem tra video co loi hay khong:
    - Mo file that bai
    - total_frames <= 0 sau khi da validate metadata/fallback
    - khong doc duoc frame dau tien
    """
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        return True

    info = get_video_info(path)
    if int(info.get("total_frames", 0)) <= 0:
        return True

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return True
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return True
    finally:
        cap.release()

    return False


# ---- Backward-compatible helpers cho preprocess pipeline hien tai ----
def get_video_frame_count(video_path: Path) -> int:
    """Wrapper cu: lay tong so frame."""
    return int(get_video_info(video_path)["total_frames"])


def compute_sampling_stride(frame_count: int, samples_per_video: int) -> int:
    """Wrapper cu: tinh stride."""
    if samples_per_video <= 0:
        raise ValueError("samples_per_video phai > 0.")
    if frame_count <= 0:
        return 1
    return max(1, frame_count // samples_per_video)


def read_frames_by_indices(
    video_path: Path, frame_indices: Sequence[int]
) -> List[Tuple[int, np.ndarray]]:
    """Wrapper cu: tra ve ca frame_idx va frame."""
    return _read_frames_with_indices(video_path, frame_indices)


def load_sampled_frames(
    video_path: Path,
    samples_per_video: int,
) -> List[Tuple[int, np.ndarray]]:
    """Wrapper cu: lay frame theo stride va tra ve (idx, frame)."""
    total_frames = int(get_video_info(video_path)["total_frames"])
    indices = sample_frame_indices(
        total_frames=total_frames,
        n_samples=samples_per_video,
        mode="stride",
    )
    return _read_frames_with_indices(video_path, indices)


# ---- Unit test nho ----
def _create_dummy_video(path: Path, total_frames: int = 20, fps: int = 10) -> None:
    """Tao video gia de test nhanh."""
    width, height = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Khong tao duoc video test.")

    try:
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 7) % 255
            frame[:, :, 1] = (i * 13) % 255
            frame[:, :, 2] = (i * 19) % 255
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    random.seed(123)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        video_path = tmp_path / "dummy.avi"
        _create_dummy_video(video_path, total_frames=20, fps=10)

        mp4v_path = tmp_path / "sample.mp4v"
        mp4v_path.write_bytes(b"")
        listed = list_video_files(tmp_path)
        assert mp4v_path in listed

        # Test get_video_info
        info = get_video_info(video_path)
        assert info["total_frames"] > 0
        assert info["reported_frame_count"] > 0
        assert info["frame_count_source"] in {"metadata", "seek_probe", "sequential", "sequential_limited", "metadata_unverified"}
        assert info["fps"] > 0
        assert info["width"] == 64
        assert info["height"] == 48
        assert info["duration_seconds"] > 0

        # Test metadata lỗi: CAP_PROP_FRAME_COUNT=0 nhưng vẫn đọc được frame
        broken_meta_path = tmp_path / "broken_meta.mp4"
        broken_meta_path.write_bytes(b"not-a-real-video")

        real_videocapture = cv2.VideoCapture

        class _FakeBrokenMetaCapture:
            def __init__(self, total: int = 12) -> None:
                self.total = total
                self.pos = 0

            def isOpened(self) -> bool:
                return True

            def get(self, prop: int) -> float:
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 0.0
                if prop == cv2.CAP_PROP_FPS:
                    return 25.0
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return 64.0
                if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 48.0
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    return float(self.pos)
                return 0.0

            def set(self, prop: int, value: float) -> bool:
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.pos = max(0, min(int(value), self.total))
                    return True
                return False

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.pos >= self.total:
                    return False, None
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                self.pos += 1
                return True, frame

            def release(self) -> None:
                return None

        def _fake_videocapture(path_like: str) -> cv2.VideoCapture:
            if Path(path_like).name == "broken_meta.mp4":
                return _FakeBrokenMetaCapture(total=12)  # type: ignore[return-value]
            return real_videocapture(path_like)

        cv2.VideoCapture = _fake_videocapture  # type: ignore[assignment]
        try:
            info_broken = get_video_info(broken_meta_path)
            assert info_broken["reported_frame_count"] == 0
            assert int(info_broken["total_frames"]) == 12
            assert info_broken["frame_count_source"] in {"seek_probe", "sequential", "sequential_limited"}
        finally:
            cv2.VideoCapture = real_videocapture  # type: ignore[assignment]

        # Test sample_frame_indices cho tung mode
        idx_uniform = sample_frame_indices(total_frames=20, n_samples=5, mode="uniform")
        assert len(idx_uniform) == 5
        assert all(0 <= x < 20 for x in idx_uniform)

        idx_jitter = sample_frame_indices(total_frames=20, n_samples=5, mode="jitter")
        assert len(idx_jitter) == 5
        assert idx_jitter == list(range(idx_jitter[0], idx_jitter[0] + 5))

        idx_stride = sample_frame_indices(total_frames=20, n_samples=5, mode="stride")
        assert len(idx_stride) == 5
        assert all(0 <= x < 20 for x in idx_stride)

        # Test read_frames_at_indices
        frames = read_frames_at_indices(video_path, [0, 3, 7, 11])
        assert len(frames) == 4
        assert frames[0].shape == (48, 64, 3)

        # Test is_video_corrupted
        assert is_video_corrupted(video_path) is False
        empty_path = tmp_path / "empty.mp4"
        empty_path.write_bytes(b"")
        assert is_video_corrupted(empty_path) is True
        assert is_video_corrupted(tmp_path / "not_exists.mp4") is True

    print("Unit test passed: preprocess/video_utils.py")
