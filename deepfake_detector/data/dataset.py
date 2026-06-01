"""Dataset definitions and dataloader helpers."""

import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, TypedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VideoSample(TypedDict):
    """Thong tin cho moi sample video."""

    video_dir: Path
    frame_paths: List[Path]
    label: int


class DeepfakeDataset(Dataset):
    """
    Dataset cho bai toan deepfake detection theo chuoi frame.

    Cau truc thu muc ky vong:
    root_dir/
      Real/
        video_001/
          frame_0001.jpg
          ...
      Fake/
        video_002/
          frame_0001.jpg
          ...
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 10,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        clip_transform: Optional[Callable[[Sequence[np.ndarray]], torch.Tensor]] = None,
        mode: str = "train",
        num_clips_eval: int = 1,
        sampling_mode: str = "uniform",
    ) -> None:
        # Luu tham so cau hinh co ban.
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.clip_transform = clip_transform
        self.mode = mode.lower()
        self.num_clips_eval = max(1, int(num_clips_eval))
        self.sampling_mode = str(sampling_mode).lower()  # [FIX-BUG-6]

        # Kiem tra mode hop le de tranh loi logic trong luc train/val/test.
        if self.mode not in {"train", "val", "test"}:
            raise ValueError("mode phai la 'train', 'val' hoac 'test'.")
        if self.sampling_mode not in {"uniform", "consecutive"}:  # [FIX-BUG-6]
            raise ValueError("sampling_mode phai la 'uniform' hoac 'consecutive'.")
        if self.mode == "train":
            # Train luon dung 1 clip/video.
            self.num_clips_eval = 1

        # Kiem tra thu muc du lieu co ton tai hay khong.
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Khong tim thay root_dir: {self.root_dir}")

        # Moi phan tu trong samples dai dien cho 1 video (1 sample).
        self.samples: List[VideoSample] = []

        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Khong tim thay video hop le trong thu muc: {self.root_dir}"
            )

    @staticmethod
    def _natural_key(path: Path) -> List[object]:
        """
        Tao key de sort ten frame kieu tu nhien:
        frame_2.jpg se dung truoc frame_10.jpg.
        """
        parts = re.split(r"(\d+)", path.stem.lower())
        key: List[object] = []
        for part in parts:
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part)
        key.append(path.suffix.lower())
        return key

    def _build_index(self) -> None:
        """Quet du lieu va tao danh sach sample theo tung video."""
        valid_ext = {".jpg", ".jpeg", ".png"}

        for class_dir in sorted([p for p in self.root_dir.iterdir() if p.is_dir()]):
            folder_name = class_dir.name
            folder_lower = folder_name.lower()

            # Ho tro ca "Real/Fake" va "real/fake".
            if folder_lower not in {"real", "fake"}:
                continue
            label = 1 if folder_lower == "fake" else 0

            # Moi thu muc con trong class tuong ung mot video.
            video_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()])

            # Fallback: neu class dir chua frame truc tiep (khong co subdir video),
            # van tao 1 sample de tranh bo sot du lieu.
            if not video_dirs:
                frame_paths = [
                    p
                    for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in valid_ext
                ]
                frame_paths = sorted(frame_paths, key=self._natural_key)
                if frame_paths:
                    self.samples.append(
                        {
                            "video_dir": class_dir,
                            "frame_paths": frame_paths,
                            "label": label,
                        }
                    )
                continue

            for video_dir in video_dirs:
                # Chi lay frame anh theo phan mo rong pho bien.
                frame_paths = [
                    p
                    for p in video_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in valid_ext
                ]

                # Sort de dam bao thu tu thoi gian on dinh.
                frame_paths = sorted(frame_paths, key=self._natural_key)

                # Neu video khong co frame thi bo qua.
                if not frame_paths:
                    continue

                self.samples.append(
                    {
                        "video_dir": video_dir,
                        "frame_paths": frame_paths,
                        "label": label,
                    }
                )

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Chuyen PIL image sang tensor [C, H, W] trong khoang [0, 1]."""
        image = image.convert("RGB")
        # Dung numpy de tao tensor on dinh, tranh warning ve buffer writeable.
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor

    def _tile_if_needed(self, frame_paths: Sequence[Path]) -> List[Path]:
        """
        Neu so frame < num_frames thi tile theo cyclic index cho du do dai.
        Vi du n=3, num_frames=10 -> [0,1,2,0,1,2,0,1,2,0].
        """
        n = len(frame_paths)
        if n == 0:
            raise ValueError("Video khong co frame hop le.")

        if n >= self.num_frames:
            return list(frame_paths)

        indices = [i % n for i in range(self.num_frames)]
        return [frame_paths[i] for i in indices]

    def _compute_eval_clip_starts(self, total_frames: int) -> List[int]:
        """
        Tinh start index cho val/test multi-clip theo dau/giua/cuoi.
        """
        if total_frames <= self.num_frames:
            return [0] * self.num_clips_eval

        max_start = total_frames - self.num_frames
        if self.num_clips_eval == 1:
            return [max_start // 2]

        starts = torch.linspace(0, max_start, steps=self.num_clips_eval).tolist()
        return [int(round(s)) for s in starts]

    def _select_train_clip_paths(self, frame_paths: Sequence[Path]) -> List[List[Path]]:
        """
        Chon clip train theo sampling_mode cau hinh.  # [FIX-BUG-6b]

        QUAN TRONG - gioi han pipeline hien tai:
        frame_paths la cac SPARSE FRAMES da duoc preprocess (khong lien tiep trong video goc).
        - sampling_mode="uniform": linspace pick -> moi frame cach nhau deu theo sparse set.
          Phu hop va duoc khuyen dung khi sparse frames < 2x num_frames.
        - sampling_mode="consecutive": random window lien tiep trong sparse set.
          Cac frame trong window van la sparse, khong phai consecutive trong video thuc.
          Chi co loi khi so sparse frames >> num_frames (co du khong gian chon window).
        De khai thac temporal thuc su, can preprocess lai de luu consecutive frames goc.
        """
        frame_paths = self._tile_if_needed(frame_paths)  # [FIX-BUG-6]
        total = len(frame_paths)
        if total == self.num_frames:
            return [list(frame_paths)]

        if self.sampling_mode == "consecutive":  # [FIX-BUG-6]
            max_start = total - self.num_frames
            start = random.randint(0, max_start)
            end = start + self.num_frames
            return [list(frame_paths[start:end])]

        indices = np.linspace(0, total - 1, num=self.num_frames)  # [FIX-BUG-6]
        indices = np.clip(np.rint(indices).astype(np.int64), 0, total - 1)  # [FIX-BUG-6]
        return [[frame_paths[int(idx)] for idx in indices.tolist()]]  # [FIX-BUG-6]

    def _select_clip_paths(self, frame_paths: Sequence[Path]) -> List[List[Path]]:
        """
        Chon clip frame theo mode:
        - train: 1 clip theo sampling_mode [T].
        - val/test: num_clips_eval clip (dau/giua/cuoi), moi clip co T frame.
        """
        if self.mode == "train":
            return self._select_train_clip_paths(frame_paths)  # [FIX-BUG-6]

        base_paths = list(frame_paths)
        if len(base_paths) < self.num_frames:
            tiled = self._tile_if_needed(base_paths)
            return [list(tiled) for _ in range(self.num_clips_eval)]

        starts = self._compute_eval_clip_starts(len(base_paths))
        clips: List[List[Path]] = []
        for start in starts:
            end = start + self.num_frames
            clips.append(list(base_paths[start:end]))
        return clips

    def __len__(self) -> int:
        """So luong sample bang so luong video."""
        return len(self.samples)

    @staticmethod
    def _pil_to_bgr_array(image: Image.Image) -> np.ndarray:
        """Chuyen PIL image sang numpy BGR [H, W, C]."""
        rgb = np.asarray(image.convert("RGB"))
        return rgb[:, :, ::-1].copy()

    def _load_frames_bgr(self, selected_paths: Sequence[Path]) -> List[np.ndarray]:
        """Doc list frame tu disk va tra ve danh sach anh BGR."""
        frames_bgr: List[np.ndarray] = []
        for frame_path in selected_paths:
            with Image.open(frame_path) as img:
                frames_bgr.append(self._pil_to_bgr_array(img))
        return frames_bgr

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Tra ve:
        - train: [T, C, H, W]
        - val/test: [num_clips_eval, T, C, H, W]
        - label: 0 (Real) hoac 1 (Fake)
        """
        sample = self.samples[index]
        frame_paths = sample["frame_paths"]
        label = sample["label"]

        clip_paths_list = self._select_clip_paths(frame_paths)

        if self.clip_transform is not None:
            clip_tensors: List[torch.Tensor] = []
            for clip_paths in clip_paths_list:
                frames_bgr = self._load_frames_bgr(clip_paths)
                frames_tensor = self.clip_transform(frames_bgr)
                if not isinstance(frames_tensor, torch.Tensor):
                    raise TypeError("clip_transform phai tra ve torch.Tensor.")
                if frames_tensor.ndim != 4:
                    raise ValueError(
                        "clip_transform phai tra ve tensor co shape [T, C, H, W]."
                    )
                if frames_tensor.shape[0] != self.num_frames:
                    raise ValueError(
                        f"clip_transform phai tra ve T={self.num_frames}, nhan {frames_tensor.shape[0]}."
                    )
                clip_tensors.append(frames_tensor)

            final_tensor = torch.stack(clip_tensors, dim=0)  # [N, T, C, H, W]
            if self.mode == "train":
                final_tensor = final_tensor[0]  # [T, C, H, W], khong dung squeeze()
            return final_tensor, int(label)

        clip_tensors: List[torch.Tensor] = []
        for clip_paths in clip_paths_list:
            frame_tensors: List[torch.Tensor] = []
            for frame_path in clip_paths:
                with Image.open(frame_path) as img:
                    img = img.convert("RGB")

                    # Neu co transform thi dung transform cua nguoi dung.
                    if self.transform is not None:
                        frame_tensor = self.transform(img)
                    else:
                        # Khong co transform thi dung chuyen doi mac dinh ve tensor.
                        frame_tensor = self._pil_to_tensor(img)

                if not isinstance(frame_tensor, torch.Tensor):
                    raise TypeError("transform phai tra ve torch.Tensor.")
                if frame_tensor.ndim != 3:
                    raise ValueError("Moi frame phai co shape [C, H, W] sau transform.")
                frame_tensors.append(frame_tensor)

            # Ghep danh sach frame thanh tensor chuoi [T, C, H, W].
            clip_tensors.append(torch.stack(frame_tensors, dim=0))

        final_tensor = torch.stack(clip_tensors, dim=0)  # [N, T, C, H, W]
        if self.mode == "train":
            final_tensor = final_tensor[0]  # [T, C, H, W], khong dung squeeze()
        return final_tensor, int(label)

    def get_labels(self) -> List[int]:
        """Tra ve list nhan cua toan bo dataset theo thu tu sample."""
        return [sample["label"] for sample in self.samples]

    def get_video_ids(self) -> List[str]:
        """Tra ve video_id (ten thu muc video) theo thu tu sample."""
        return [sample["video_dir"].name for sample in self.samples]

