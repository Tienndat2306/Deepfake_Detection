"""PyTorch Lightning-compatible DataModule for deepfake detection."""

from __future__ import annotations

import random
import re
from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data.augmentation import (
    get_train_clip_transform,
    get_train_transform,
    get_val_clip_transform,
    get_val_transform,
)
from data.dataset import DeepfakeDataset

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    pl = None


def _stratified_split_indices(
    labels: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    """Split train/val theo stratified labels o cap sample."""
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("val_ratio phai nam trong (0, 1).")

    rng = random.Random(seed)
    labels_list = [int(x) for x in labels]
    idx_by_class: dict[int, list[int]] = {}
    for idx, y in enumerate(labels_list):
        idx_by_class.setdefault(y, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for class_idx in sorted(idx_by_class.keys()):
        indices = idx_by_class[class_idx]
        rng.shuffle(indices)

        if len(indices) <= 1:
            train_idx.extend(indices)
            continue

        n_val = max(1, int(round(len(indices) * float(val_ratio))))
        n_val = min(n_val, len(indices) - 1)
        val_part = indices[:n_val]
        train_part = indices[n_val:]
        train_idx.extend(train_part)
        val_idx.extend(val_part)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _stratified_group_split_indices(
    group_ids: Sequence[str],
    labels: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    """
    Split train/val theo group_id de tranh leakage.
    Moi group chi nam o mot trong hai tap.
    """
    if len(group_ids) != len(labels):
        raise ValueError("group_ids va labels phai cung do dai.")
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("val_ratio phai nam trong (0, 1).")

    rng = random.Random(seed)
    group_to_indices: dict[str, list[int]] = {}
    group_to_label: dict[str, int] = {}

    for idx, (gid, y) in enumerate(zip(group_ids, labels)):
        gid = str(gid)
        group_to_indices.setdefault(gid, []).append(idx)
        if gid not in group_to_label:
            group_to_label[gid] = int(y)

    groups_by_class: dict[int, list[str]] = {}
    for gid, y in group_to_label.items():
        groups_by_class.setdefault(y, []).append(gid)

    train_groups: list[str] = []
    val_groups: list[str] = []
    for class_idx in sorted(groups_by_class.keys()):
        groups = groups_by_class[class_idx]
        rng.shuffle(groups)
        if len(groups) <= 1:
            train_groups.extend(groups)
            continue

        n_val = max(1, int(round(len(groups) * float(val_ratio))))
        n_val = min(n_val, len(groups) - 1)
        val_groups.extend(groups[:n_val])
        train_groups.extend(groups[n_val:])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for gid in train_groups:
        train_idx.extend(group_to_indices[gid])
    for gid in val_groups:
        val_idx.extend(group_to_indices[gid])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class _PILToBGRAdapter:
    """Adapter de dung transform BGR-ndarray voi Dataset dang doc PIL."""

    def __init__(self, frame_transform) -> None:
        self.frame_transform = frame_transform

    def __call__(self, pil_image):
        rgb = np.asarray(pil_image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        return self.frame_transform(bgr)


class DeepfakeDataModule(pl.LightningDataModule if pl is not None else object):
    """
    DataModule quan ly train/val/test dataloader.

    Neu chua cai pytorch_lightning, class van ton tai de fallback import,
    nhung khuyen nghi cai lightning khi muon dung trainer cua Lightning.
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        num_frames: int = 10,
        num_clips_eval: int = 3,
        img_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 8,
        eval_num_workers: int = 12,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        use_weighted_sampler: bool = True,
        auto_split: bool = False,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        group_by_video_id: bool = False,
        use_clip_consistent_train: bool = False,
        pixel_temporal_jitter: float = 0.0,
        sampling_mode: str = "uniform",
    ) -> None:
        if pl is not None:
            super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_frames = int(num_frames)
        self.num_clips_eval = max(1, int(num_clips_eval))
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.eval_num_workers = int(eval_num_workers)
        self.prefetch_factor = int(max(1, prefetch_factor))
        self.pin_memory = bool(pin_memory)
        self.use_weighted_sampler = bool(use_weighted_sampler)

        self.auto_split = bool(auto_split)
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self.group_by_video_id = bool(group_by_video_id)

        self.use_clip_consistent_train = bool(use_clip_consistent_train)
        self.pixel_temporal_jitter = float(max(0.0, pixel_temporal_jitter))
        self.sampling_mode = str(sampling_mode).lower()  # [FIX-BUG-6]

        self.train_dataset: Optional[DeepfakeDataset] = None
        self.val_dataset: Optional[DeepfakeDataset] = None
        self.test_dataset: Optional[DeepfakeDataset] = None

    @staticmethod
    def _normalize_video_id(video_id: str) -> str:
        """
        Chuan hoa video_id de group split.
        Co strip hau to frame-index neu ten thu muc co dang ..._f0001.
        """
        raw = str(video_id).strip().lower()
        return re.sub(r"_f\d+$", "", raw)

    @staticmethod
    def _extract_labels(dataset_obj) -> np.ndarray:
        """Lay labels tu DeepfakeDataset hoac Subset[DeepfakeDataset]."""
        if isinstance(dataset_obj, Subset):
            if not hasattr(dataset_obj.dataset, "get_labels"):
                raise TypeError("Subset.dataset khong co get_labels().")
            base_labels = dataset_obj.dataset.get_labels()
            labels = [int(base_labels[i]) for i in dataset_obj.indices]
            return np.asarray(labels, dtype=np.int64)

        if not hasattr(dataset_obj, "get_labels"):
            raise TypeError("Dataset khong co get_labels().")
        return np.asarray(dataset_obj.get_labels(), dtype=np.int64)

    def _build_train_val_split(self, dataset: DeepfakeDataset) -> Tuple[list[int], list[int]]:
        labels = dataset.get_labels()
        if self.group_by_video_id:
            video_ids = [self._normalize_video_id(v) for v in dataset.get_video_ids()]
            return _stratified_group_split_indices(
                group_ids=video_ids,
                labels=labels,
                val_ratio=self.val_ratio,
                seed=self.split_seed,
            )
        return _stratified_split_indices(
            labels=labels,
            val_ratio=self.val_ratio,
            seed=self.split_seed,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_tf = _PILToBGRAdapter(get_train_transform(self.img_size))
        val_tf = _PILToBGRAdapter(get_val_transform(self.img_size))

        train_clip_tf = (
            get_train_clip_transform(
                img_size=self.img_size,
                pixel_temporal_jitter=self.pixel_temporal_jitter,
            )
            if self.use_clip_consistent_train
            else None
        )
        # Val/test luon dung clip transform de giu tinh nhat quan theo temporal clip.
        eval_clip_tf = get_val_clip_transform(self.img_size)

        if stage in (None, "fit"):
            use_auto_split = self.auto_split or self.val_dir is None
            if use_auto_split:
                train_base = DeepfakeDataset(
                    root_dir=self.train_dir,
                    num_frames=self.num_frames,
                    transform=train_tf,
                    clip_transform=train_clip_tf,
                    mode="train",
                    num_clips_eval=1,
                    sampling_mode=self.sampling_mode,  # [FIX-BUG-6]
                )
                val_base = DeepfakeDataset(
                    root_dir=self.train_dir,
                    num_frames=self.num_frames,
                    transform=val_tf,
                    clip_transform=eval_clip_tf,
                    mode="val",
                    num_clips_eval=self.num_clips_eval,
                    sampling_mode=self.sampling_mode,  # [FIX-BUG-6]
                )
                train_idx, val_idx = self._build_train_val_split(train_base)
                self.train_dataset = Subset(train_base, train_idx)
                self.val_dataset = Subset(val_base, val_idx)
            else:
                self.train_dataset = DeepfakeDataset(
                    root_dir=self.train_dir,
                    num_frames=self.num_frames,
                    transform=train_tf,
                    clip_transform=train_clip_tf,
                    mode="train",
                    num_clips_eval=1,
                    sampling_mode=self.sampling_mode,  # [FIX-BUG-6]
                )
                self.val_dataset = DeepfakeDataset(
                    root_dir=self.val_dir,
                    num_frames=self.num_frames,
                    transform=val_tf,
                    clip_transform=eval_clip_tf,
                    mode="val",
                    num_clips_eval=self.num_clips_eval,
                    sampling_mode=self.sampling_mode,  # [FIX-BUG-6]
                )

        if stage in (None, "test") and self.test_dir is not None:
            self.test_dataset = DeepfakeDataset(
                root_dir=self.test_dir,
                num_frames=self.num_frames,
                transform=val_tf,
                clip_transform=eval_clip_tf,
                mode="test",
                num_clips_eval=self.num_clips_eval,
                sampling_mode=self.sampling_mode,  # [FIX-BUG-6]
            )

    def _build_train_sampler(self):
        if self.train_dataset is None or not self.use_weighted_sampler:
            return None

        labels = self._extract_labels(self.train_dataset)
        if labels.size == 0:
            return None

        class_counts = np.bincount(labels, minlength=2)
        class_weights = np.zeros_like(class_counts, dtype=np.float64)
        nonzero = class_counts > 0
        class_weights[nonzero] = 1.0 / class_counts[nonzero]
        sample_weights = class_weights[labels]

        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') truoc khi lay train_dataloader.")

        sampler = self._build_train_sampler()
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": True,  # On dinh hon cho training step theo batch co kich thuoc dong deu.
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = True

        return DataLoader(
            self.train_dataset,
            **loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') truoc khi lay val_dataloader.")

        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": None,
            "num_workers": self.eval_num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }
        if self.eval_num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = True

        return DataLoader(
            self.val_dataset,
            **loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') truoc khi lay test_dataloader.")

        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": None,
            "num_workers": self.eval_num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }
        if self.eval_num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = True

        return DataLoader(
            self.test_dataset,
            **loader_kwargs,
        )
