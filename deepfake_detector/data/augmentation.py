"""Data augmentation pipeline definitions."""

from __future__ import annotations

import inspect  # [FIX-BUG-2]
from typing import Any, Dict, List, Optional, Sequence

import albumentations as A
import numpy as np
import torch
from torchvision import transforms as T

# Mean / std chuan ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameAugmentation:
    """
    Wrapper callable cho 1 frame (numpy HWC, BGR tu OpenCV) -> torch.Tensor [C, H, W].
    """

    def __init__(self, augment: A.BasicTransform) -> None:
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    @staticmethod
    def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
        """Albumentations xu ly anh RGB tot hon cho cac phep mau sac."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Frame dau vao phai co shape [H, W, 3] (BGR).")
        return frame_bgr[..., ::-1]

    def _postprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Chuyen RGB numpy -> tensor va normalize theo ImageNet."""
        tensor = self.to_tensor(image_rgb)
        return self.normalize(tensor)

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Ap dung augmentation ngau nhien (cho truong hop frame don le)."""
        image_rgb = self._bgr_to_rgb(frame_bgr)
        output = self.augment(image=image_rgb)
        return self._postprocess(output["image"])

    def apply_and_get_replay(
        self, frame_bgr: np.ndarray
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Ap dung augmentation len frame dau tien va tra ve replay dict.
        Replay dict nay duoc dung de lap lai CHINH XAC cung phep bien doi cho cac frame sau.
        """
        if not isinstance(self.augment, A.ReplayCompose):
            raise TypeError("apply_and_get_replay chi dung voi A.ReplayCompose.")

        image_rgb = self._bgr_to_rgb(frame_bgr)
        output = self.augment(image=image_rgb)
        replay = output.get("replay")
        if replay is None:
            raise RuntimeError("Khong lay duoc replay tu ReplayCompose.")
        return self._postprocess(output["image"]), replay

    def apply_with_replay(
        self, frame_bgr: np.ndarray, replay: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Ap dung lai dung replay da luu cho frame tiep theo trong clip.
        """
        if not isinstance(self.augment, A.ReplayCompose):
            raise TypeError("apply_with_replay chi dung voi A.ReplayCompose.")

        image_rgb = self._bgr_to_rgb(frame_bgr)
        output = A.ReplayCompose.replay(replay, image=image_rgb)
        return self._postprocess(output["image"])


class ClipAugmentation:
    """
    Augmentation cho ca clip:
    - Spatial aug: nhat quan theo thoi gian (ReplayCompose).
    - Pixel aug: co the dong bo hoac dao dong nhe giua cac frame.
    """

    def __init__(
        self,
        spatial_aug: A.BasicTransform,
        pixel_aug: Optional[A.BasicTransform] = None,
        pixel_temporal_jitter: float = 0.0,
    ) -> None:
        self.spatial_aug = spatial_aug
        self.pixel_aug = pixel_aug
        self.pixel_temporal_jitter = float(max(0.0, pixel_temporal_jitter))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    @staticmethod
    def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Frame dau vao phai co shape [H, W, 3] (BGR).")
        return frame_bgr[..., ::-1]

    @staticmethod
    def _to_rgb_from_any(frame: np.ndarray, input_is_bgr: bool) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Moi frame phai co shape [H, W, 3].")
        if input_is_bgr:
            return frame[..., ::-1]
        return frame

    def _postprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        tensor = self.to_tensor(image_rgb)
        return self.normalize(tensor)

    def apply_spatial_consistent(self, frames_bgr: Sequence[np.ndarray]) -> List[np.ndarray]:
        """
        Ap dung spatial augmentation nhat quan cho toan clip.
        Tra ve list anh RGB sau spatial.
        """
        if len(frames_bgr) == 0:
            raise ValueError("frames_bgr khong duoc rong.")

        first_rgb = self._bgr_to_rgb(frames_bgr[0])
        if isinstance(self.spatial_aug, A.ReplayCompose):
            output = self.spatial_aug(image=first_rgb)
            replay = output.get("replay")
            if replay is None:
                raise RuntimeError("Khong lay duoc replay tu spatial ReplayCompose.")
            spatial_frames = [output["image"]]
            for frame in frames_bgr[1:]:
                rgb = self._bgr_to_rgb(frame)
                replayed = A.ReplayCompose.replay(replay, image=rgb)
                spatial_frames.append(replayed["image"])
            return spatial_frames

        spatial_frames = [self.spatial_aug(image=first_rgb)["image"]]
        for frame in frames_bgr[1:]:
            rgb = self._bgr_to_rgb(frame)
            spatial_frames.append(self.spatial_aug(image=rgb)["image"])
        return spatial_frames

    def apply_pixel_per_frame(
        self,
        frames_rgb_or_bgr: Sequence[np.ndarray],
        temporal_jitter_strength: Optional[float] = None,
        input_is_bgr: bool = False,
    ) -> List[np.ndarray]:
        """
        Ap dung pixel augmentation:
        - jitter <= 0: dong bo theo toan clip (1 replay).
        - jitter > 0: moi frame sample ngau nhien doc lap de tao robust sensor noise.
        """
        if len(frames_rgb_or_bgr) == 0:
            raise ValueError("frames_rgb_or_bgr khong duoc rong.")
        if self.pixel_aug is None:
            return [self._to_rgb_from_any(frame, input_is_bgr) for frame in frames_rgb_or_bgr]

        jitter = self.pixel_temporal_jitter if temporal_jitter_strength is None else float(max(0.0, temporal_jitter_strength))
        frames_rgb = [self._to_rgb_from_any(frame, input_is_bgr) for frame in frames_rgb_or_bgr]

        if isinstance(self.pixel_aug, A.ReplayCompose) and jitter <= 0.0:
            output = self.pixel_aug(image=frames_rgb[0])
            replay = output.get("replay")
            if replay is None:
                raise RuntimeError("Khong lay duoc replay tu pixel ReplayCompose.")
            out_frames = [output["image"]]
            for frame in frames_rgb[1:]:
                replayed = A.ReplayCompose.replay(replay, image=frame)
                out_frames.append(replayed["image"])
            return out_frames

        return [self.pixel_aug(image=frame)["image"] for frame in frames_rgb]

    def __call__(self, frames_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        """
        Apply train clip augmentation va tra ve tensor [T, C, H, W].
        """
        spatial_frames = self.apply_spatial_consistent(frames_bgr)
        pixel_frames = self.apply_pixel_per_frame(
            spatial_frames,
            temporal_jitter_strength=self.pixel_temporal_jitter,
            input_is_bgr=False,
        )
        tensors = [self._postprocess(frame) for frame in pixel_frames]
        return torch.stack(tensors, dim=0)


class ClipValTransform:
    """Transform deterministic cho val/test clip."""

    def __init__(self, spatial_aug: A.BasicTransform) -> None:
        self.spatial_aug = spatial_aug
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    @staticmethod
    def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Frame dau vao phai co shape [H, W, 3] (BGR).")
        return frame_bgr[..., ::-1]

    def __call__(self, frames_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        if len(frames_bgr) == 0:
            raise ValueError("frames_bgr khong duoc rong.")
        out: List[torch.Tensor] = []
        for frame in frames_bgr:
            rgb = self._bgr_to_rgb(frame)
            aug = self.spatial_aug(image=rgb)["image"]
            tensor = self.to_tensor(aug)
            out.append(self.normalize(tensor))
        return torch.stack(out, dim=0)


def build_train_spatial_augment(img_size: int = 256) -> A.ReplayCompose:
    """Spatial aug cho train clip (replay de giu tinh nhat quan thoi gian)."""
    return A.ReplayCompose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=8, interpolation=1, border_mode=4, p=0.3),
        ]
    )


def _get_transform_param_names(transform_cls: type) -> set[str]:
    """Lay tap tham so hop le de giu tuong thich API qua nhieu version."""  # [FIX-BUG-2]
    signature = None
    for candidate in (transform_cls, getattr(transform_cls, "__init__", None)):
        if candidate is None:
            continue
        try:
            signature = inspect.signature(candidate)
            break
        except (TypeError, ValueError):
            continue

    if signature is None:
        return set()

    return {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _resolve_range(
    value: Any,
    default: tuple[float, float],
    cast_type=float,
) -> tuple[Any, Any]:
    """Chuan hoa gia tri scalar/list ve tuple (low, high)."""  # [FIX-BUG-2]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) >= 2:
            return cast_type(values[0]), cast_type(values[1])
        if len(values) == 1:
            casted = cast_type(values[0])
            return casted, casted

    if isinstance(value, (int, float)):
        casted = cast_type(value)
        return casted, casted

    return cast_type(default[0]), cast_type(default[1])


def _build_gauss_noise_transform(noise_cfg: Dict[str, Any]) -> A.BasicTransform:
    """Tao GaussNoise voi mapping kwargs theo version Albumentations."""  # [FIX-BUG-2]
    var_low, var_high = _resolve_range(noise_cfg.get("var_limit", (10, 50)), (10, 50))
    var_low = float(max(0.0, var_low))
    var_high = float(max(var_low, var_high))
    p = float(noise_cfg.get("p", 0.2))

    param_names = _get_transform_param_names(A.GaussNoise)
    kwargs: Dict[str, Any] = {"p": p}
    if "var_limit" in param_names:
        kwargs["var_limit"] = (var_low, var_high)
    else:
        kwargs["std_range"] = ((var_low ** 0.5) / 255.0, (var_high ** 0.5) / 255.0)
        if "mean_range" in param_names:
            kwargs["mean_range"] = (0.0, 0.0)

    return A.GaussNoise(**kwargs)


def _build_image_compression_transform(compression_cfg: Dict[str, Any]) -> A.BasicTransform:
    """Tao ImageCompression tu block jpeg_compression trong YAML."""  # [FIX-BUG-2]
    quality_low, quality_high = _resolve_range(
        (
            compression_cfg.get("quality_lower", 60),
            compression_cfg.get("quality_upper", 100),
        ),
        (60, 100),
        cast_type=int,
    )
    quality_low = int(max(0, quality_low))
    quality_high = int(max(quality_low, quality_high))
    p = float(compression_cfg.get("p", 0.3))

    param_names = _get_transform_param_names(A.ImageCompression)
    kwargs: Dict[str, Any] = {"p": p}
    if "quality_lower" in param_names:
        kwargs["quality_lower"] = quality_low
        kwargs["quality_upper"] = quality_high
    else:
        kwargs["quality_range"] = (quality_low, quality_high)
        if "compression_type" in param_names:
            kwargs["compression_type"] = "jpeg"

    return A.ImageCompression(**kwargs)


def _build_coarse_dropout_transform(dropout_cfg: Dict[str, Any]) -> A.BasicTransform:
    """Tao CoarseDropout voi mapping kwargs backward/forward compatible."""  # [FIX-BUG-2]
    max_holes = int(max(1, dropout_cfg.get("max_holes", 4)))
    max_height = int(max(1, dropout_cfg.get("max_height", 32)))
    max_width = int(max(1, dropout_cfg.get("max_width", 32)))
    p = float(dropout_cfg.get("p", 0.2))

    param_names = _get_transform_param_names(A.CoarseDropout)
    kwargs: Dict[str, Any] = {"p": p}
    if "max_holes" in param_names:
        kwargs["max_holes"] = max_holes
        kwargs["max_height"] = max_height
        kwargs["max_width"] = max_width
    else:
        if "num_holes_range" in param_names:
            kwargs["num_holes_range"] = (1, max_holes)
        if "hole_height_range" in param_names:
            kwargs["hole_height_range"] = (1, max_height)
        if "hole_width_range" in param_names:
            kwargs["hole_width_range"] = (1, max_width)

    return A.CoarseDropout(**kwargs)


def build_train_pixel_augment(
    cfg: Optional[Dict[str, Any]] = None,
) -> A.ReplayCompose:
    """Pixel aug cho train clip/frame."""
    # [FIX B1] Doc tham so tu config neu co, fallback ve mac dinh an toan.
    cfg = cfg or {}
    cj_cfg = cfg.get("color_jitter", {}) if isinstance(cfg.get("color_jitter"), dict) else {}
    cj_brightness = float(cj_cfg.get("brightness", 0.15))
    cj_contrast = float(cj_cfg.get("contrast", 0.15))
    cj_saturation = float(cj_cfg.get("saturation", 0.10))
    cj_hue = float(cj_cfg.get("hue", 0.0))
    cj_p = float(cj_cfg.get("p", 0.8))

    blur_cfg = cfg.get("gaussian_blur", {}) if isinstance(cfg.get("gaussian_blur"), dict) else {}
    blur_limit = tuple(blur_cfg.get("blur_limit", (3, 5)))
    blur_p = float(blur_cfg.get("p", 0.3))

    noise_cfg = cfg.get("gaussian_noise", {}) if isinstance(cfg.get("gaussian_noise"), dict) else {}  # [FIX-BUG-2]
    compression_cfg = (
        cfg.get("jpeg_compression", {}) if isinstance(cfg.get("jpeg_compression"), dict) else {}
    )  # [FIX-BUG-2]
    dropout_cfg = cfg.get("coarse_dropout", {}) if isinstance(cfg.get("coarse_dropout"), dict) else {}  # [FIX-BUG-2]
    gray_p = float(cfg.get("random_grayscale_p", 0.1))
    return A.ReplayCompose(
        [
            A.ColorJitter(
                brightness=cj_brightness,
                contrast=cj_contrast,
                saturation=cj_saturation,
                hue=cj_hue,
                p=cj_p,
            ),
            A.GaussianBlur(blur_limit=blur_limit, p=blur_p),
            _build_gauss_noise_transform(noise_cfg),  # [FIX-BUG-2]
            _build_image_compression_transform(compression_cfg),  # [FIX-BUG-2]
            A.ToGray(p=gray_p, num_output_channels=3),
            _build_coarse_dropout_transform(dropout_cfg),  # [FIX-BUG-2]
        ]
    )


def get_train_clip_transform(
    img_size: int = 256,
    pixel_temporal_jitter: float = 0.0,
    cfg: Optional[Dict[str, Any]] = None,
) -> ClipAugmentation:
    """Tra ve clip-level train transform (spatial consistent, pixel configurable)."""
    return ClipAugmentation(
        spatial_aug=build_train_spatial_augment(img_size=img_size),
        # [FIX B1] Truyen cfg de bo tham so augmentation hardcode.
        pixel_aug=build_train_pixel_augment(cfg=cfg),
        pixel_temporal_jitter=float(max(0.0, pixel_temporal_jitter)),
    )


def get_val_clip_transform(img_size: int = 256) -> ClipValTransform:
    """Tra ve clip-level deterministic val/test transform."""
    resize_size = int(img_size * 1.14)
    spatial = A.Compose(
        [
            A.Resize(height=resize_size, width=resize_size),
            A.CenterCrop(height=img_size, width=img_size),
        ]
    )
    return ClipValTransform(spatial_aug=spatial)


def get_train_transform(img_size: int = 256) -> FrameAugmentation:
    """
    Train transform cho single frame (input BGR tu OpenCV):
    - RandomHorizontalFlip
    - ColorJitter (nhe)
    - GaussianBlur (p=0.3)
    - RandomGrayscale (p=0.1)
    - Normalize theo ImageNet
    """
    # Backward-compatible single-frame transform: gom ca spatial + pixel trong 1 pipeline.
    train_aug = A.ReplayCompose(
        build_train_spatial_augment(img_size=img_size).transforms
        + build_train_pixel_augment().transforms
    )
    return FrameAugmentation(train_aug)


def get_val_transform(img_size: int = 256) -> FrameAugmentation:
    """
    Val/Test transform khong random:
    - Resize
    - CenterCrop
    - Normalize theo ImageNet
    """
    resize_size = int(img_size * 1.14)
    val_aug = A.Compose(
        [
            A.Resize(height=resize_size, width=resize_size),
            A.CenterCrop(height=img_size, width=img_size),
        ]
    )
    return FrameAugmentation(val_aug)


def apply_train_transform_consistent(
    frames_bgr: Sequence[np.ndarray],
    train_transform: Optional[FrameAugmentation] = None,
) -> torch.Tensor:
    """
    Ap dung train augmentation NHAT QUAN cho tat ca frame trong cung 1 clip.

    Tai sao dung ReplayCompose:
    - Neu goi transform ngau nhien rieng cho tung frame, moi frame se bi bien doi khac nhau
      (flip frame nay, khong flip frame kia; muc color jitter cung khac), gay "nhay" theo thoi gian.
    - ReplayCompose cho phep sample ngau nhien 1 lan o frame dau tien,
      sau do replay lai dung tham so do cho cac frame con lai.
    - Nho vay clip giu duoc tinh lien tuc thoi gian, model hoc dac trung motion/on dinh tot hon.
    """
    if len(frames_bgr) == 0:
        raise ValueError("frames_bgr khong duoc rong.")

    transform = train_transform or get_train_transform()
    first_tensor, replay = transform.apply_and_get_replay(frames_bgr[0])

    clip_tensors = [first_tensor]
    for frame in frames_bgr[1:]:
        clip_tensors.append(transform.apply_with_replay(frame, replay))

    return torch.stack(clip_tensors, dim=0)

