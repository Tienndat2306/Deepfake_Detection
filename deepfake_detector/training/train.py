"""Training entrypoint script."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

try:
    from data.augmentation import (
        get_train_clip_transform,
        get_train_transform,
        get_val_clip_transform,
        get_val_transform,
    )
    from data.dataset import DeepfakeDataset
    from models.deepfake_model import DeepfakeDetector
    from training.loss import FocalLossWithSmoothing
    from training.trainer import Trainer
except ImportError:
    from ..data.augmentation import (
        get_train_clip_transform,
        get_train_transform,
        get_val_clip_transform,
        get_val_transform,
    )
    from ..data.dataset import DeepfakeDataset
    from ..models.deepfake_model import DeepfakeDetector
    from .loss import FocalLossWithSmoothing
    from .trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train deepfake detection model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path den train_config.yaml.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path den checkpoint de resume training (optional).",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load train config YAML.
    Neu file --config chua co 'model', se merge tu model_config.yaml cung thu muc.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay config: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise RuntimeError("Config YAML khong hop le (khong phai dict).")

    if "model" not in config:
        fallback_model_cfg = path.parent / "model_config.yaml"
        if fallback_model_cfg.exists():
            with fallback_model_cfg.open("r", encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f) or {}
            if isinstance(model_cfg, dict) and "model" in model_cfg:
                config["model"] = model_cfg["model"]

    return config


def load_aug_config(train_config_path: str) -> Dict[str, Any]:
    """Load aug_config.yaml cung thu muc voi train_config (neu co)."""
    # [FIX B1] Doc aug_config.yaml de tranh hardcode augmentation trong Python.
    aug_path = Path(train_config_path).parent / "aug_config.yaml"
    if not aug_path.exists():
        return {}

    with aug_path.open("r", encoding="utf-8") as f:
        aug_cfg = yaml.safe_load(f) or {}
    if not isinstance(aug_cfg, dict):
        return {}
    return aug_cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed random de reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(device_str: str) -> torch.device:
    """Resolve device tu config."""
    requested = str(device_str).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    """Tra ve (tong params, trainable params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def infer_feat_dim(model_cfg: Dict[str, Any]) -> int:
    """
    Suy ra feat_dim tu model config.
    Uu tien model.feat_dim, fallback theo efficientnet.model_name.
    """
    if "feat_dim" in model_cfg:
        return int(model_cfg["feat_dim"])

    eff_cfg = model_cfg.get("efficientnet", {}) if isinstance(model_cfg, dict) else {}
    model_name = str(eff_cfg.get("model_name", "efficientnet_b4")).lower()
    feat_dim_map = {
        "efficientnet_b0": 1280,
        "efficientnet_b4": 1792,
    }
    return int(feat_dim_map.get(model_name, 1792))


def build_checkpoint_path(config: Dict[str, Any]) -> Path:
    """
    Build duong dan checkpoint theo key config moi + tuong thich key cu.

    Uu tien:
    1) checkpoint.path
    2) training.checkpoint_path (legacy)
    3) checkpoint.save_dir + checkpoint.filename / <experiment.name>_best.pth
    """
    checkpoint_cfg = config.get("checkpoint", {})
    training_cfg = config.get("training", {})
    exp_cfg = config.get("experiment", {})

    explicit_path = checkpoint_cfg.get("path")
    if explicit_path:
        return Path(str(explicit_path))

    legacy_path = training_cfg.get("checkpoint_path")
    if legacy_path:
        return Path(str(legacy_path))

    save_dir = Path(str(checkpoint_cfg.get("save_dir", "checkpoints")))
    filename = checkpoint_cfg.get("filename")
    if not filename:
        exp_name = str(exp_cfg.get("name", "experiment"))
        filename = f"{exp_name}_best.pth"

    return save_dir / str(filename)


def stratified_split_indices(
    labels: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    """
    Stratified split train/val theo class labels.
    """
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("val_ratio phai nam trong (0, 1).")

    rng = random.Random(seed)
    labels_list = [int(x) for x in labels]
    idx_by_class: Dict[int, list[int]] = {}
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


def extract_labels_from_dataset(dataset) -> list[int]:
    """
    Lay labels tu DeepfakeDataset hoac Subset[DeepfakeDataset].
    """
    if isinstance(dataset, Subset):
        base_labels = dataset.dataset.get_labels()
        return [int(base_labels[i]) for i in dataset.indices]
    return [int(x) for x in dataset.get_labels()]


def build_sampler_from_labels(labels: Sequence[int]) -> WeightedRandomSampler:
    """
    Tao WeightedRandomSampler de can bang class trong train set.
    """
    if len(labels) == 0:
        raise ValueError("labels rong, khong tao duoc sampler.")

    label_arr = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(label_arr, minlength=2)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)

    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero]

    sample_weights = class_weights[label_arr]
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(sample_weights),
        replacement=True,
    )


class _PILToBGRAdapter:
    """Adapter de dung augmentation (input BGR numpy) voi Dataset dang doc PIL."""

    def __init__(self, frame_transform) -> None:
        self.frame_transform = frame_transform

    def __call__(self, pil_image):
        rgb = np.asarray(pil_image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        return self.frame_transform(bgr)


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    sampler=None,
    prefetch_factor: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    """Build DataLoader voi guard cho num_workers/persistent_workers."""
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
    }

    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(max(1, prefetch_factor))

    return DataLoader(**kwargs)


def maybe_resume(
    resume_path: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> Dict[str, Any]:
    """Resume model/optimizer/scheduler neu co checkpoint."""
    if not resume_path:
        return {"start_epoch": 1, "best_val_auc": float("-inf"), "resume_path": None}

    ckpt_path = Path(resume_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Khong tim thay resume checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint khong hop le (khong phai dict).")

    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint khong chua model state hop le.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[Resume] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[Resume] Unexpected keys: {len(unexpected_keys)}")

    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        if ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_auc = float(ckpt.get("val_auc", float("-inf")))
    return {
        "start_epoch": start_epoch,
        "best_val_auc": best_val_auc,
        "resume_path": str(ckpt_path),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    aug_config = load_aug_config(args.config)

    experiment_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    loss_cfg = config.get("loss", {})
    checkpoint_cfg = config.get("checkpoint", {})
    model_cfg = config.get("model", {})

    input_cfg = model_cfg.get("input", {})
    efficientnet_cfg = model_cfg.get("efficientnet", {})
    transformer_cfg = model_cfg.get("transformer", {})

    seed = int(experiment_cfg.get("seed", 42))
    deterministic = bool(experiment_cfg.get("deterministic", True))
    set_seed(seed=seed, deterministic=deterministic)

    device = resolve_device(str(training_cfg.get("device", "cuda")))

    num_frames = int(input_cfg.get("num_frames", model_cfg.get("num_frames", 10)))
    img_size = int(input_cfg.get("img_size", model_cfg.get("image_size", 256)))

    batch_size = int(training_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    # [FIX B3] Val/test multi-clip nang I/O hon train -> cap workers rieng.
    eval_num_workers = int(
        data_cfg.get("eval_num_workers", min(num_workers * 2, 12))
    )
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))

    num_epochs = int(training_cfg.get("num_epochs", training_cfg.get("epochs", 20)))
    patience = int(training_cfg.get("patience", 5))
    min_delta = float(training_cfg.get("min_delta", 0.0))
    grad_clip = float(training_cfg.get("gradient_clip_val", training_cfg.get("max_grad_norm", 1.0)))

    precision = int(training_cfg.get("precision", 16))
    use_amp = bool(training_cfg.get("use_amp", precision == 16))

    train_dir = data_cfg.get("train_dir")
    val_dir = data_cfg.get("val_dir")
    val_split = float(data_cfg.get("val_split", 0.15))
    num_clips_eval = int(data_cfg.get("num_clips_eval", 3))
    sampling_mode = str(data_cfg.get("sampling_mode", "uniform")).lower()  # [FIX-BUG-6]

    if train_dir is None:
        raise ValueError("Config can co data.train_dir.")

    train_transform = _PILToBGRAdapter(get_train_transform(img_size=img_size))
    val_transform = _PILToBGRAdapter(get_val_transform(img_size=img_size))
    train_aug_cfg = {}
    if isinstance(aug_config.get("augmentation"), dict):
        train_aug_cfg = aug_config.get("augmentation", {})
    elif isinstance(aug_config.get("train_augmentation"), dict):
        # [FIX B1] Backward-compatible voi cau truc aug_config cu.
        train_aug_cfg = aug_config.get("train_augmentation", {})
    # [FIX A2] Dung clip-level transform de giu temporal consistency trong clip.
    train_clip_transform = get_train_clip_transform(
        img_size=img_size,
        cfg=train_aug_cfg,
    )
    val_clip_transform = get_val_clip_transform(img_size=img_size)

    train_base = DeepfakeDataset(
        root_dir=str(train_dir),
        num_frames=num_frames,
        transform=None,
        clip_transform=train_clip_transform,
        mode="train",
        sampling_mode=sampling_mode,  # [FIX-BUG-6]
    )

    if val_dir is None:
        val_base = DeepfakeDataset(
            root_dir=str(train_dir),
            num_frames=num_frames,
            transform=None,
            clip_transform=val_clip_transform,
            mode="val",
            num_clips_eval=num_clips_eval,
            sampling_mode=sampling_mode,  # [FIX-BUG-6]
        )
        all_labels = train_base.get_labels()
        train_idx, val_idx = stratified_split_indices(
            labels=all_labels,
            val_ratio=val_split,
            seed=seed,
        )
        train_dataset = Subset(train_base, train_idx)
        val_dataset = Subset(val_base, val_idx)
    else:
        train_dataset = train_base
        val_dataset = DeepfakeDataset(
            root_dir=str(val_dir),
            num_frames=num_frames,
            transform=None,
            clip_transform=val_clip_transform,
            mode="val",
            num_clips_eval=num_clips_eval,
            sampling_mode=sampling_mode,  # [FIX-BUG-6]
        )

    train_labels = extract_labels_from_dataset(train_dataset)
    train_sampler = build_sampler_from_labels(train_labels)

    requested_pin_memory = bool(data_cfg.get("pin_memory", True))
    pin_memory = requested_pin_memory and device.type == "cuda"

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=train_sampler,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        sampler=None,
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    feat_dim = infer_feat_dim(model_cfg)
    dropout = float(
        efficientnet_cfg.get(
            "dropout",
            transformer_cfg.get("dropout", 0.3),
        )
    )

    model = DeepfakeDetector(
        feat_dim=feat_dim,
        d_model=int(transformer_cfg.get("d_model", 512)),
        nhead=int(transformer_cfg.get("nhead", 8)),
        num_layers=int(transformer_cfg.get("num_layers", 4)),
        dropout=dropout,
        num_frames=num_frames,
    )

    if bool(efficientnet_cfg.get("freeze_at_start", False)) and args.resume is None:
        model.backbone.freeze_backbone()  # [FIX-BUG-1b]

    lr_backbone = float(optimizer_cfg.get("lr_backbone", training_cfg.get("lr_backbone", 1e-5)))
    lr_head = float(optimizer_cfg.get("lr_head", training_cfg.get("lr_head", 1e-4)))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    betas_cfg = optimizer_cfg.get("betas", [0.9, 0.999])
    if isinstance(betas_cfg, (list, tuple)) and len(betas_cfg) >= 2:
        betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    else:
        betas = (0.9, 0.999)

    optimizer = AdamW(
        model.get_optimizer_groups(lr_backbone=lr_backbone, lr_head=lr_head),
        weight_decay=weight_decay,
        betas=betas,
    )

    t_max = int(scheduler_cfg.get("T_max", num_epochs))
    eta_min = float(scheduler_cfg.get("eta_min", training_cfg.get("eta_min", 1e-7)))
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))

    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max - warmup_epochs),
            eta_min=eta_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=eta_min,
        )

    criterion = FocalLossWithSmoothing(
        gamma=float(loss_cfg.get("gamma", training_cfg.get("focal_gamma", 2.0))),
        alpha=float(loss_cfg.get("alpha", training_cfg.get("focal_alpha", 0.75))),  # [FIX-BUG-7]
        smoothing=float(loss_cfg.get("smoothing", training_cfg.get("label_smoothing", 0.1))),
    )

    checkpoint_path = build_checkpoint_path(config)
    resume_info = maybe_resume(args.resume, model, optimizer, scheduler)
    # [FIX-BUG-1b] Tinh unfreeze_backbone_lr theo scheduler state thuc te sau resume.
    # Neu chua resume, dung lr_backbone goc. Neu da resume, lay LR hien tai cua optimizer
    # de tranh LR jump khi unfreeze backbone sau khi scheduler da decay.
    _effective_unfreeze_lr = lr_backbone
    if resume_info["resume_path"] is not None:
        # Lay LR nho nhat trong cac param group hien tai lam LR unfreeze an toan.
        _current_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        _effective_unfreeze_lr = min(_current_lrs) if _current_lrs else lr_backbone
        # Clamp tren lr_backbone de khong unfreeze voi LR lon hon ban dau.
        _effective_unfreeze_lr = min(_effective_unfreeze_lr, lr_backbone)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config={
            "patience": patience,
            "min_delta": min_delta,
            "checkpoint_path": str(checkpoint_path),
            "use_amp": use_amp,
            "max_grad_norm": grad_clip,
            "accumulate_grad_batches": int(
                training_cfg.get("accumulate_grad_batches", 1)
            ),  # [FIX-BUG-3]
            "save_top_k": int(checkpoint_cfg.get("save_top_k", 1)),  # [FIX-BUG-4]
            "unfreeze_after_epoch": int(
                efficientnet_cfg.get(
                    "unfreeze_after_epoch",
                    model_cfg.get("unfreeze_after_epoch", -1),
                )
            ),  # [FIX-BUG-1]
            "unfreeze_last_n_blocks": int(
                efficientnet_cfg.get(
                    "unfreeze_last_n_blocks",
                    model_cfg.get("unfreeze_last_n_blocks", 0),
                )
            ),  # [FIX-BUG-1]
            "unfreeze_backbone_lr": _effective_unfreeze_lr,  # [FIX-BUG-1b] dung LR thuc sau resume
        },
    )
    if math.isfinite(resume_info["best_val_auc"]):
        trainer.best_val_auc = float(resume_info["best_val_auc"])
        trainer.best_epoch = int(resume_info["start_epoch"]) - 1
    # [FIX-BUG-1b] Neu resume sau unfreeze_after_epoch, danh dau da unfreeze
    # de Trainer khong thuc hien lai o epoch dau sau resume (luc do backbone
    # da duoc load dung state tu checkpoint).
    if resume_info["resume_path"] is not None:
        _ue = trainer.unfreeze_after_epoch
        _se = int(resume_info["start_epoch"])
        if _ue >= 0 and _se > _ue:
            # Backbone da duoc unfreeze truoc do, set flag de tranh unfreeze lai
            # voi LR co the khong phu hop voi giai doan hien tai.
            trainer._unfreeze_applied = True
        elif _ue >= 0 and bool(efficientnet_cfg.get("freeze_at_start", False)):
            # [FIX-BUG-1b] Resume truoc/sat moc unfreeze: giu nguyen warmup freeze.
            model.backbone.freeze_backbone()

    total_params, trainable_params = count_params(model)
    print("========== Training Summary ==========")
    print(f"Train videos: {len(train_dataset)}")
    print(f"Val videos:   {len(val_dataset)}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable:    {trainable_params:,}")
    print(f"Device:       {device}")

    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        mem_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        print(f"GPU:          {gpu_name} ({mem_gb:.2f} GB)")
    else:
        print("GPU:          None (CPU mode)")

    print(f"Checkpoint:   {checkpoint_path}")
    if val_dir is None:
        print(f"Val split:    {val_split:.2%} (stratified from train_dir)")
    else:
        print(f"Val dir:      {val_dir}")

    if resume_info["resume_path"] is not None:
        print(f"Resume from:  {resume_info['resume_path']}")
        print(f"Start epoch:  {resume_info['start_epoch']}")
    print("======================================")

    start_epoch = int(resume_info["start_epoch"])
    if start_epoch > num_epochs:
        print(
            f"Resume epoch ({start_epoch}) > total epochs ({num_epochs}), skip training."
        )
        return

    remaining_epochs = num_epochs - start_epoch + 1
    print(f"Run training for {remaining_epochs} epoch(s).")
    trainer.fit(num_epochs=remaining_epochs, start_epoch=start_epoch)  # [FIX-BUG-1]


if __name__ == "__main__":
    main()
