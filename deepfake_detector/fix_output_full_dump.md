## FILE DA SUA - DAY DU NOI DUNG
### 1. evaluation/evaluate.py
```python
"""Model evaluation entrypoint script."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    from data.dataset import DeepfakeDataset
    from evaluation.metrics import (
        compute_metrics,
        find_optimal_threshold,
        plot_confusion_matrix,
        plot_roc_curve,
    )
    from models.deepfake_model import DeepfakeDetector
except ImportError:
    from ..data.dataset import DeepfakeDataset
    from .metrics import (
        compute_metrics,
        find_optimal_threshold,
        plot_confusion_matrix,
        plot_roc_curve,
    )
    from ..models.deepfake_model import DeepfakeDetector

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    """Parse command line args."""
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path checkpoint. Neu bo qua, se tu tim trong checkpoint.save_dir.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Thu muc test data (co Real/ va Fake/). Neu bo qua, dung data.test_dir/val_dir/train_dir.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path train_config.yaml.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Thu muc luu ket qua danh gia.",
    )
    parser.add_argument(
        "--failure_top_k",
        type=int,
        default=50,
        help="So luong mau du doan sai nhat can luu trong failure report.",
    )
    parser.add_argument(
        "--save_failure_report",
        dest="save_failure_report",
        action="store_true",
        default=True,
        help="Bat xuat failure analysis report (JSON + CSV).",
    )
    parser.add_argument(
        "--no-save_failure_report",
        dest="save_failure_report",
        action="store_false",
        help="Tat xuat failure analysis report.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load config va merge model_config.yaml neu can."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay config: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise RuntimeError("Config YAML khong hop le.")

    if "model" not in config:
        fallback_model_cfg = path.parent / "model_config.yaml"
        if fallback_model_cfg.exists():
            with fallback_model_cfg.open("r", encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f) or {}
            if isinstance(model_cfg, dict) and "model" in model_cfg:
                config["model"] = model_cfg["model"]

    return config


def resolve_device(device_str: str) -> torch.device:
    """Chon device theo config + tinh san sang cua CUDA."""
    requested = str(device_str).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def infer_feat_dim(model_cfg: Dict[str, Any]) -> int:
    """Suy ra feat_dim tu model config."""
    if "feat_dim" in model_cfg:
        return int(model_cfg["feat_dim"])

    eff_cfg = model_cfg.get("efficientnet", {}) if isinstance(model_cfg, dict) else {}
    model_name = str(eff_cfg.get("model_name", "efficientnet_b4")).lower()
    feat_dim_map = {
        "efficientnet_b0": 1280,
        "efficientnet_b4": 1792,
    }
    return int(feat_dim_map.get(model_name, 1792))


def resolve_data_dir(arg_data_dir: str | None, config: Dict[str, Any]) -> str:
    """
    Resolve test data dir:
    - Uu tien --data_dir
    - fallback data.test_dir -> data.val_dir -> data.train_dir
    """
    if arg_data_dir:
        return str(arg_data_dir)

    data_cfg = config.get("data", {})
    for key in ("test_dir", "val_dir", "train_dir"):
        value = data_cfg.get(key)
        if value:
            return str(value)

    raise ValueError(
        "Khong xac dinh duoc data_dir. Hay truyen --data_dir hoac dat data.test_dir/val_dir/train_dir trong config."
    )


def _score_checkpoint(path: Path) -> Tuple[float, float]:
    """Tra ve (val_auc, epoch) de sap xep checkpoint."""
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return float("-inf"), float("-inf")

    if not isinstance(payload, dict):
        return float("-inf"), float("-inf")

    val_auc = payload.get("val_auc", float("-inf"))
    epoch = payload.get("epoch", float("-inf"))

    try:
        val_auc_f = float(val_auc)
    except Exception:
        val_auc_f = float("-inf")

    try:
        epoch_f = float(epoch)
    except Exception:
        epoch_f = float("-inf")

    return val_auc_f, epoch_f


def resolve_checkpoint_path(arg_checkpoint: str | None, config: Dict[str, Any]) -> Path:
    """
    Resolve checkpoint path:
    - Uu tien --checkpoint
    - fallback checkpoint.path
    - fallback checkpoint.save_dir (chon checkpoint co val_auc cao nhat)
    """
    if arg_checkpoint:
        ckpt = Path(arg_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt}")
        return ckpt

    checkpoint_cfg = config.get("checkpoint", {})
    explicit = checkpoint_cfg.get("path")
    if explicit:
        ckpt = Path(str(explicit))
        if not ckpt.exists():
            raise FileNotFoundError(f"Khong tim thay checkpoint.path: {ckpt}")
        return ckpt

    save_dir = Path(str(checkpoint_cfg.get("save_dir", "checkpoints")))
    if not save_dir.exists():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint.save_dir: {save_dir}. Hay truyen --checkpoint hoac cap nhat config."
        )

    candidates = sorted(save_dir.rglob("*.pth"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Khong tim thay file *.pth trong {save_dir}. Hay truyen --checkpoint hoac kiem tra training output."
        )

    scored = []
    for ckpt in candidates:
        val_auc, epoch = _score_checkpoint(ckpt)
        scored.append((val_auc, epoch, ckpt))

    scored.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    return scored[0][2]


class DeterministicTTATransform:
    """
    Transform deterministic cho test/TTA:
    PIL -> Tensor [C, H, W], da normalize theo ImageNet.
    """

    def __init__(
        self,
        img_size: int = 256,
        hflip: bool = False,
        center_crop_scale: float = 1.0,
    ) -> None:
        self.img_size = img_size
        self.resize_size = int(img_size * 1.14)
        self.hflip = hflip
        self.center_crop_scale = float(center_crop_scale)

    def __call__(self, image):
        image = image.convert("RGB")
        image = TF.resize(image, self.resize_size)

        # [FIX C2] TTA an toan cho forensic: chi dung bien doi hinh hoc nhe (crop/resize).
        crop_size = int(round(self.img_size * self.center_crop_scale))
        crop_size = max(1, min(crop_size, self.resize_size))
        image = TF.center_crop(image, crop_size)
        if crop_size != self.img_size:
            image = TF.resize(image, self.img_size)

        if self.hflip:
            image = TF.hflip(image)

        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor


def build_test_loader(
    data_dir: str,
    num_frames: int,
    transform,
    num_clips_eval: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DeepfakeDataset, DataLoader]:
    """Build test dataset + dataloader."""
    dataset = DeepfakeDataset(
        root_dir=data_dir,
        num_frames=num_frames,
        transform=transform,
        mode="test",
        # [FIX A1] Truyen num_clips_eval de kich hoat multi-clip path trong dataset.
        num_clips_eval=num_clips_eval,
    )

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(**loader_kwargs)
    return dataset, loader


def run_inference_once(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chay inference 1 lan tren toan bo test set.
    Yeu cau: torch.no_grad() + autocast.
    """
    model.eval()
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for frames, labels in loader:
            labels = labels.to(device, non_blocking=True).float().view(-1)
            batch_size = labels.shape[0]

            # [FIX A1] Ho tro ca [B, T, C, H, W] va [B, N, T, C, H, W].
            if frames.ndim == 6:
                num_clips = int(frames.shape[1])
                frames_flat = frames.reshape(
                    batch_size * num_clips,
                    frames.shape[2],
                    frames.shape[3],
                    frames.shape[4],
                    frames.shape[5],
                ).to(device, non_blocking=True)
                # [FIX A1] autocast chi bao quanh forward pass.
                with autocast(enabled=device.type == "cuda"):
                    logits_flat = model(frames_flat).view(-1)
                probs = torch.sigmoid(logits_flat.float()).view(batch_size, num_clips)
                probs = probs.max(dim=1).values
            elif frames.ndim == 5:
                frames = frames.to(device, non_blocking=True)
                # [FIX A1] autocast chi bao quanh forward pass.
                with autocast(enabled=device.type == "cuda"):
                    logits = model(frames).view(-1)
                probs = torch.sigmoid(logits.float())
            else:
                raise ValueError(
                    f"frames phai co ndim=5 hoac 6, nhan duoc ndim={frames.ndim}."
                )

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_scores = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return y_true, y_scores


def build_tta_transforms(img_size: int) -> List[DeterministicTTATransform]:
    """Tao TTA transforms forensic-safe (khong color jitter)."""
    return [
        DeterministicTTATransform(img_size=img_size),
        DeterministicTTATransform(img_size=img_size, hflip=True),
        DeterministicTTATransform(img_size=img_size, center_crop_scale=0.95),
        DeterministicTTATransform(img_size=img_size, center_crop_scale=1.05),
    ]


def print_metrics_table(metrics: Dict[str, float], title: str = "Metric Results") -> None:
    """In bang metrics ra terminal (tabulate neu co)."""
    rows = [
        ["AUC-ROC", metrics.get("auc_roc", float("nan"))],
        ["EER", metrics.get("eer", float("nan"))],
        ["EER Threshold", metrics.get("eer_threshold", float("nan"))],
        ["AP", metrics.get("ap", float("nan"))],
        [f"Accuracy@{metrics.get('threshold', float('nan')):.2f}", metrics.get("accuracy", float("nan"))],
        [f"Precision@{metrics.get('threshold', float('nan')):.2f}", metrics.get("precision", float("nan"))],
        [f"Recall@{metrics.get('threshold', float('nan')):.2f}", metrics.get("recall", float("nan"))],
        [f"F1@{metrics.get('threshold', float('nan')):.2f}", metrics.get("f1", float("nan"))],
    ]

    print(title)
    if tabulate is not None:
        printable_rows = [[name, f"{value:.6f}"] for name, value in rows]
        print(tabulate(printable_rows, headers=["Metric", "Value"], tablefmt="github"))
    else:
        for name, value in rows:
            print(f"- {name:16s}: {value:.6f}")


def run_tta_inference(
    model: torch.nn.Module,
    data_dir: str,
    tta_transforms: List[DeterministicTTATransform],
    num_frames: int,
    num_clips_eval: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Chay TTA va average score cho 1 split."""
    tta_scores: List[np.ndarray] = []
    y_true_ref: np.ndarray | None = None
    video_paths_ref: List[str] = []
    num_videos = 0

    for tta_idx, transform in enumerate(tta_transforms, start=1):
        dataset, loader = build_test_loader(
            data_dir=data_dir,
            num_frames=num_frames,
            transform=transform,
            num_clips_eval=num_clips_eval,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if tta_idx == 1:
            num_videos = len(dataset)
            video_paths_ref = extract_video_paths(dataset)

        y_true, y_scores = run_inference_once(model=model, loader=loader, device=device)
        if y_true_ref is None:
            y_true_ref = y_true
        elif not np.array_equal(y_true_ref, y_true):
            raise RuntimeError(f"Nhan giua cac TTA pass cua {split_name} khong khop thu tu.")

        tta_scores.append(y_scores)
        print(f"[{split_name} TTA {tta_idx}/{len(tta_transforms)}] done.")

    if y_true_ref is None:
        raise RuntimeError(f"Khong thu duoc ket qua split {split_name}.")

    y_scores_avg = np.mean(np.stack(tta_scores, axis=0), axis=0)
    return y_true_ref, y_scores_avg, video_paths_ref, num_videos


def extract_video_paths(dataset: DeepfakeDataset) -> List[str]:
    """
    Lay danh sach video_path theo thu tu dataset.
    Fallback ve index neu dataset khong co metadata samples.
    """
    if hasattr(dataset, "samples"):
        try:
            return [str(sample["video_dir"]) for sample in dataset.samples]
        except Exception:
            pass
    return [f"sample_{idx}" for idx in range(len(dataset))]


def build_failure_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    video_paths: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Tao danh sach cac mau du doan sai nhat (FP/FN) de failure analysis.
    """
    if len(video_paths) != len(y_true):
        video_paths = [f"sample_{idx}" for idx in range(len(y_true))]

    rows: List[Dict[str, Any]] = []
    for idx, (yt, ys, yp) in enumerate(zip(y_true, y_scores, y_pred)):
        yt_i = int(yt)
        yp_i = int(yp)
        if yt_i == yp_i:
            continue

        score = float(ys)
        if yt_i == 1:
            margin = 1.0 - score
            err_type = "FN"
        else:
            margin = score
            err_type = "FP"

        rows.append(
            {
                "video_path": str(video_paths[idx]),
                "y_true": yt_i,
                "y_score": score,
                "y_pred": yp_i,
                "error_margin": float(margin),
                "error_type": err_type,
            }
        )

    rows.sort(key=lambda item: item["error_margin"], reverse=True)
    top_k = max(0, int(top_k))
    if top_k == 0:
        return rows
    return rows[:top_k]


def save_failure_analysis_report(
    failure_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Luu failure analysis ra JSON + CSV."""
    json_path = output_dir / "failure_report.json"
    csv_path = output_dir / "failure_report.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(failure_rows, f, indent=2, ensure_ascii=False)

    fieldnames = ["video_path", "y_true", "y_score", "y_pred", "error_margin", "error_type"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in failure_rows:
            writer.writerow(row)

    return json_path, csv_path


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    input_cfg = model_cfg.get("input", {})
    efficientnet_cfg = model_cfg.get("efficientnet", {})
    transformer_cfg = model_cfg.get("transformer", {})

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, config)
    test_data_dir = resolve_data_dir(args.data_dir, config)
    val_data_dir_cfg = data_cfg.get("val_dir")
    val_data_dir = str(val_data_dir_cfg) if val_data_dir_cfg else None

    device = resolve_device(str(training_cfg.get("device", "cuda")))
    batch_size = int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 8)))
    num_workers = int(data_cfg.get("num_workers", 4))
    # [FIX A1] Doc num_clips_eval tu config de khop pipeline validate().
    num_clips_eval = int(data_cfg.get("num_clips_eval", 3))
    num_frames = int(input_cfg.get("num_frames", model_cfg.get("num_frames", 10)))
    img_size = int(input_cfg.get("img_size", model_cfg.get("image_size", 256)))

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
    model.to(device)

    ckpt_info = model.load_checkpoint(str(checkpoint_path))
    model.eval()

    requested_pin_memory = bool(data_cfg.get("pin_memory", True))
    pin_memory = requested_pin_memory and device.type == "cuda"

    print("========== Evaluation Summary ==========")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Test dir:     {test_data_dir}")
    print(f"Val dir:      {val_data_dir if val_data_dir is not None else 'None'}")
    print(f"Device:       {device}")
    print(f"Batch size:   {batch_size}")
    print(f"Num workers:  {num_workers}")
    print(f"Num clips:    {num_clips_eval}")
    print("========================================")

    tta_transforms = build_tta_transforms(img_size=img_size)
    # [FIX C3] Tune threshold tren val set (neu co), sau do ap dung cho test.
    if val_data_dir is not None:
        y_true_val, y_scores_val, _, _ = run_tta_inference(
            model=model,
            data_dir=val_data_dir,
            tta_transforms=tta_transforms,
            num_frames=num_frames,
            num_clips_eval=num_clips_eval,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device,
            split_name="VAL",
        )
        optimal_threshold, best_val_f1 = find_optimal_threshold(y_true_val, y_scores_val)
        print(f"[VAL] Optimal threshold theo F1: {optimal_threshold:.2f} (F1={best_val_f1:.4f})")
    else:
        optimal_threshold = 0.5
        best_val_f1 = float("nan")
        print("[WARN] Khong co val_dir trong config, fallback threshold=0.50.")

    y_true_test, y_scores_test, video_paths_ref, num_videos = run_tta_inference(
        model=model,
        data_dir=test_data_dir,
        tta_transforms=tta_transforms,
        num_frames=num_frames,
        num_clips_eval=num_clips_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
        split_name="TEST",
    )

    metrics_default = compute_metrics(y_true_test, y_scores_test, threshold=0.5)
    metrics_optimal = compute_metrics(y_true_test, y_scores_test, threshold=optimal_threshold)
    print_metrics_table(metrics_default, title="Metric Results (threshold=0.50)")
    print_metrics_table(
        metrics_optimal,
        title=f"Metric Results (threshold=optimal={optimal_threshold:.2f})",
    )
    y_pred = (y_scores_test >= float(optimal_threshold)).astype(np.int64)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_payload = {
        "checkpoint": str(checkpoint_path),
        "test_data_dir": str(test_data_dir),
        "val_data_dir": str(val_data_dir) if val_data_dir is not None else None,
        "num_test_videos": int(num_videos),
        "device": str(device),
        "num_clips_eval": int(num_clips_eval),
        "checkpoint_meta": {
            "epoch": ckpt_info.get("epoch"),
            "val_auc": ckpt_info.get("val_auc"),
            "missing_keys": ckpt_info.get("missing_keys", []),
            "unexpected_keys": ckpt_info.get("unexpected_keys", []),
        },
        "thresholding": {
            "default_threshold": 0.5,
            "optimal_threshold_from_val": float(optimal_threshold),
            "best_val_f1": float(best_val_f1),
        },
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
    }
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    roc_path = output_dir / "roc_curve.png"
    cm_path = output_dir / "confusion_matrix.png"
    plot_roc_curve(y_true_test, y_scores_test, save_path=str(roc_path))
    plot_confusion_matrix(y_true_test, y_pred, save_path=str(cm_path))

    print(f"Saved results to: {results_path}")
    print(f"Saved ROC curve to: {roc_path}")
    print(f"Saved confusion matrix to: {cm_path}")

    if args.save_failure_report:
        failure_rows = build_failure_analysis(
            y_true=y_true_test,
            y_scores=y_scores_test,
            y_pred=y_pred,
            video_paths=video_paths_ref,
            top_k=int(args.failure_top_k),
        )
        failure_json, failure_csv = save_failure_analysis_report(
            failure_rows=failure_rows,
            output_dir=output_dir,
        )
        print(f"Saved failure report JSON to: {failure_json}")
        print(f"Saved failure report CSV to: {failure_csv}")


if __name__ == "__main__":
    main()
```

### 2. training/train.py
```python
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
    )

    if val_dir is None:
        val_base = DeepfakeDataset(
            root_dir=str(train_dir),
            num_frames=num_frames,
            transform=None,
            clip_transform=val_clip_transform,
            mode="val",
            num_clips_eval=num_clips_eval,
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

    if bool(efficientnet_cfg.get("freeze_at_start", False)):
        model.backbone.freeze_backbone()

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
        alpha=float(loss_cfg.get("alpha", training_cfg.get("focal_alpha", 0.25))),
        smoothing=float(loss_cfg.get("smoothing", training_cfg.get("label_smoothing", 0.1))),
    )

    checkpoint_path = build_checkpoint_path(config)

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
        },
    )

    resume_info = maybe_resume(args.resume, model, optimizer, scheduler)
    if math.isfinite(resume_info["best_val_auc"]):
        trainer.best_val_auc = float(resume_info["best_val_auc"])
        trainer.best_epoch = int(resume_info["start_epoch"]) - 1

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
    trainer.fit(num_epochs=remaining_epochs)


if __name__ == "__main__":
    main()
```

### 3. data/augmentation.py
```python
"""Data augmentation pipeline definitions."""

from __future__ import annotations

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
            A.ToGray(p=gray_p, num_output_channels=3),
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

```

### 4. clean_processed_dataset.py
```python
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

# â”€â”€ Defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ Frame-level helpers (must be top-level for pickling) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Folder collection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Checkpoint helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Parallel scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # Each worker audits one complete folder (CPU-bound â†’ ProcessPoolExecutor)
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


# â”€â”€ Parallel delete â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ SCAN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ DELETE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
```

### 5. training/trainer.py
```python
"""Trainer class and training loop controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    """
    Quan ly vong lap huan luyen cho DeepfakeDetector bang PyTorch thuan.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion: torch.nn.Module,
        device: torch.device | str,
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = torch.device(device)
        self.config = config

        self.model.to(self.device)

        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.patience = int(self.config.get("patience", 5))
        self.min_delta = float(self.config.get("min_delta", 0.0))
        self.checkpoint_path = self.config.get(
            "checkpoint_path", "checkpoints/best_model.pth"
        )
        self.load_best_at_end = bool(self.config.get("load_best_at_end", True))

        # Chi bat AMP khi train tren CUDA.
        self.use_amp = bool(self.config.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_val_auc = float("-inf")
        self.best_epoch = -1
        self.nonfinite_loss_steps = 0
        self.nonfinite_grad_steps = 0

    def _has_finite_gradients(self) -> bool:
        """Kiem tra tat ca gradient hien tai deu finite."""
        for param in self.model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
        return True

    def train_one_epoch(self) -> Dict[str, float]:
        """
        Huan luyen 1 epoch voi AMP + gradient clipping.
        Tra ve {'loss': float, 'acc': float}.
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for frames, targets in self.train_loader:
            frames = frames.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).float().view(-1)
            batch_size = targets.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(frames).view(-1)  # [B]

            # [FIX C1] Tinh loss ngoai autocast de criterion xu ly o fp32 on dinh hon.
            logits_fp32 = logits.float()
            loss = self.criterion(logits_fp32, targets)

            if not torch.isfinite(loss):
                self.nonfinite_loss_steps += 1
                if self.nonfinite_loss_steps <= 5 or self.nonfinite_loss_steps % 50 == 0:
                    print(
                        f"[WARN] Non-finite loss o train step, bo qua batch. "
                        f"count={self.nonfinite_loss_steps}"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            # Can unscale truoc khi clip gradient:
            # Neu clip tren scaled gradients, nguong clip se bi sai lech
            # (gradient dang bi nhan he so scale), day la loi AMP rat hay gap.
            self.scaler.unscale_(self.optimizer)

            if not self._has_finite_gradients():
                self.nonfinite_grad_steps += 1
                if self.nonfinite_grad_steps <= 5 or self.nonfinite_grad_steps % 50 == 0:
                    print(
                        f"[WARN] Non-finite gradient o train step, bo qua optimizer step. "
                        f"count={self.nonfinite_grad_steps}"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue

            clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                correct = (preds == targets.long()).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += int(correct)
            total_samples += batch_size

        mean_loss = total_loss / max(total_samples, 1)
        mean_acc = total_correct / max(total_samples, 1)
        return {
            "loss": float(mean_loss),
            "acc": float(mean_acc),
            "skipped_nonfinite_loss": float(self.nonfinite_loss_steps),
            "skipped_nonfinite_grad": float(self.nonfinite_grad_steps),
        }

    def validate(self) -> Dict[str, float]:
        """
        Danh gia tren tap val voi ho tro multi-clip:
        - Loss monitor/early stopping tinh tren tat ca clips.
        - Metrics (AUC/F1/ACC) tinh o cap video sau max-pooling theo clip.
        """
        self.model.eval()

        total_loss_sum = 0.0
        total_clip_samples = 0
        all_video_probs = []
        all_video_targets = []

        with torch.inference_mode():
            for frames, targets in self.val_loader:
                targets = targets.to(self.device, non_blocking=True).float().view(-1)
                batch_size = targets.size(0)

                # Ho tro ca batch [B, T, C, H, W] va [B, N, T, C, H, W].
                if frames.ndim == 6:
                    num_clips = int(frames.shape[1])
                    frames_flat = frames.view(
                        batch_size * num_clips,
                        frames.shape[2],
                        frames.shape[3],
                        frames.shape[4],
                        frames.shape[5],
                    )
                elif frames.ndim == 5:
                    num_clips = 1
                    frames_flat = frames
                else:
                    raise ValueError(
                        f"Val batch frames phai co ndim 5 hoac 6, nhan duoc {frames.ndim}."
                    )

                frames_flat = frames_flat.to(self.device, non_blocking=True)
                labels_flat = targets.repeat_interleave(num_clips)

                # Chi dung autocast cho forward pass.
                with autocast(enabled=self.use_amp):
                    logits_flat = self.model(frames_flat).view(-1)  # [B*num_clips]

                logits_flat = logits_flat.float()
                labels_flat = labels_flat.float()

                # Criterion hien tai tra ve mean loss, scale len tong de monitor ro rang.
                loss_mean = self.criterion(logits_flat, labels_flat)
                loss_sum = loss_mean * labels_flat.numel()
                total_loss_sum += float(loss_sum.item())
                total_clip_samples += int(labels_flat.numel())

                probs = torch.sigmoid(logits_flat).view(batch_size, num_clips)
                # Asymmetric task: chi can 1 clip co artifact -> video fake.
                video_probs, _ = probs.max(dim=1)

                all_video_probs.append(video_probs.cpu())
                all_video_targets.append(targets.cpu())

        mean_loss = total_loss_sum / max(total_clip_samples, 1)

        if len(all_video_targets) == 0:
            return {"loss": float(mean_loss), "auc": 0.0, "f1": 0.0, "acc": 0.0}

        y_score = torch.cat(all_video_probs, dim=0).numpy()
        y_true = torch.cat(all_video_targets, dim=0).numpy()
        y_pred = (y_score >= 0.5).astype(int)

        # roc_auc_score can loi neu val set chi co 1 class.
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = 0.5
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        return {"loss": float(mean_loss), "auc": auc, "f1": f1, "acc": acc}

    def _save_best_checkpoint(self, epoch: int, val_auc: float) -> None:
        """Luu checkpoint khi val_auc cai thien."""
        ckpt_path = Path(self.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        model_for_ckpt = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_for_ckpt, "save_checkpoint"):
            model_for_ckpt.save_checkpoint(str(ckpt_path), epoch=epoch, val_auc=val_auc)
            return

        payload = {
            "epoch": int(epoch),
            "val_auc": float(val_auc),
            "model_state_dict": model_for_ckpt.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
        }
        torch.save(payload, ckpt_path)

    def _load_best_checkpoint_into_model(self) -> bool:
        """
        Nap best checkpoint vao model sau khi fit (optional).
        """
        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            return False

        model_for_load = self.model.module if hasattr(self.model, "module") else self.model
        try:
            if hasattr(model_for_load, "load_checkpoint"):
                model_for_load.load_checkpoint(str(ckpt_path))
                return True

            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if not isinstance(checkpoint, dict):
                return False
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if not isinstance(state_dict, dict):
                return False
            model_for_load.load_state_dict(state_dict, strict=False)
            return True
        except Exception as exc:
            print(f"[WARN] Khong load duoc best checkpoint vao cuoi training: {exc}")
            return False

    def fit(self, num_epochs: int) -> Dict[str, float]:
        """
        Chay vong lap train/val nhieu epoch voi early stopping.
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs phai > 0.")

        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            # Update scheduler sau khi co val metric.
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["auc"])
                else:
                    self.scheduler.step()

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | "
                f"val_f1={val_metrics.get('f1', 0.0):.4f} | "
                f"val_acc={val_metrics.get('acc', 0.0):.4f} | "
                f"lr={current_lr:.6e}"
            )

            # Chi luu checkpoint khi AUC co y nghia (tot hon random baseline).
            if val_metrics["auc"] > (self.best_val_auc + self.min_delta) and val_metrics["auc"] > 0.5:
                self.best_val_auc = val_metrics["auc"]
                self.best_epoch = epoch
                epochs_no_improve = 0
                self._save_best_checkpoint(epoch=epoch, val_auc=self.best_val_auc)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"Early stopping tai epoch {epoch}. "
                        f"Best val_auc={self.best_val_auc:.4f} tai epoch {self.best_epoch}."
                    )
                    break

        if self.load_best_at_end and self.best_epoch >= 1:
            loaded = self._load_best_checkpoint_into_model()
            if loaded:
                print(
                    f"Loaded best checkpoint sau khi fit: {self.checkpoint_path} "
                    f"(epoch={self.best_epoch}, val_auc={self.best_val_auc:.4f})."
                )

        return {"best_val_auc": float(self.best_val_auc), "best_epoch": int(self.best_epoch)}

```

### 6. evaluation/metrics.py
```python
"""Evaluation metrics utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _to_numpy_1d(array_like) -> np.ndarray:
    """Convert input ve numpy 1D array."""
    arr = np.asarray(array_like).reshape(-1)
    return arr


def _compute_eer_from_roc(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float]:
    """
    Tinh EER tu duong ROC.
    EER la diem ma FAR (FPR) = FRR (1 - TPR).
    """
    fnr = 1.0 - tpr
    diff = fpr - fnr

    # Tim giao diem (co doi dau) de noi suy tuyen tinh cho EER min hon.
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if sign_changes.size > 0:
        i = int(sign_changes[0])
        x1, x2 = diff[i], diff[i + 1]
        y1, y2 = fpr[i], fpr[i + 1]
        t1, t2 = thresholds[i], thresholds[i + 1]

        # He so noi suy, bao ve truong hop x1 == x2.
        denom = (x1 - x2)
        if abs(denom) < 1e-12:
            w = 0.5
        else:
            w = x1 / denom
        w = float(np.clip(w, 0.0, 1.0))

        eer = float(y1 + w * (y2 - y1))
        eer_threshold = float(t1 + w * (t2 - t1))
        return eer, eer_threshold

    # Neu khong co giao diem ro rang, lay diem gan nhat.
    idx = int(np.argmin(np.abs(diff)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def find_optimal_threshold(y_true, y_scores) -> Tuple[float, float]:
    """
    Tim threshold toi uu theo F1 tren khoang [0.1, 0.9], buoc 0.01.
    Returns: (best_threshold, best_f1)
    """
    # [FIX C3] Ho tro tune threshold tren val truoc khi apply sang test.
    y_true_arr = _to_numpy_1d(y_true).astype(np.int64)
    y_scores_arr = _to_numpy_1d(y_scores).astype(np.float64)
    if y_true_arr.shape[0] != y_scores_arr.shape[0]:
        raise ValueError("y_true va y_scores phai cung so phan tu.")
    if y_true_arr.size == 0:
        raise ValueError("y_true/y_scores khong duoc rong.")

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.1, 0.901, 0.01):
        y_pred = (y_scores_arr >= float(threshold)).astype(np.int64)
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def compute_metrics(y_true, y_scores, threshold: float = 0.5) -> Dict[str, float]:
    """
    Tinh cac metrics cho deepfake detection.

    Args:
        y_true: numpy array nhan that (0/1).
        y_scores: numpy array score/xac suat du doan cho class Fake.

    Returns:
        Dict gom:
        - auc_roc
        - eer
        - ap
        - accuracy
        - precision
        - recall
        - f1
        - eer_threshold
    """
    y_true_arr = _to_numpy_1d(y_true).astype(np.int64)
    y_scores_arr = _to_numpy_1d(y_scores).astype(np.float64)

    if y_true_arr.shape[0] != y_scores_arr.shape[0]:
        raise ValueError(
            "y_true va y_scores phai cung so phan tu, "
            f"nhan duoc {y_true_arr.shape[0]} va {y_scores_arr.shape[0]}."
        )
    if y_true_arr.size == 0:
        raise ValueError("y_true/y_scores khong duoc rong.")

    # [FIX C3] Cho phep custom threshold thay vi khoa cung 0.5.
    threshold = float(threshold)
    y_pred = (y_scores_arr >= threshold).astype(np.int64)

    # EER quan trong hon accuracy trong deepfake detection vi:
    # - Accuracy de bi "ao" khi class imbalance (doan da so class van cho acc cao).
    # - Forensics thuong quan tam can bang FAR/FRR va chi phi false alarm / miss khac nhau.
    #   EER phan anh diem can bang loi giua hai loai sai nay ro rang hon.
    try:
        auc_roc = float(roc_auc_score(y_true_arr, y_scores_arr))
    except ValueError:
        auc_roc = float("nan")

    try:
        ap = float(average_precision_score(y_true_arr, y_scores_arr))
    except ValueError:
        ap = float("nan")

    try:
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_scores_arr)
        eer, eer_threshold = _compute_eer_from_roc(fpr, tpr, thresholds)
    except ValueError:
        eer = float("nan")
        eer_threshold = float("nan")

    accuracy = float(accuracy_score(y_true_arr, y_pred))
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    return {
        "auc_roc": auc_roc,
        "eer": eer,
        "ap": ap,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer_threshold": eer_threshold,
        "threshold": threshold,
    }


def plot_roc_curve(y_true, y_scores, save_path: str) -> None:
    """
    Ve va luu ROC curve ra file.
    """
    y_true_arr = _to_numpy_1d(y_true).astype(np.int64)
    y_scores_arr = _to_numpy_1d(y_scores).astype(np.float64)
    if y_true_arr.shape[0] != y_scores_arr.shape[0]:
        raise ValueError("y_true va y_scores phai cung so phan tu.")

    fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)
    auc_roc = roc_auc_score(y_true_arr, y_scores_arr)

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_roc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Deepfake Detection")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_file, dpi=200)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    """
    Ve va luu confusion matrix ra file.
    """
    y_true_arr = _to_numpy_1d(y_true).astype(np.int64)
    y_pred_arr = _to_numpy_1d(y_pred).astype(np.int64)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true va y_pred phai cung so phan tu.")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    class_names = ["Real (0)", "Fake (1)"]
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix - Deepfake Detection",
    )

    # Ghi so luong mau trong tung o.
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_file, dpi=200)
    plt.close(fig)

```

### 7. configs/aug_config.yaml
```yaml
# Augmentation config tach rieng khoi train_config.

preprocess:
  crop_margin: 0.3
  target_size: 256
  min_face_confidence: 0.6
  min_face_size_ratio: 0.05      # bo qua mat qua nho (< 5% dien tich frame)

# [FIX B1] Cau truc chuan de train.py/augmentation.py doc truc tiep.
augmentation:
  horizontal_flip_p: 0.5

  color_jitter:
    brightness: 0.15
    contrast: 0.15
    saturation: 0.1
    hue: 0.0
    p: 0.8

  gaussian_blur:
    blur_limit: [3, 7]
    p: 0.3

  gaussian_noise:
    var_limit: [10, 50]
    p: 0.2

  # JPEG compression augmentation dac biet quan trong voi deepfake detection:
  # deepfake artifacts thuong bi lo ro hon sau pipeline nen/luu lai JPEG.
  # Mo phong nhieu muc quality giup model robust hon voi du lieu thuc te
  # (social media re-encode, upload/download, trich xuat khung hinh).
  jpeg_compression:
    quality_lower: 60
    quality_upper: 100
    p: 0.3

  random_grayscale_p: 0.05

  coarse_dropout:
    max_holes: 4
    max_height: 32
    max_width: 32
    p: 0.2

val_augmentation:
  # Chi resize + normalize, khong dung random augment de danh gia on dinh.
  resize:
    image_size: 256
    resize_scale: 1.14
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### 8. configs/train_config.yaml
```yaml
# Cau hinh tong the cho qua trinh train
experiment:
  name: "dfdc_efficientnet_transformer_v1"
  seed: 42
  deterministic: true             # torch.backends.cudnn.deterministic

data:
  train_dir: "/root/deepfake_detector/processed_faces/train"
  val_dir: "/root/deepfake_detector/processed_faces/val"
  test_dir: "/root/deepfake_detector/processed_faces/test"
  val_split: 0.0                  # da co val set rieng, khong auto split tu train
  auto_split: false
  num_clips_eval: 3               # so clips dung cho val/test inference
  num_workers: 8
  # [FIX B3] workers rieng cho val/test multi-clip (I/O nang hon train).
  eval_num_workers: 12
  pin_memory: true
  prefetch_factor: 2

training:
  batch_size: 16
  num_epochs: 50
  precision: 16                   # fp16 AMP
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2      # gradient accumulation khi GPU nho
  patience: 7                     # early stopping
  min_delta: 0.001                # improvement toi thieu de tinh la cai thien

optimizer:
  name: "AdamW"
  lr_backbone: 1.0e-5             # learning rate cho EfficientNet (nho vi pretrained)
  lr_head: 1.0e-4                 # learning rate cho Transformer head
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  name: "CosineAnnealingLR"
  T_max: 50                       # = num_epochs
  eta_min: 1.0e-7                 # learning rate toi thieu
  warmup_epochs: 3                # linear warmup truoc khi cosine

loss:
  name: "FocalLossWithSmoothing"
  gamma: 2.0
  alpha: 0.25
  smoothing: 0.1

checkpoint:
  save_dir: "/root/deepfake_detector/checkpoints"
  save_top_k: 3                   # giu 3 checkpoint tot nhat
  monitor: "val_auc"
  mode: "max"
  save_last: true

logging:
  use_wandb: false                # bat khi co account W&B
  wandb_project: "deepfake-detector"
  log_every_n_steps: 10

hardware:
  gpus: 1
  strategy: "auto"                # "ddp" neu multi-GPU
```

