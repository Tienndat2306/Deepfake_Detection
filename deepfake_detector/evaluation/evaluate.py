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
