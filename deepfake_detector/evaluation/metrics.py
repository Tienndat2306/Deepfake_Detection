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

