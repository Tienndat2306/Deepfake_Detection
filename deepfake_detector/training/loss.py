"""Loss functions for training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithSmoothing(nn.Module):
    """
    Ket hop Focal Loss + Label Smoothing cho bai toan binary classification.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,  # [FIX-BUG-7]
        smoothing: float = 0.1,
        consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma phai >= 0.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha phai nam trong [0, 1].")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing phai nam trong [0, 1).")
        if consistency_weight < 0.0:
            raise ValueError("consistency_weight phai >= 0.")

        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.consistency_weight = float(consistency_weight)

    @staticmethod
    def _temporal_consistency_loss(clip_frame_logits: torch.Tensor) -> torch.Tensor:
        """
        Loss nhe de ep du doan frame trong cung clip dong nhat.
        clip_frame_logits: [B, T] hoac [B, T, 1].
        """
        if clip_frame_logits.ndim == 3 and clip_frame_logits.shape[-1] == 1:
            clip_frame_logits = clip_frame_logits.squeeze(-1)
        if clip_frame_logits.ndim != 2:
            raise ValueError("clip_frame_logits phai co shape [B, T] hoac [B, T, 1].")

        probs = torch.sigmoid(clip_frame_logits.float())
        center = probs.mean(dim=1, keepdim=True)
        return torch.mean((probs - center) ** 2)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        clip_frame_logits: torch.Tensor | None = None,
        consistency_weight: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Tensor [B], raw logits (chua sigmoid).
            targets: Tensor [B], gia tri 0/1 kieu float.
            clip_frame_logits: Tensor optional [B, T] hoac [B, T, 1].
            consistency_weight: Override tam thoi cho trong luong consistency loss.
        Returns:
            Scalar loss.
        """
        logits = logits.float().view(-1)
        targets = targets.float().view(-1)

        if logits.shape != targets.shape:
            raise ValueError(
                f"logits va targets phai cung shape, nhan duoc {logits.shape} va {targets.shape}."
            )

        # 1) Label smoothing:
        # y_smooth = y * (1 - eps) + 0.5 * eps
        # Muc tieu: giam over-confident prediction, on dinh train hon.
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # 2) BCE voi logits (khong reduction de xu ly tung sample).
        ce = F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            reduction="none",
        )

        # 3) Focal weight: (1 - p_t)^gamma
        # p_t la xac suat cua nhan dung (sau smoothing).
        probs = torch.sigmoid(logits)
        p_t = probs * targets_smooth + (1.0 - probs) * (1.0 - targets_smooth)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # 4) Alpha_t cho class balancing:
        # alpha cho positive class, (1-alpha) cho negative class.
        # Dung targets goc (0/1) de giu y nghia can bang lop ro rang.
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # 5) Loss phan loai.
        loss = (alpha_t * focal_weight * ce).mean()

        # 6) Optional temporal consistency hook (mac dinh tat).
        weight = self.consistency_weight if consistency_weight is None else float(consistency_weight)
        if clip_frame_logits is not None and weight > 0.0:
            cons = self._temporal_consistency_loss(clip_frame_logits)
            loss = loss + (weight * cons)
        return loss


if __name__ == "__main__":
    # Unit test nho: kiem tra loss chay duoc va tra ve scalar.
    criterion = FocalLossWithSmoothing(gamma=2.0, alpha=0.75, smoothing=0.1)  # [FIX-BUG-7]
    dummy_logits = torch.tensor([0.2, -1.1, 2.3, 0.0], dtype=torch.float32)
    dummy_targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    dummy_loss = criterion(dummy_logits, dummy_targets)
    assert dummy_loss.ndim == 0, "Loss phai la scalar tensor (ndim = 0)."
    print("Unit test passed. Loss =", float(dummy_loss.item()))

