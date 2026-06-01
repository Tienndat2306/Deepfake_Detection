"""Main deepfake model that combines backbone + head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

try:
    from .efficientnet import EfficientNetExtractor
    from .transformer_head import TransformerHead
except ImportError:
    from models.efficientnet import EfficientNetExtractor
    from models.transformer_head import TransformerHead


class DeepfakeDetector(nn.Module):
    """
    Mo hinh deepfake detection hoan chinh:
    EfficientNetExtractor + TransformerHead.
    """

    def __init__(
        self,
        feat_dim: int = 1792,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_frames: int = 10,
        use_global_context_pooling: bool = False,
        drop_path_rate: float = 0.0,
        use_relative_pos_bias: bool = False,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = EfficientNetExtractor(
            dropout_p=dropout,
            use_global_context_pooling=use_global_context_pooling,
            pretrained=pretrained_backbone,
        )
        if self.backbone.feature_dim != feat_dim:
            raise ValueError(
                f"feat_dim={feat_dim} khong khop voi backbone.feature_dim="
                f"{self.backbone.feature_dim}."
            )

        self.head = TransformerHead(
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            num_frames=num_frames,
            drop_path_rate=drop_path_rate,
            use_relative_pos_bias=use_relative_pos_bias,
        )
        self.num_frames = num_frames
        self.feat_dim = feat_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W]
        Returns:
            logits: [B]
        """
        if frames.ndim != 5:
            raise ValueError("frames phai co shape [B, T, C, H, W].")

        batch_size, seq_len, channels, height, width = frames.shape
        if seq_len > self.num_frames:
            raise ValueError(
                f"T={seq_len} vuot num_frames={self.num_frames}. "
                "Hay tang num_frames khi khoi tao model."
            )

        # 1) [B, T, C, H, W] -> [B*T, C, H, W]
        x = frames.reshape(batch_size * seq_len, channels, height, width)

        # 2) Qua EfficientNetExtractor -> [B*T, 1792]
        features = self.backbone(x)

        # 3) [B*T, 1792] -> [B, T, 1792]
        features = features.reshape(batch_size, seq_len, self.feat_dim)

        # 4) Qua TransformerHead -> logit [B]
        logits = self.head(features)
        return logits

    def get_optimizer_groups(
        self, lr_backbone: float = 1e-5, lr_head: float = 1e-4
    ) -> List[Dict[str, Any]]:
        """
        Tra ve param groups de dat learning rate khac nhau cho backbone va head.

        Tai sao can LR khac nhau:
        - Backbone la model pretrained (da hoc dac trung tong quat), nen thuong can LR nho
          de fine-tune nhe, tranh pha vo tri thuc da hoc.
        - Head thuong khoi tao moi, can LR lon hon de hoc nhanh theo bai toan deepfake.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        groups: List[Dict[str, Any]] = []
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": lr_backbone,
                    "name": "backbone",
                }
            )
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": lr_head,
                    "name": "head",
                }
            )

        if not groups:
            raise RuntimeError(
                "Khong co parameter nao requires_grad=True de tao optimizer groups."
            )
        return groups

    def get_optim_groups(
        self, lr_backbone: float = 1e-5, lr_head: float = 1e-4
    ) -> List[Dict[str, Any]]:
        """
        Alias backward-compatible cho naming convention khac.
        """
        return self.get_optimizer_groups(lr_backbone=lr_backbone, lr_head=lr_head)

    def save_checkpoint(self, path: str, epoch: int, val_auc: float) -> None:
        """
        Luu checkpoint model.
        """
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": int(epoch),
            "val_auc": float(val_auc),
            "model_state_dict": self.state_dict(),
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load checkpoint vao model.
        Tra ve metadata de trainer co the tiep tuc huan luyen/evaluate.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint khong hop le (khong phai dict).")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError("state_dict trong checkpoint khong hop le.")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        return {
            "epoch": checkpoint.get("epoch"),
            "val_auc": checkpoint.get("val_auc"),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }

