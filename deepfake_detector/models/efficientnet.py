"""EfficientNet backbone wrapper."""

from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNetExtractor(nn.Module):
    """
    Wrapper EfficientNet-B4 de dung lam feature extractor.
    """

    def __init__(
        self,
        dropout_p: float = 0.3,
        use_global_context_pooling: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.use_global_context_pooling = bool(use_global_context_pooling)

        # num_classes=0: bo classification head mac dinh cua timm, model tra ve feature.
        # global_pool='avg': ap dung global average pooling de dau ra la vector co do dai co dinh.
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="" if self.use_global_context_pooling else "avg",
        )

        self.feature_dim = int(getattr(self.backbone, "num_features", 1792))
        if self.feature_dim != 1792:
            raise ValueError(
                f"EfficientNet-B4 expected feature_dim=1792, got {self.feature_dim}."
            )

        # Optional global-context pooling: tron avg/max pooling roi projection ve feature_dim.
        self.context_proj = (
            nn.Linear(self.feature_dim * 2, self.feature_dim)
            if self.use_global_context_pooling
            else None
        )

        # Dropout sau global pool de regularize dac trung truoc khi dua sang head tiep theo.
        self.dropout = nn.Dropout(p=dropout_p)

    def _get_blocks(self) -> List[nn.Module]:
        """
        Lay danh sach cac block/stage trong backbone.blocks.
        """
        if not hasattr(self.backbone, "blocks"):
            raise AttributeError("Backbone khong co thuoc tinh 'blocks'.")
        return list(getattr(self.backbone, "blocks").children())

    def freeze_backbone(self) -> None:
        """
        Dong bang toan bo backbone (tat ca tham so requires_grad=False).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """
        Mo khoa n block cuoi trong backbone.blocks.
        Cach lam:
        - Dong bang toan bo backbone.
        - Chi mo khoa n block cuoi de fine-tune theo giai doan.
        """
        if n < 0:
            raise ValueError("n phai >= 0.")

        self.freeze_backbone()
        if n == 0:
            return

        blocks = self._get_blocks()
        if len(blocks) == 0:
            raise RuntimeError("Khong tim thay block nao trong backbone.blocks.")

        n = min(n, len(blocks))
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B*T, C, H, W]
        Output:
            features: [B*T, 1792]
        """
        if self.use_global_context_pooling:
            if hasattr(self.backbone, "forward_features"):
                features = self.backbone.forward_features(x)
            else:
                features = self.backbone(x)

            if features.ndim == 4:
                avg_pool = torch.flatten(F.adaptive_avg_pool2d(features, output_size=1), 1)
                max_pool = torch.flatten(F.adaptive_max_pool2d(features, output_size=1), 1)
                fused = torch.cat([avg_pool, max_pool], dim=1)
                if self.context_proj is None:
                    raise RuntimeError("context_proj chua duoc khoi tao.")
                features = self.context_proj(fused)
            elif features.ndim > 2:
                features = torch.flatten(features, start_dim=1)
        else:
            features = self.backbone(x)
            if features.ndim > 2:
                features = torch.flatten(features, start_dim=1)
        features = self.dropout(features)
        return features

