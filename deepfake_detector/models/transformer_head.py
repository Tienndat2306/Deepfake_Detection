"""Transformer head for sequence/fusion modeling."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from .pos_encoding import RelativePositionBias1D, TemporalPositionalEncoding
except ImportError:
    from models.pos_encoding import RelativePositionBias1D, TemporalPositionalEncoding


class DropPath(nn.Module):
    """Stochastic depth (drop path) per-sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * random_tensor / keep_prob


class TemporalTransformerEncoderLayer(nn.Module):
    """Encoder layer Pre-LN co ho tro attn bias + stochastic depth."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.nhead = int(nhead)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()
        self.drop_path = DropPath(drop_prob=drop_path_prob)

    def _expand_attn_bias(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attn_bias is None:
            return None
        if attn_bias.ndim != 3:
            raise ValueError("attn_bias phai co shape [num_heads, T, T].")
        num_heads, tgt_len, src_len = attn_bias.shape
        if num_heads != self.nhead:
            raise ValueError(
                f"num_heads khong khop: expected {self.nhead}, got {num_heads}."
            )
        batch_size = x.shape[0]
        mask = attn_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return mask.reshape(batch_size * self.nhead, tgt_len, src_len)

    def _sa_block(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor]) -> torch.Tensor:
        attn_mask = self._expand_attn_bias(x, attn_bias)
        out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.dropout_attn(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.ffn_drop(x)
        x = self.linear2(x)
        return self.dropout_ffn(x)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path(self._sa_block(self.norm1(x), attn_bias=attn_bias))
        x = x + self.drop_path(self._ff_block(self.norm2(x)))
        return x


class TransformerHead(nn.Module):
    """
    Transformer head cho deepfake video detection theo sequence.
    """

    def __init__(
        self,
        feat_dim: int = 1792,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
        num_frames: int = 10,
        drop_path_rate: float = 0.0,
        use_relative_pos_bias: bool = False,
    ) -> None:
        super().__init__()
        if feat_dim <= 0 or d_model <= 0:
            raise ValueError("feat_dim va d_model phai > 0.")
        if nhead <= 0 or num_layers <= 0:
            raise ValueError("nhead va num_layers phai > 0.")
        if num_frames <= 0:
            raise ValueError("num_frames phai > 0.")
        if not (0.0 <= float(drop_path_rate) < 1.0):
            raise ValueError("drop_path_rate phai nam trong [0, 1).")

        self.feat_dim = int(feat_dim)
        self.d_model = int(d_model)
        self.num_frames = int(num_frames)
        self.use_relative_pos_bias = bool(use_relative_pos_bias)

        # 1) Linear projection: feat_dim -> d_model.
        self.input_proj = nn.Linear(self.feat_dim, self.d_model)

        # 2) Learnable CLS token de tong hop thong tin toan clip.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # 3) Positional encoding cho chuoi [CLS] + frames.
        self.pos_encoding = TemporalPositionalEncoding(
            d_model=self.d_model,
            max_len=self.num_frames + 1,
            dropout=dropout,
        )
        self.rel_pos_bias = (
            RelativePositionBias1D(num_heads=nhead, max_len=self.num_frames + 1)
            if self.use_relative_pos_bias
            else None
        )

        # 4) Stack encoder layers voi drop-path tang dan.
        if num_layers == 1:
            dpr = [float(drop_path_rate)]
        else:
            dpr = torch.linspace(0, float(drop_path_rate), steps=num_layers).tolist()
        self.encoder_layers = nn.ModuleList(
            [
                TemporalTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=self.d_model * 4,
                    dropout=dropout,
                    drop_path_prob=float(dpr[i]),
                )
                for i in range(num_layers)
            ]
        )
        # Alias de giu backward-compat neu ben ngoai truy cap head.encoder.
        self.encoder = self.encoder_layers

        # 5) LayerNorm cuoi.
        self.final_norm = nn.LayerNorm(self.d_model)

        # 6) Dropout -> Linear(d_model, 1) de tra ve raw logit.
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.d_model, 1)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Khoi tao tham so cho CLS token va cac layer projection/head."""
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, feat_dim]
        Returns:
            logits: [B]
        """
        if x.ndim != 3:
            raise ValueError("x phai co shape [B, T, feat_dim].")

        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.feat_dim:
            raise ValueError(
                f"feat_dim khong khop: expected {self.feat_dim}, got {input_dim}."
            )
        if seq_len > self.num_frames:
            raise ValueError(
                f"seq_len={seq_len} vuot num_frames={self.num_frames}. "
                "Hay tang num_frames khi khoi tao TransformerHead."
            )

        # Project dac trung frame ve khong gian d_model.
        tokens = self.input_proj(x)  # [B, T, d_model]

        # Prepend CLS token vao dau sequence.
        cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, T+1, d_model]

        # Them positional encoding va dua qua encoder.
        tokens = self.pos_encoding(tokens)
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(
                seq_len=tokens.shape[1],
                device=tokens.device,
                dtype=tokens.dtype,
            )
        for layer in self.encoder_layers:
            tokens = layer(tokens, attn_bias=attn_bias)
        tokens = self.final_norm(tokens)

        # Chi lay CLS token (index 0) de tong hop thong tin toan clip.
        cls_out = tokens[:, 0, :]  # [B, d_model]

        # Tra ve raw logit [B].
        logits = self.classifier(self.dropout(cls_out)).squeeze(-1)
        return logits

