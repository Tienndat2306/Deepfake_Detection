"""Positional encoding modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalPositionalEncoding(nn.Module):
    """
    Learnable temporal positional encoding cho chuoi frame video.
    """

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model phai > 0.")
        if max_len <= 0:
            raise ValueError("max_len phai > 0.")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Khoi tao theo cong thuc sinusoidal (sin/cos) de co prior ve vi tri thoi gian.
        # Sau do dat lam nn.Parameter de model co the tiep tuc hoc/chinh sua khi training.
        pe = self._build_sinusoidal_table(max_len=max_len, d_model=d_model)
        self.pos_embedding = nn.Parameter(pe, requires_grad=True)  # [1, max_len, d_model]

        # Nen dat max_len lon hon num_frames train:
        # de khi infer voi so frame nhieu hon (hoac thay doi T), model van con "khong gian"
        # positional encoding de generalize, tranh bi gioi han cung theo T train.

    @staticmethod
    def _build_sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
        """Tao bang positional encoding sin/cos voi shape [1, max_len, d_model]."""
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # [ceil(d_model/2)]

        pe[:, 0::2] = torch.sin(position * div_term)
        # Neu d_model le, nhanh 1::2 ngan hon 0::2 -> cat div_term cho khop kich thuoc.
        odd_width = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(position * div_term[:odd_width])
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor co shape [B, T, d_model]
        Returns:
            Tensor co shape [B, T, d_model]
        """
        if x.ndim != 3:
            raise ValueError("x phai co shape [B, T, d_model].")

        _, seq_len, hidden_dim = x.shape
        if hidden_dim != self.d_model:
            raise ValueError(
                f"d_model khong khop: expected {self.d_model}, got {hidden_dim}."
            )
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} vuot qua max_len={self.max_len}. "
                "Hay tang max_len khi khoi tao module."
            )

        pos = self.pos_embedding[:, :seq_len, :]  # [1, T, d_model]
        out = x + pos
        return self.dropout(out)


class RelativePositionBias1D(nn.Module):
    """
    Relative positional bias cho attention 1D (theo thoi gian).
    Dung cho sequence ngan nhu clip T~=10.
    """

    def __init__(self, num_heads: int, max_len: int = 32) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads phai > 0.")
        if max_len <= 0:
            raise ValueError("max_len phai > 0.")

        self.num_heads = int(num_heads)
        self.max_len = int(max_len)
        self.relative_bias_table = nn.Parameter(
            torch.zeros(2 * self.max_len - 1, self.num_heads)
        )
        nn.init.normal_(self.relative_bias_table, mean=0.0, std=0.02)

    def forward(
        self,
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        if seq_len <= 0:
            raise ValueError("seq_len phai > 0.")
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} vuot max_len={self.max_len}. "
                "Hay tang max_len khi khoi tao RelativePositionBias1D."
            )

        coords = torch.arange(seq_len, device=device)
        rel = coords[:, None] - coords[None, :]
        rel = rel + (self.max_len - 1)

        bias = self.relative_bias_table[rel]  # [T, T, H]
        bias = bias.permute(2, 0, 1).contiguous()  # [H, T, T]
        if dtype is not None:
            bias = bias.to(dtype=dtype)
        return bias

