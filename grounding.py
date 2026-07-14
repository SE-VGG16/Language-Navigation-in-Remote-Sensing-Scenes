from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .config import GroundingConfig


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, cells: Tensor, text: Tensor, text_mask: Tensor) -> Tensor:
        q = self.q_norm(cells)
        kv = self.kv_norm(text)
        attended, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=~text_mask.bool(),
            need_weights=False,
        )
        cells = cells + attended
        return cells + self.ff(self.ff_norm(cells))


class LanguageGrounding(nn.Module):
    def __init__(
        self,
        bev_dim: int,
        text_dim: int,
        grid_size: int,
        config: GroundingConfig,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.text_adapter = nn.Linear(text_dim, bev_dim)
        self.position = nn.Parameter(torch.zeros(1, grid_size**2, bev_dim))
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    bev_dim, config.num_heads, config.mlp_ratio, config.dropout
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(bev_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, bev: Tensor, text: Tensor, text_mask: Tensor) -> Tensor:
        b, c, h, w = bev.shape
        if h != self.grid_size or w != self.grid_size:
            raise ValueError("BEV grid size does not match grounding configuration.")
        cells = bev.flatten(2).transpose(1, 2) + self.position
        text = self.text_adapter(text)
        for block in self.blocks:
            cells = block(cells, text, text_mask)
        cells = self.norm(cells)
        return cells.transpose(1, 2).reshape(b, c, h, w)
