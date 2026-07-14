from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .config import BEVConfig
from .geometry import warp_perspective_normalized


class NearOrthographicBEVLift(nn.Module):
    """Reshape spatial tokens into BEV and refine local artifacts."""

    def __init__(self, embed_dim: int, grid_size: int, config: BEVConfig):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        if config.use_refinement:
            self.refine = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
        else:
            self.refine = nn.Identity()

    def forward(self, tokens: Tensor, homography: Tensor | None = None) -> Tensor:
        b, n, c = tokens.shape
        if n != self.grid_size**2 or c != self.embed_dim:
            raise ValueError("Token shape is incompatible with the configured BEV grid.")
        x = tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)
        if homography is not None:
            x = warp_perspective_normalized(x, homography)
        return x + self.refine(x)
