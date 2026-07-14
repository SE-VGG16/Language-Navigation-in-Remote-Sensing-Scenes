from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from .config import EncoderConfig


class SpatialViTEncoder(nn.Module):
    """Vision Transformer that preserves all spatial patch tokens."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.config = config
        self.grid_size = config.image_size // config.patch_size
        self.num_tokens = self.grid_size**2

        self.patch_embed = nn.Conv2d(
            3,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, self.num_tokens, config.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        if x.shape[1] != self.num_tokens:
            raise RuntimeError(
                f"Expected {self.num_tokens} tokens, got {x.shape[1]}. "
                "Check tokenizer and encoder image sizes."
            )
        x = x + self.position
        x = self.blocks(x)
        return self.norm(x)

    def load_checkpoint(self, path: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load a native or partially compatible checkpoint.

        The function accepts either a state dictionary or a dictionary containing
        `state_dict` or `model`. External ViT checkpoints may require key conversion.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload:
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint does not contain a state dictionary.")

        cleaned = {}
        for key, value in payload.items():
            for prefix in ("module.", "encoder."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value
        result = self.load_state_dict(cleaned, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
