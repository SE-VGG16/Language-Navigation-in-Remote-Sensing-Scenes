from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TokenizerConfig
from .geometry import metric_sampling_theta


class MetricTokenizer(nn.Module):
    """Telemetry-conditioned image canonicalization.

    The module has no learnable parameters. It performs a centered metric crop,
    resizes it to a fixed image size, and optionally rotates it to north-up.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        images: Tensor,
        altitude_m: Tensor,
        heading_deg: Tensor,
    ) -> dict[str, Tensor]:
        if images.ndim < 4:
            raise ValueError("images must end with [C, H, W].")
        if images.shape[-3] != 3:
            raise ValueError("images must contain three RGB channels.")

        leading = images.shape[:-3]
        h, w = images.shape[-2:]
        flat_images = images.reshape(-1, 3, h, w)
        flat_altitude = altitude_m.reshape(-1).to(dtype=flat_images.dtype)
        flat_heading = heading_deg.reshape(-1).to(dtype=flat_images.dtype)

        if flat_images.shape[0] != flat_altitude.numel():
            raise ValueError("Telemetry leading dimensions must match image leading dimensions.")

        theta, coverage, crop_pixels = metric_sampling_theta(
            flat_altitude,
            flat_heading,
            h,
            w,
            self.config.field_of_view_m,
            self.config.focal_length_px,
            self.config.canonicalize_heading,
        )
        grid = F.affine_grid(
            theta,
            size=(
                flat_images.shape[0],
                3,
                self.config.output_size,
                self.config.output_size,
            ),
            align_corners=self.config.align_corners,
        )
        canonical = F.grid_sample(
            flat_images,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.config.align_corners,
        )

        out_shape = (*leading, 3, self.config.output_size, self.config.output_size)
        return {
            "image": canonical.reshape(out_shape),
            "coverage": coverage.reshape(leading),
            "crop_pixels": crop_pixels.reshape(leading),
            "theta": theta.reshape(*leading, 2, 3),
        }
