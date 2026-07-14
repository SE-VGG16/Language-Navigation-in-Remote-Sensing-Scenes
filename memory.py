from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MemoryConfig


class PoseRegisteredBEVMemory(nn.Module):
    """Differentiable allocentric memory registered by planar translation.

    Positions are x=east, y=north in meters and are assumed to share a common
    episode coordinate system. The world map is centered at (0, 0).
    """

    def __init__(self, field_of_view_m: float, config: MemoryConfig):
        super().__init__()
        self.field_of_view_m = float(field_of_view_m)
        self.world_size_m = float(config.world_size_m)
        self.world_resolution_m = float(config.world_resolution_m)
        self.eps = float(config.eps)
        self.world_grid_size = int(round(self.world_size_m / self.world_resolution_m))
        if self.world_grid_size < 2:
            raise ValueError("World grid must contain at least two cells.")

    def _world_to_local_grid(
        self,
        position_xy_m: Tensor,
        world_h: int,
        world_w: int,
        dtype: torch.dtype,
    ) -> Tensor:
        b = position_xy_m.shape[0]
        ys = torch.linspace(
            self.world_size_m / 2.0,
            -self.world_size_m / 2.0,
            world_h,
            device=position_xy_m.device,
            dtype=dtype,
        )
        xs = torch.linspace(
            -self.world_size_m / 2.0,
            self.world_size_m / 2.0,
            world_w,
            device=position_xy_m.device,
            dtype=dtype,
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = xx.unsqueeze(0) - position_xy_m[:, 0, None, None]
        dy = yy.unsqueeze(0) - position_xy_m[:, 1, None, None]
        gx = 2.0 * dx / self.field_of_view_m
        gy = -2.0 * dy / self.field_of_view_m
        return torch.stack([gx, gy], dim=-1)

    def _local_to_world_grid(
        self,
        position_xy_m: Tensor,
        local_h: int,
        local_w: int,
        dtype: torch.dtype,
    ) -> Tensor:
        ys = torch.linspace(
            self.field_of_view_m / 2.0,
            -self.field_of_view_m / 2.0,
            local_h,
            device=position_xy_m.device,
            dtype=dtype,
        )
        xs = torch.linspace(
            -self.field_of_view_m / 2.0,
            self.field_of_view_m / 2.0,
            local_w,
            device=position_xy_m.device,
            dtype=dtype,
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        wx = xx.unsqueeze(0) + position_xy_m[:, 0, None, None]
        wy = yy.unsqueeze(0) + position_xy_m[:, 1, None, None]
        gx = 2.0 * wx / self.world_size_m
        gy = -2.0 * wy / self.world_size_m
        return torch.stack([gx, gy], dim=-1)

    def forward(
        self,
        local_maps: Tensor,
        positions_xy_m: Tensor,
    ) -> dict[str, Tensor]:
        if local_maps.ndim != 5:
            raise ValueError("local_maps must have shape [B, T, C, G, G].")
        if positions_xy_m.shape[:2] != local_maps.shape[:2]:
            raise ValueError("positions_xy_m must have shape [B, T, 2].")

        b, t, c, g_h, g_w = local_maps.shape
        m = self.world_grid_size
        world_sum = torch.zeros(b, c, m, m, device=local_maps.device, dtype=local_maps.dtype)
        world_count = torch.zeros(b, 1, m, m, device=local_maps.device, dtype=local_maps.dtype)
        local_ones = torch.ones(b, 1, g_h, g_w, device=local_maps.device, dtype=local_maps.dtype)

        for step in range(t):
            grid = self._world_to_local_grid(
                positions_xy_m[:, step], m, m, local_maps.dtype
            )
            warped = F.grid_sample(
                local_maps[:, step],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            visible = F.grid_sample(
                local_ones,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            world_sum = world_sum + warped
            world_count = world_count + visible

        world_map = world_sum / world_count.clamp_min(self.eps)
        query_grid = self._local_to_world_grid(
            positions_xy_m[:, -1], g_h, g_w, local_maps.dtype
        )
        current_crop = F.grid_sample(
            world_map,
            query_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return {
            "crop": current_crop,
            "world_map": world_map,
            "count_map": world_count,
        }
