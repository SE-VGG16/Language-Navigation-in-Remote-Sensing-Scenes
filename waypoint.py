from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .geometry import metric_cell_coordinates


class MetricWaypointHead(nn.Module):
    def __init__(self, embed_dim: int, grid_size: int, field_of_view_m: float):
        super().__init__()
        self.grid_size = grid_size
        self.field_of_view_m = float(field_of_view_m)
        self.score = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.stop = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 1),
        )
        coords = metric_cell_coordinates(grid_size, field_of_view_m)
        self.register_buffer("coordinates_xy_m", coords, persistent=False)

    def forward(self, grounded_bev: Tensor) -> dict[str, Tensor]:
        logits = self.score(grounded_bev).squeeze(1)
        probabilities = torch.softmax(logits.flatten(1), dim=-1)
        waypoint = torch.matmul(
            probabilities, self.coordinates_xy_m.to(probabilities.dtype)
        )
        return {
            "waypoint_logits": logits,
            "waypoint_probabilities": probabilities.reshape_as(logits),
            "waypoint_xy_m": waypoint,
            "stop_logit": self.stop(grounded_bev).squeeze(-1),
        }
