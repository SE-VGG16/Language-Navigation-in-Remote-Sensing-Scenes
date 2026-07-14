from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig
from .geometry import metric_cell_coordinates


class SOARLoss(nn.Module):
    def __init__(self, grid_size: int, field_of_view_m: float, config: LossConfig):
        super().__init__()
        self.grid_size = grid_size
        self.field_of_view_m = float(field_of_view_m)
        self.config = config
        coords = metric_cell_coordinates(grid_size, field_of_view_m)
        self.register_buffer("coordinates_xy_m", coords, persistent=False)

    def _nearest_cell(self, target_xy_m: Tensor) -> Tensor:
        distances = torch.cdist(
            target_xy_m[:, None, :],
            self.coordinates_xy_m[None].to(target_xy_m.dtype),
        ).squeeze(1)
        return distances.argmin(dim=1)

    def forward(
        self, output: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        target_xy = batch["target_waypoint_xy_m"].to(output["waypoint_xy_m"].dtype)
        target_index = self._nearest_cell(target_xy)
        heatmap = F.cross_entropy(
            output["waypoint_logits"].flatten(1), target_index
        )
        coordinate = F.smooth_l1_loss(output["waypoint_xy_m"], target_xy)

        stop_target = batch.get("stop_target")
        if stop_target is None:
            stop = output["stop_logit"].sum() * 0.0
        else:
            stop = F.binary_cross_entropy_with_logits(
                output["stop_logit"], stop_target.reshape(-1).to(output["stop_logit"].dtype)
            )

        total = (
            self.config.heatmap_weight * heatmap
            + self.config.coordinate_weight * coordinate
            + self.config.stop_weight * stop
        )
        return {
            "loss": total,
            "heatmap_loss": heatmap.detach(),
            "coordinate_loss": coordinate.detach(),
            "stop_loss": stop.detach(),
        }
