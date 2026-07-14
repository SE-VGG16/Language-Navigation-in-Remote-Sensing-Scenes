from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TelemetryNoise:
    position_sigma_m: float = 0.0
    position_bias_m: tuple[float, float] = (0.0, 0.0)
    altitude_sigma_m: float = 0.0
    altitude_bias_m: float = 0.0
    heading_sigma_deg: float = 0.0
    heading_bias_deg: float = 0.0
    heading_random_walk_deg: float = 0.0


def perturb_telemetry(
    batch: dict[str, Tensor],
    noise: TelemetryNoise,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Return a shallow batch copy with controlled telemetry perturbations."""
    out = dict(batch)

    position = batch["position_xy_m"].clone()
    if noise.position_sigma_m:
        position = position + torch.randn(
            position.shape, device=position.device, dtype=position.dtype, generator=generator
        ) * noise.position_sigma_m
    bias = torch.tensor(noise.position_bias_m, device=position.device, dtype=position.dtype)
    position = position + bias
    out["position_xy_m"] = position

    altitude = batch["altitude_m"].clone()
    if noise.altitude_sigma_m:
        altitude = altitude + torch.randn(
            altitude.shape, device=altitude.device, dtype=altitude.dtype, generator=generator
        ) * noise.altitude_sigma_m
    altitude = (altitude + noise.altitude_bias_m).clamp_min(0.5)
    out["altitude_m"] = altitude

    heading = batch["heading_deg"].clone()
    if noise.heading_sigma_deg:
        heading = heading + torch.randn(
            heading.shape, device=heading.device, dtype=heading.dtype, generator=generator
        ) * noise.heading_sigma_deg
    heading = heading + noise.heading_bias_deg
    if noise.heading_random_walk_deg:
        increments = torch.randn(
            heading.shape, device=heading.device, dtype=heading.dtype, generator=generator
        ) * noise.heading_random_walk_deg
        heading = heading + torch.cumsum(increments, dim=-1)
    out["heading_deg"] = torch.remainder(heading, 360.0)
    return out
