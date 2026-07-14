from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .bev_lift import NearOrthographicBEVLift
from .config import ModelConfig
from .encoder import SpatialViTEncoder
from .grounding import LanguageGrounding
from .memory import PoseRegisteredBEVMemory
from .text import build_text_encoder
from .tokenizer import MetricTokenizer
from .waypoint import MetricWaypointHead


class SOARModel(nn.Module):
    """End-to-end metric-consistent aerial VLN model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.metric_tokenizer = MetricTokenizer(config.tokenizer)
        self.encoder = SpatialViTEncoder(config.encoder)
        self.bev_lift = NearOrthographicBEVLift(
            config.encoder.embed_dim, self.encoder.grid_size, config.bev
        )
        self.memory = PoseRegisteredBEVMemory(
            config.tokenizer.field_of_view_m, config.memory
        )
        self.text_encoder = build_text_encoder(config.text)
        self.grounding = LanguageGrounding(
            bev_dim=config.encoder.embed_dim,
            text_dim=self.text_encoder.output_dim,
            grid_size=self.encoder.grid_size,
            config=config.grounding,
        )
        self.waypoint_head = MetricWaypointHead(
            config.encoder.embed_dim,
            self.encoder.grid_size,
            config.tokenizer.field_of_view_m,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        images = batch["images"]
        if images.ndim == 4:
            images = images[:, None]
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, 3, H, W].")
        b, t, _, _, _ = images.shape

        altitude = batch["altitude_m"]
        heading = batch.get("heading_deg", torch.zeros_like(altitude))
        positions = batch.get(
            "position_xy_m",
            torch.zeros(b, t, 2, device=images.device, dtype=images.dtype),
        )
        if altitude.ndim == 1:
            altitude = altitude[:, None]
        if heading.ndim == 1:
            heading = heading[:, None]

        tokenized = self.metric_tokenizer(images, altitude, heading)
        canonical = tokenized["image"].reshape(
            b * t,
            3,
            self.config.tokenizer.output_size,
            self.config.tokenizer.output_size,
        )
        tokens = self.encoder(canonical)

        homography = batch.get("tilt_homography")
        flat_h = None
        if homography is not None:
            flat_h = homography.reshape(b * t, 3, 3).to(tokens.dtype)
        local_bev = self.bev_lift(tokens, flat_h)
        g = local_bev.shape[-1]
        local_bev = local_bev.reshape(b, t, local_bev.shape[1], g, g)

        memory = self.memory(local_bev, positions.to(local_bev.dtype))
        text = self.text_encoder(
            batch["instruction_ids"], batch["instruction_mask"]
        )
        grounded = self.grounding(
            memory["crop"], text, batch["instruction_mask"]
        )
        output = self.waypoint_head(grounded)
        output.update(
            {
                "canonical_images": tokenized["image"],
                "coverage": tokenized["coverage"],
                "crop_pixels": tokenized["crop_pixels"],
                "local_bev": local_bev,
                "memory_crop": memory["crop"],
                "world_map": memory["world_map"],
                "world_count": memory["count_map"],
                "grounded_bev": grounded,
            }
        )
        return output
