from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .text import HashTokenizer


class SyntheticVLNDataset(Dataset):
    """Small deterministic dataset used only for software tests."""

    instructions = (
        "fly toward the red landmark",
        "move north of the blue building",
        "approach the green target",
        "continue past the bright square",
    )

    def __init__(
        self,
        length: int = 128,
        window_size: int = 4,
        image_size: int = 160,
        max_text_length: int = 48,
        vocab_size: int = 8192,
        seed: int = 7,
    ):
        self.length = length
        self.window_size = window_size
        self.image_size = image_size
        self.seed = seed
        self.tokenizer = HashTokenizer(vocab_size, max_text_length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(self.seed + index)
        t, h, w = self.window_size, self.image_size, self.image_size
        images = torch.rand(t, 3, h, w, generator=gen) * 0.12

        target = (torch.rand(2, generator=gen) - 0.5) * 50.0
        altitude = 35.0 + torch.rand(t, generator=gen) * 70.0
        heading = torch.rand(t, generator=gen) * 360.0
        steps = torch.randn(t, 2, generator=gen) * 2.0
        position = torch.cumsum(steps, dim=0)
        position = position - position[0]

        # Draw a simple bright patch whose location roughly reflects the target.
        px = int((target[0].item() / 80.0 + 0.5) * (w - 1))
        py = int((0.5 - target[1].item() / 80.0) * (h - 1))
        px = max(3, min(w - 4, px))
        py = max(3, min(h - 4, py))
        images[:, 0, py - 3 : py + 4, px - 3 : px + 4] = 1.0

        text = self.instructions[index % len(self.instructions)]
        ids, mask = self.tokenizer.encode(text)
        return {
            "images": images,
            "altitude_m": altitude,
            "heading_deg": heading,
            "position_xy_m": position,
            "instruction_ids": ids,
            "instruction_mask": mask,
            "target_waypoint_xy_m": target,
            "stop_target": torch.tensor(0.0),
            "episode_id": f"synthetic-{index:06d}",
        }


class TorchEpisodeDataset(Dataset):
    """Load trusted torch-saved episode dictionaries from a directory."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt episode files found in {self.directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = torch.load(self.files[index], map_location="cpu", weights_only=False)
        required = {
            "images",
            "altitude_m",
            "heading_deg",
            "position_xy_m",
            "instruction_ids",
            "instruction_mask",
            "target_waypoint_xy_m",
        }
        missing = required.difference(item)
        if missing:
            raise KeyError(f"{self.files[index]} is missing fields: {sorted(missing)}")
        item.setdefault("stop_target", torch.tensor(0.0))
        item.setdefault("episode_id", self.files[index].stem)
        return item


def collate_episodes(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
