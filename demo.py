from __future__ import annotations

from pathlib import Path

import torch
import yaml

from soar.config import load_config
from soar.data import SyntheticVLNDataset, collate_episodes
from soar.model import SOARModel


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "soar_tiny.yaml")
    dataset = SyntheticVLNDataset(
        length=2,
        image_size=160,
        max_text_length=config.text.max_length,
        vocab_size=config.text.vocab_size,
    )
    batch = collate_episodes([dataset[0], dataset[1]])
    model = SOARModel(config)
    model.eval()
    with torch.no_grad():
        output = model(batch)

    print("waypoint_xy_m:", output["waypoint_xy_m"])
    print("waypoint_logits:", tuple(output["waypoint_logits"].shape))
    print("world_map:", tuple(output["world_map"].shape))
    print("coverage:", output["coverage"])


if __name__ == "__main__":
    main()
