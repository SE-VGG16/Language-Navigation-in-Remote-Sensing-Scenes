from __future__ import annotations

import torch

from soar.config import (
    BEVConfig,
    EncoderConfig,
    GroundingConfig,
    MemoryConfig,
    ModelConfig,
    TextConfig,
    TokenizerConfig,
)
from soar.data import SyntheticVLNDataset, collate_episodes
from soar.model import SOARModel
from soar.tokenizer import MetricTokenizer


def tiny_config() -> ModelConfig:
    return ModelConfig(
        tokenizer=TokenizerConfig(
            output_size=64,
            field_of_view_m=40.0,
            focal_length_px=160.0,
        ),
        encoder=EncoderConfig(
            image_size=64,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            dropout=0.0,
        ),
        bev=BEVConfig(use_refinement=True),
        memory=MemoryConfig(world_size_m=80.0, world_resolution_m=4.0),
        text=TextConfig(
            mode="tiny",
            vocab_size=1024,
            max_length=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            dropout=0.0,
            freeze=False,
        ),
        grounding=GroundingConfig(depth=1, num_heads=4, dropout=0.0),
    )


def test_metric_tokenizer_shapes_and_coverage():
    config = tiny_config()
    module = MetricTokenizer(config.tokenizer)
    images = torch.rand(2, 3, 96, 96)
    altitude = torch.tensor([40.0, 100.0])
    heading = torch.tensor([0.0, 45.0])
    out = module(images, altitude, heading)
    assert out["image"].shape == (2, 3, 64, 64)
    assert torch.all(out["coverage"] >= 0)
    assert torch.all(out["coverage"] <= 1)


def test_model_forward():
    config = tiny_config()
    dataset = SyntheticVLNDataset(
        length=2,
        window_size=2,
        image_size=96,
        max_text_length=config.text.max_length,
        vocab_size=config.text.vocab_size,
    )
    batch = collate_episodes([dataset[0], dataset[1]])
    model = SOARModel(config)
    output = model(batch)
    assert output["waypoint_xy_m"].shape == (2, 2)
    assert output["waypoint_logits"].shape == (2, 4, 4)
    assert output["world_map"].shape[-2:] == (20, 20)
    assert torch.isfinite(output["waypoint_xy_m"]).all()
