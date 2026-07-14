from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TokenizerConfig:
    output_size: int = 224
    field_of_view_m: float = 80.0
    focal_length_px: float = 320.0
    canonicalize_heading: bool = True
    align_corners: bool = False


@dataclass
class EncoderConfig:
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class BEVConfig:
    use_refinement: bool = True


@dataclass
class MemoryConfig:
    world_size_m: float = 256.0
    world_resolution_m: float = 1.0
    eps: float = 1e-6


@dataclass
class TextConfig:
    mode: str = "tiny"
    vocab_size: int = 8192
    max_length: int = 64
    embed_dim: int = 768
    depth: int = 2
    num_heads: int = 12
    dropout: float = 0.1
    freeze: bool = True
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class GroundingConfig:
    depth: int = 2
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class LossConfig:
    heatmap_weight: float = 1.0
    coordinate_weight: float = 1.0
    stop_weight: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    num_workers: int = 0
    grad_clip: float = 1.0
    success_radius_m: float = 5.0


@dataclass
class ModelConfig:
    seed: int = 42
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bev: BEVConfig = field(default_factory=BEVConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    text: TextConfig = field(default_factory=TextConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _construct(cls: type, value: dict[str, Any] | None):
    return cls(**(value or {}))


def load_config(path: str | Path) -> ModelConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ModelConfig(
        seed=int(raw.get("seed", 42)),
        tokenizer=_construct(TokenizerConfig, raw.get("tokenizer")),
        encoder=_construct(EncoderConfig, raw.get("encoder")),
        bev=_construct(BEVConfig, raw.get("bev")),
        memory=_construct(MemoryConfig, raw.get("memory")),
        text=_construct(TextConfig, raw.get("text")),
        grounding=_construct(GroundingConfig, raw.get("grounding")),
        loss=_construct(LossConfig, raw.get("loss")),
        training=_construct(TrainingConfig, raw.get("training")),
    )
