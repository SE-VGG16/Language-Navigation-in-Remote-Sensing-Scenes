from __future__ import annotations

import hashlib
import re
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from .config import TextConfig


class HashTokenizer:
    """Deterministic whitespace/punctuation tokenizer for smoke tests."""

    PAD = 0
    UNK = 1

    def __init__(self, vocab_size: int = 8192, max_length: int = 64):
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16.")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        return 2 + value % (self.vocab_size - 2)

    def encode(self, text: str) -> tuple[Tensor, Tensor]:
        tokens = re.findall(r"[\w'-]+|[^\w\s]", text.lower(), flags=re.UNICODE)
        ids = [self._id(t) for t in tokens[: self.max_length]]
        mask = [True] * len(ids)
        while len(ids) < self.max_length:
            ids.append(self.PAD)
            mask.append(False)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class TinyTextEncoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.output_dim = config.embed_dim
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, config.max_length, config.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

        if config.freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        if ids.shape[1] > self.position.shape[1]:
            raise ValueError("Instruction length exceeds configured max_length.")
        x = self.embedding(ids) + self.position[:, : ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        return self.norm(x)


class HuggingFaceTextEncoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "Install requirements-full.txt to use text.mode='hf'."
            ) from exc

        self.model = AutoModel.from_pretrained(config.model_name)
        self.output_dim = int(self.model.config.hidden_size)
        if config.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        # Keep a frozen pretrained encoder in evaluation mode.
        frozen = not any(p.requires_grad for p in self.model.parameters())
        return super().train(False if frozen else mode)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        output = self.model(input_ids=ids, attention_mask=mask.long())
        return output.last_hidden_state


def build_text_encoder(config: TextConfig) -> nn.Module:
    if config.mode == "tiny":
        return TinyTextEncoder(config)
    if config.mode == "hf":
        return HuggingFaceTextEncoder(config)
    raise ValueError(f"Unsupported text encoder mode: {config.mode}")
