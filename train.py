from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from soar.config import load_config
from soar.data import (
    SyntheticVLNDataset,
    TorchEpisodeDataset,
    collate_episodes,
    move_batch,
)
from soar.losses import SOARLoss
from soar.model import SOARModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.data:
        dataset = TorchEpisodeDataset(args.data)
    else:
        dataset = SyntheticVLNDataset(
            length=128,
            image_size=max(config.tokenizer.output_size, 160),
            max_text_length=config.text.max_length,
            vocab_size=config.text.vocab_size,
        )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_episodes,
    )
    model = SOARModel(config).to(device)
    criterion = SOARLoss(
        model.encoder.grid_size,
        config.tokenizer.field_of_view_m,
        config.loss,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history = []
    for epoch in range(config.training.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output_dict = model(batch)
            losses = criterion(output_dict, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            epoch_loss += float(losses["loss"].detach())

        mean_loss = epoch_loss / max(len(loader), 1)
        row = {"epoch": epoch + 1, "loss": mean_loss}
        history.append(row)
        print(json.dumps(row))
        torch.save(
            {"model": model.state_dict(), "config": args.config, "epoch": epoch + 1},
            output / "model_last.pt",
        )

    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
