from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from soar.config import load_config
from soar.data import (
    SyntheticVLNDataset,
    TorchEpisodeDataset,
    collate_episodes,
    move_batch,
)
from soar.metrics import summarize, waypoint_metrics
from soar.model import SOARModel


def load_model(config_path: str, checkpoint: str, device: torch.device) -> SOARModel:
    config = load_config(config_path)
    model = SOARModel(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    model = load_model(args.config, args.checkpoint, device)
    dataset = (
        TorchEpisodeDataset(args.data)
        if args.data
        else SyntheticVLNDataset(
            length=64,
            image_size=max(config.tokenizer.output_size, 160),
            max_text_length=config.text.max_length,
            vocab_size=config.text.vocab_size,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )

    rows = []
    values = {"waypoint_error_m": [], "success": []}
    with torch.no_grad():
        for batch in loader:
            episode_ids = batch["episode_id"]
            batch = move_batch(batch, device)
            output = model(batch)
            metrics = waypoint_metrics(
                output["waypoint_xy_m"],
                batch["target_waypoint_xy_m"],
                config.training.success_radius_m,
            )
            for i, episode_id in enumerate(episode_ids):
                row = {
                    "episode_id": episode_id,
                    "waypoint_error_m": float(metrics["waypoint_error_m"][i]),
                    "success": float(metrics["success"][i]),
                }
                rows.append(row)
                for key in values:
                    values[key].append(row[key])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "per_episode.csv", index=False)
    summary = summarize(values)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
