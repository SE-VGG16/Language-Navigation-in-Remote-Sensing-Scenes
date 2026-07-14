from __future__ import annotations

import argparse
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
from soar.metrics import waypoint_metrics
from soar.model import SOARModel
from soar.telemetry_noise import TelemetryNoise, perturb_telemetry


PRESETS = {
    "nominal": TelemetryNoise(),
    "position_0.5m": TelemetryNoise(position_sigma_m=0.5),
    "position_1m": TelemetryNoise(position_sigma_m=1.0),
    "position_2m": TelemetryNoise(position_sigma_m=2.0),
    "position_5m": TelemetryNoise(position_sigma_m=5.0),
    "altitude_+1m": TelemetryNoise(altitude_bias_m=1.0),
    "altitude_+2m": TelemetryNoise(altitude_bias_m=2.0),
    "altitude_+5m": TelemetryNoise(altitude_bias_m=5.0),
    "heading_1deg": TelemetryNoise(heading_bias_deg=1.0),
    "heading_3deg": TelemetryNoise(heading_bias_deg=3.0),
    "heading_5deg": TelemetryNoise(heading_bias_deg=5.0),
    "heading_10deg": TelemetryNoise(heading_bias_deg=10.0),
    "heading_drift": TelemetryNoise(heading_random_walk_deg=1.0),
    "combined_moderate": TelemetryNoise(
        position_sigma_m=1.0,
        altitude_bias_m=1.0,
        heading_bias_deg=3.0,
    ),
}


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
    model = SOARModel(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload))
    model.eval()

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
    for preset_name, noise in PRESETS.items():
        errors, successes = [], []
        generator = torch.Generator(device=device).manual_seed(config.seed)
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                noisy = perturb_telemetry(batch, noise, generator)
                output = model(noisy)
                metrics = waypoint_metrics(
                    output["waypoint_xy_m"],
                    batch["target_waypoint_xy_m"],
                    config.training.success_radius_m,
                )
                errors.extend(metrics["waypoint_error_m"].cpu().tolist())
                successes.extend(metrics["success"].cpu().tolist())
        rows.append(
            {
                "condition": preset_name,
                "mean_waypoint_error_m": sum(errors) / len(errors),
                "success_rate": sum(successes) / len(successes),
                "episodes": len(errors),
            }
        )
        print(rows[-1])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
