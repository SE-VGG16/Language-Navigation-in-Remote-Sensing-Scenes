from __future__ import annotations

import argparse
from pathlib import Path

import torch

from soar.data import SyntheticVLNDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = SyntheticVLNDataset(
        length=args.num_episodes,
        window_size=args.window_size,
        image_size=args.image_size,
    )
    for index, episode in enumerate(dataset):
        torch.save(episode, output / f"episode_{index:06d}.pt")
    print(f"Wrote {len(dataset)} synthetic episodes to {output}")


if __name__ == "__main__":
    main()
