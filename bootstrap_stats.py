from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soar.metrics import holm_adjust, paired_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soar", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    soar = pd.read_csv(args.soar)
    control = pd.read_csv(args.control)
    merged = soar.merge(
        control,
        on="episode_id",
        suffixes=("_soar", "_control"),
        validate="one_to_one",
    )
    rows = []
    for metric in args.metrics:
        a = merged[f"{metric}_soar"].to_numpy()
        b = merged[f"{metric}_control"].to_numpy()
        result = paired_bootstrap(a, b, args.iterations, args.seed)
        rows.append({"metric": metric, **result, "episodes": len(merged)})

    adjusted = holm_adjust(row["p_value"] for row in rows)
    for row, p_adjusted in zip(rows, adjusted):
        row["p_value_holm"] = p_adjusted
        row["supported_0.05"] = (
            row["ci_low"] > 0 or row["ci_high"] < 0
        ) and p_adjusted < 0.05

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
