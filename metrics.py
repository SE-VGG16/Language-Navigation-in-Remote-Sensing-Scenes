from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor


def waypoint_metrics(
    predicted_xy_m: Tensor,
    target_xy_m: Tensor,
    success_radius_m: float,
) -> dict[str, Tensor]:
    error = torch.linalg.vector_norm(predicted_xy_m - target_xy_m, dim=-1)
    success = (error <= success_radius_m).to(torch.float32)
    return {"waypoint_error_m": error, "success": success}


def summarize(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    result = {}
    for key, seq in values.items():
        array = np.asarray(seq, dtype=np.float64)
        result[key] = {
            "mean": float(array.mean()) if array.size else float("nan"),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "count": int(array.size),
        }
    return result


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be one-dimensional arrays with equal shape.")
    rng = np.random.default_rng(seed)
    n = a.size
    observed = float(np.mean(a - b))
    indices = rng.integers(0, n, size=(iterations, n))
    differences = (a[indices] - b[indices]).mean(axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    p = 2.0 * min(
        (np.count_nonzero(differences <= 0) + 1) / (iterations + 1),
        (np.count_nonzero(differences >= 0) + 1) / (iterations + 1),
    )
    return {
        "difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(p, 1.0)),
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()
