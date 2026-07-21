"""Cluster-bootstrap utilities for FairJob Stage1.1 diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    statistic: Callable[[pd.DataFrame], Mapping[str, float]],
    repeats: int = 200,
    seed: int = 2019,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap complete clusters and return percentile intervals.

    Repeatedly sampled clusters receive new bootstrap-local identifiers. This
    prevents duplicate impressions from being merged when a statistic performs
    a second groupby on the cluster column.
    """

    if cluster_column not in frame.columns:
        raise KeyError(f"Unknown bootstrap cluster column: {cluster_column}")
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")
    grouped = {key: value for key, value in frame.groupby(cluster_column, sort=False)}
    cluster_keys = np.asarray(list(grouped), dtype=object)
    if len(cluster_keys) < 2:
        raise ValueError("Cluster bootstrap requires at least two clusters.")

    point = {name: float(value) for name, value in statistic(frame).items()}
    samples = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    original_name = f"_original_{cluster_column}"
    for _ in range(repeats):
        selected = rng.choice(cluster_keys, size=len(cluster_keys), replace=True)
        blocks = []
        for draw_index, key in enumerate(selected):
            block = grouped[key].copy()
            block[original_name] = block[cluster_column]
            block[cluster_column] = draw_index
            blocks.append(block)
        replicate = statistic(pd.concat(blocks, ignore_index=True))
        for name in point:
            value = float(replicate[name])
            if not np.isfinite(value):
                raise ValueError(f"Bootstrap statistic {name} is not finite.")
            samples[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    result = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        result[name] = {
            "point": point[name],
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if repeats > 1 else 0.0,
            "ci_low": float(np.quantile(array, alpha)),
            "ci_high": float(np.quantile(array, 1.0 - alpha)),
        }
    return {
        "cluster_column": cluster_column,
        "clusters": int(len(cluster_keys)),
        "repeats": repeats,
        "seed": seed,
        "confidence": confidence,
        "interval": "percentile",
        "metrics": result,
    }
