"""Calibration metrics for FairJob probability predictions."""

from __future__ import annotations

import numpy as np


def brier_score(y_true, y_pred) -> float:
    """Return the mean squared error of binary click probabilities."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("Brier score inputs must have equal length.")
    return float(np.mean((y_pred - y_true) ** 2))


def expected_calibration_error(y_true, y_pred, n_bins: int = 10) -> float:
    """Compute equal-width expected calibration error.

    The final bin includes probability 1.0. Empty bins contribute zero, and the
    remaining bin gaps are weighted by their share of the evaluated rows.
    """

    if n_bins < 2:
        raise ValueError("ECE requires at least two bins.")
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("ECE inputs must be non-empty and have equal length.")
    if not np.isfinite(y_pred).all() or ((y_pred < 0) | (y_pred > 1)).any():
        raise ValueError("ECE predictions must be finite probabilities.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.searchsorted(edges, y_pred, side="right") - 1, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            gap = abs(float(y_pred[mask].mean()) - float(y_true[mask].mean()))
            ece += float(mask.mean()) * gap
    return float(ece)
