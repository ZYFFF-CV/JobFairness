"""Stage1.1 FairJob disparity, group-performance, and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fuxictr_ext.fairjob.calibration import brier_score, expected_calibration_error
from fuxictr_ext.fairjob.metrics import binary_log_loss, roc_auc


def demographic_parity_details(frame: pd.DataFrame) -> dict:
    """Return signed and absolute senior-scope DP with auditable group details."""

    senior = frame[frame["senior"].astype(int) == 1]
    group_0 = senior[senior["protected_attribute"].astype(int) == 0]["y_pred"]
    group_1 = senior[senior["protected_attribute"].astype(int) == 1]["y_pred"]
    if group_0.empty or group_1.empty:
        raise ValueError("Senior-scope DP requires both protected proxy groups.")
    mean_0 = float(group_0.mean())
    mean_1 = float(group_1.mean())
    signed = mean_1 - mean_0
    return {
        "DP_signed": float(signed),
        "DP_abs": float(abs(signed)),
        "DP_group_0_mean": mean_0,
        "DP_group_1_mean": mean_1,
        "DP_group_0_n": int(len(group_0)),
        "DP_group_1_n": int(len(group_1)),
        "DP_senior_n": int(len(senior)),
        "DP_prediction_ratio_1_over_0": float(mean_1 / mean_0) if mean_0 != 0 else None,
    }


def _safe_auc(y_true, y_pred) -> float | None:
    if len(np.unique(np.asarray(y_true, dtype=int))) < 2:
        return None
    return roc_auc(y_true, y_pred)


def performance_metrics(y_true, y_pred, ece_bins: int = 10) -> dict:
    """Compute probability and ranking metrics for one evaluation slice."""

    return {
        "NLLH": binary_log_loss(y_true, y_pred),
        "AUC": _safe_auc(y_true, y_pred),
        "Brier": brier_score(y_true, y_pred),
        "ECE": expected_calibration_error(y_true, y_pred, n_bins=ece_bins),
        "n": int(len(y_true)),
        "positive_rate": float(np.asarray(y_true, dtype=float).mean()),
        "prediction_mean": float(np.asarray(y_pred, dtype=float).mean()),
    }


def compute_extended_metrics(frame: pd.DataFrame, ece_bins: int = 10) -> dict:
    """Return flattened overall and protected-proxy group diagnostics."""

    required = {"y_true", "y_pred", "protected_attribute", "senior"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Extended metrics require columns: {missing}")

    result = demographic_parity_details(frame)
    overall = performance_metrics(frame["y_true"], frame["y_pred"], ece_bins)
    result.update({f"overall_{key}": value for key, value in overall.items()})

    group_metrics = {}
    for group in (0, 1):
        group_frame = frame[frame["protected_attribute"].astype(int) == group]
        if group_frame.empty:
            raise ValueError(f"Protected proxy group {group} is empty.")
        values = performance_metrics(group_frame["y_true"], group_frame["y_pred"], ece_bins)
        group_metrics[group] = values
        result.update({f"group_{group}_{key}": value for key, value in values.items()})

    for metric in ("NLLH", "AUC", "Brier", "ECE"):
        left = group_metrics[0][metric]
        right = group_metrics[1][metric]
        result[f"group_gap_{metric}"] = (
            float(right - left) if left is not None and right is not None else None
        )
    return result
