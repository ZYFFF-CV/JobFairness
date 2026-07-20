"""FairJob metric definitions used by the shared evaluator.

The functions operate on exported prediction CSVs plus raw meta columns. They do
not read FuxiCTR feature maps because FairJob utility metrics require original
impression/product/protected-attribute values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEMALE_RATIO = 0.536 / 0.392619
MALE_RATIO = 0.464 / 0.607381


def binary_log_loss(y_true, y_pred) -> float:
    """Compute binary negative log likelihood with clipping for stability."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1.0 - 1e-15)
    return float(-(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)).mean())


def roc_auc(y_true, y_pred) -> float:
    """Compute ROC-AUC without requiring sklearn in lightweight checks."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative labels.")
    ranks = pd.Series(y_pred).rank(method="average").to_numpy()
    sum_pos_ranks = ranks[y_true == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, y_pred) -> float:
    """Compute average precision for the optional AVG_P_SCORE report field."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    positives = int((y_true == 1).sum())
    if positives == 0:
        raise ValueError("Average precision requires at least one positive label.")
    order = np.argsort(-y_pred)
    sorted_true = y_true[order]
    precision_at_k = np.cumsum(sorted_true) / (np.arange(len(sorted_true)) + 1)
    return float((precision_at_k * sorted_true).sum() / positives)


def demographic_parity(frame: pd.DataFrame) -> float:
    """FairJob DP over senior job rows.

    The task card defines DP as the absolute difference in predicted click
    probability means between protected groups within the senior-job scope.
    """

    senior = frame[frame["senior"].astype(int) == 1]
    if senior.empty:
        raise ValueError("DP requires at least one senior=1 row.")
    group_1 = senior[senior["protected_attribute"].astype(int) == 1]["y_pred"]
    group_0 = senior[senior["protected_attribute"].astype(int) == 0]["y_pred"]
    if group_1.empty or group_0.empty:
        raise ValueError("DP requires both protected groups in senior scope.")
    return float(abs(group_1.mean() - group_0.mean()))


def add_prediction_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank rows within each impression by ascending predicted probability."""

    ranked = frame.copy()
    ranked["pred_rank"] = ranked.groupby("impression_id")["y_pred"].rank(
        method="first", ascending=True
    )
    return ranked


def utility(frame: pd.DataFrame) -> float:
    """Compute FairJob click utility on randomized-display rows.

    Higher predicted clicked items receive larger ranks because ranking is
    ascending and ranks are 1-based, matching the official FairJob implementation.
    """

    random_rows = frame[frame["displayrandom"].astype(int) == 1].copy()
    if random_rows.empty:
        raise ValueError("U requires at least one displayrandom=1 row.")
    ranked = add_prediction_rank(random_rows)
    per_impression = ranked.groupby("impression_id").apply(
        lambda g: float((g["pred_rank"] * g["click"]).mean())
    )
    return float(per_impression.mean())


def utility_product(frame: pd.DataFrame, unbiased_ratio: bool = True) -> float:
    """Compute product-level utility with optional population correction.

    ``unbiased_ratio=True`` yields U_TILDE by applying the campaign/population
    ratios from the FairJob reference constants.
    """

    random_rows = frame[frame["displayrandom"].astype(int) == 1].copy()
    if random_rows.empty:
        raise ValueError("U_TILDE requires at least one displayrandom=1 row.")
    ranked = add_prediction_rank(random_rows)
    if unbiased_ratio:
        ranked["ratio"] = np.where(
            ranked["protected_attribute"].astype(int) == 0, FEMALE_RATIO, MALE_RATIO
        )
    else:
        ranked["ratio"] = 1.0

    values = []
    for _, product_group in ranked.groupby("product_id"):
        n_impressions = product_group["impression_id"].nunique()
        total = 0.0
        for _, impression_group in product_group.groupby("impression_id"):
            total += float(
                (
                    impression_group["pred_rank"]
                    * impression_group["ratio"]
                    * impression_group["click"]
                ).sum()
            ) / n_impressions
        values.append(total)
    if not values:
        raise ValueError("U_TILDE requires at least one product group.")
    return float(np.mean(values))


def compute_fairjob_metrics(pred: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Compute the full FairJob metric payload for one aligned prediction file."""

    frame = meta.copy()
    frame["y_true"] = pred["y_true"].astype(float).to_numpy()
    frame["y_pred"] = pred["y_pred"].astype(float).to_numpy()
    frame["click"] = frame["click"].astype(float)

    # Keep all public metric names stable for CSV/JSON reports. The alias
    # UTILITY_PRODUCT_FAIR is included for easier comparison with reference code.
    metrics = {
        "NLLH": binary_log_loss(frame["y_true"], frame["y_pred"]),
        "AUC": roc_auc(frame["y_true"], frame["y_pred"]),
        "AVG_P_SCORE": average_precision(frame["y_true"], frame["y_pred"]),
        "DP": demographic_parity(frame),
        "U": utility(frame),
        "U_TILDE": utility_product(frame, unbiased_ratio=True),
        "UTILITY_PRODUCT_FAIR": utility_product(frame, unbiased_ratio=True),
        "n_rows": int(len(frame)),
        "n_displayrandom": int((frame["displayrandom"].astype(int) == 1).sum()),
        "female_ratio": FEMALE_RATIO,
        "male_ratio": MALE_RATIO,
    }
    for key, value in metrics.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"Metric {key} is not finite: {value}")
    return metrics
