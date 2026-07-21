"""Decompose FairJob prediction disparity into within-context and composition terms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def decompose_dp(
    frame: pd.DataFrame,
    context_columns: Sequence[str],
    reference: str = "pooled",
) -> dict:
    """Return aggregate, within-context, and residual composition disparity.

    Only contexts observed in both protected groups identify a within-context
    contrast. The pooled reference weights common-support contexts by their
    total row frequency. ``group_average`` instead averages the two group-wise
    context distributions before renormalizing on common support. The residual
    is descriptive and must not be interpreted causally.
    """

    required = {"protected_attribute", "y_pred", *context_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"DP decomposition requires columns: {missing}")
    if not context_columns:
        raise ValueError("At least one context column is required.")
    if reference not in {"pooled", "group_average"}:
        raise KeyError(f"Unknown context reference: {reference}")

    work = frame[["protected_attribute", "y_pred", *context_columns]].copy()
    work["protected_attribute"] = work["protected_attribute"].astype(int)
    groups = {
        group: work[work["protected_attribute"] == group] for group in (0, 1)
    }
    if any(group.empty for group in groups.values()):
        raise ValueError("DP decomposition requires both protected groups.")
    aggregate = float(groups[1]["y_pred"].mean() - groups[0]["y_pred"].mean())

    grouped = (
        work.groupby([*context_columns, "protected_attribute"], dropna=False)["y_pred"]
        .agg(["mean", "size"])
        .reset_index()
    )
    means = grouped.pivot(index=list(context_columns), columns="protected_attribute", values="mean")
    sizes = grouped.pivot(index=list(context_columns), columns="protected_attribute", values="size").fillna(0)
    common = means.dropna(subset=[0, 1]).index
    if len(common) == 0:
        raise ValueError("No context has observations from both protected groups.")
    means = means.loc[common]
    sizes = sizes.loc[common]

    if reference == "pooled":
        weights = sizes[0] + sizes[1]
    else:
        weights = sizes[0] / len(groups[0]) + sizes[1] / len(groups[1])
    weights = weights / weights.sum()
    context_gaps = means[1] - means[0]
    within = float(np.sum(weights.to_numpy() * context_gaps.to_numpy()))

    rows = []
    for key, weight in weights.items():
        key_values = key if isinstance(key, tuple) else (key,)
        row = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in zip(context_columns, key_values)
        }
        row.update(
            {
                "group_0_n": int(sizes.loc[key, 0]),
                "group_1_n": int(sizes.loc[key, 1]),
                "group_0_mean": float(means.loc[key, 0]),
                "group_1_mean": float(means.loc[key, 1]),
                "signed_gap": float(context_gaps.loc[key]),
                "reference_weight": float(weight),
            }
        )
        rows.append(row)

    common_counts = sizes.sum(axis=0)
    return {
        "context_columns": list(context_columns),
        "reference": reference,
        "aggregate_dp_signed": aggregate,
        "within_context_dp_signed": within,
        "composition_residual_signed": float(aggregate - within),
        "common_contexts": int(len(common)),
        "group_0_common_support_fraction": float(common_counts[0] / len(groups[0])),
        "group_1_common_support_fraction": float(common_counts[1] / len(groups[1])),
        "contexts": rows,
        "interpretation": "descriptive_residual_not_causal_effect",
    }
