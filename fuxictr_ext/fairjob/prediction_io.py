"""Prediction export and alignment helpers for FairJob.

All baselines converge on this CSV contract before evaluation. The strict row_id
checks protect FairJob metrics from accidentally grouping by FuxiCTR-tokenized
IDs or by a shuffled test order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PRED_COLUMNS = ["row_id", "model", "regime", "mode", "y_true", "y_pred"]
REQUIRED_META_COLUMNS = [
    "row_id",
    "click",
    "protected_attribute",
    "senior",
    "displayrandom",
    "impression_id",
    "product_id",
]


def read_meta(meta_path: str | Path) -> pd.DataFrame:
    """Read evaluator metadata and validate required raw grouping fields."""

    meta = pd.read_csv(meta_path)
    missing = [col for col in REQUIRED_META_COLUMNS if col not in meta.columns]
    if missing:
        raise ValueError(f"Meta file is missing columns {missing}: {meta_path}")
    return meta


def validate_probability(y_pred: np.ndarray) -> np.ndarray:
    """Normalize prediction shape and require finite probabilities in [0, 1]."""

    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    if ((y_pred < 0.0) | (y_pred > 1.0)).any():
        raise ValueError("Predictions must be probabilities in [0, 1].")
    return y_pred


def write_prediction_csv(
    out_path: str | Path,
    meta_path: str | Path,
    y_pred,
    model: str,
    regime: str,
    mode: str,
    expid: str | None = None,
    hparams_source: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Write the canonical prediction CSV aligned to a meta file.

    ``row_id`` and ``y_true`` are copied from meta rather than trusted from model
    code, making row-order mistakes fail as length/alignment errors downstream.
    """

    meta = read_meta(meta_path)
    y_pred = validate_probability(np.asarray(y_pred, dtype=float))
    if len(meta) != len(y_pred):
        raise ValueError(
            f"Prediction length {len(y_pred)} does not match meta rows {len(meta)}."
        )

    pred = pd.DataFrame(
        {
            "row_id": meta["row_id"].astype(int).to_numpy(),
            "model": model,
            "regime": regime,
            "mode": mode,
            "y_true": meta["click"].astype(float).to_numpy(),
            "y_pred": y_pred,
        }
    )
    if expid is not None:
        pred["expid"] = expid
    if hparams_source is not None:
        pred["hparams_source"] = hparams_source
    if seed is not None:
        pred["seed"] = seed

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(pred)}")
    return pred


def check_prediction_alignment(pred_path: str | Path, meta_path: str | Path) -> dict:
    """Raise if prediction rows cannot be joined exactly to evaluator metadata."""

    pred = pd.read_csv(pred_path)
    meta = read_meta(meta_path)
    missing = [col for col in REQUIRED_PRED_COLUMNS if col not in pred.columns]
    if missing:
        raise ValueError(f"Prediction file is missing columns {missing}: {pred_path}")
    if len(pred) != len(meta):
        raise ValueError(f"Row mismatch: pred={len(pred)} meta={len(meta)}")
    if not pred["row_id"].astype(int).equals(meta["row_id"].astype(int)):
        raise ValueError("row_id sequence does not match meta.")
    if not pred["y_true"].astype(float).equals(meta["click"].astype(float)):
        raise ValueError("y_true does not match meta click.")
    validate_probability(pred["y_pred"].to_numpy(dtype=float))
    return {"rows": len(pred), "pred_path": str(pred_path), "meta_path": str(meta_path)}
