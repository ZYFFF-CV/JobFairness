"""Hyperparameter loading and recording for FairJob baselines.

The integration flow distinguishes parameter provenance from parameter values.
This prevents fallback or quick-trial settings from being reported as official
FairJob paper hyperparameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


XGB_KEYS = [
    "max_depth",
    "min_child_weight",
    "subsample",
    "learning_rate",
    "colsample_bytree",
    "reg_lambda",
    "gamma",
]


def read_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def official_hparam_files(reference_repo: str | Path | None, pattern: str) -> list[Path]:
    """Find optional official artifacts without making them runtime dependencies."""

    if not reference_repo:
        return []
    root = Path(reference_repo) / "output" / "model_hyperparameters"
    return sorted(root.glob(pattern)) if root.exists() else []


def load_official_xgb_params(reference_repo: str | Path | None) -> dict[str, Any] | None:
    """Load mappable XGBoost params from a FairJob reference output CSV."""

    files = official_hparam_files(reference_repo, "XGBoost_opt_params*.csv")
    if not files:
        return None
    latest = files[-1]
    df = pd.read_csv(latest)
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()
    params = {}
    for key in XGB_KEYS:
        if key in row and pd.notna(row[key]):
            value = row[key]
            params[key] = int(value) if key == "max_depth" else float(value)
    params["_source_file"] = str(latest)
    return params


def resolve_xgb_hparams(
    hparams_path: str | Path,
    reference_repo: str | Path | None = None,
    tune_trials: int = 0,
) -> tuple[dict, str]:
    """Resolve XGBoost params and return both values and provenance label.

    Priority is official reference output, manual YAML, explicit quick tuning,
    then hard-coded fallback. The caller performs tuning when the returned source
    is ``quick_trials``.
    """

    official = load_official_xgb_params(reference_repo)
    base = read_yaml(hparams_path)
    if official:
        base.update({k: v for k, v in official.items() if not k.startswith("_")})
        base["official_source_file"] = official["_source_file"]
        return base, "official_opt_csv"
    if tune_trials > 0:
        return base, "quick_trials"
    if base:
        return base, "manual_reference_yaml"
    return {
        "tree_method": "hist",
        "scale_pos_weight": "auto_neg_pos_ratio",
        "n_estimators": 100,
        "n_jobs": -1,
        "random_state": 2019,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "learning_rate": 0.05,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }, "fallback_reference"


def write_hparams(path: str | Path, params: dict, source: str, extra: dict | None = None) -> None:
    """Write the exact hparams used by one run for report traceability."""

    payload = {"hparams_source": source, "hparams": params}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
