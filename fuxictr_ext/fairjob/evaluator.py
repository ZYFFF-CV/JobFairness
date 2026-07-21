"""Evaluate FairJob prediction CSV files with shared metrics.

The evaluator is deliberately model-agnostic. It first proves prediction/meta
alignment, then computes metrics only from raw meta fields and predicted click
probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.metrics import compute_fairjob_metrics
from fuxictr_ext.fairjob.metrics_extended import compute_extended_metrics
from fuxictr_ext.fairjob.prediction_io import check_prediction_alignment, read_meta
from fuxictr_ext.fairjob.protocols import scientific_regime_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def evaluate_prediction_file(pred_path: str | Path, meta_path: str | Path) -> dict:
    """Evaluate one aligned prediction file and return a JSON-ready payload."""

    check_prediction_alignment(pred_path, meta_path)
    pred = pd.read_csv(pred_path)
    meta = read_meta(meta_path)
    metrics = compute_fairjob_metrics(pred, meta)
    frame = meta.copy()
    frame["y_true"] = pred["y_true"].astype(float).to_numpy()
    frame["y_pred"] = pred["y_pred"].astype(float).to_numpy()
    metrics.update(compute_extended_metrics(frame))

    # Prediction files are single-run artifacts, so the first row carries the
    # model/regime/mode labels used in the metric summary.
    first = pred.iloc[0].to_dict()
    return {
        "prediction_path": str(pred_path),
        "meta_path": str(meta_path),
        "model": first.get("model"),
        "regime": first.get("regime"),
        "regime_scientific": scientific_regime_name(first.get("regime")),
        "protocol": first.get("protocol"),
        "mode": first.get("mode"),
        "expid": first.get("expid", Path(pred_path).stem),
        "hparams_source": first.get("hparams_source"),
        **metrics,
    }


def write_evaluation_outputs(payload: dict, out_path: str | Path) -> None:
    """Write matching JSON and one-row CSV evaluator artifacts."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([payload]).to_csv(out.with_suffix(".csv"), index=False)


def main() -> None:
    args = parse_args()
    payload = evaluate_prediction_file(args.pred, args.meta)
    write_evaluation_outputs(payload, args.out)
    out = Path(args.out)
    csv_out = out.with_suffix(".csv")
    print(f"Wrote {out}")
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
