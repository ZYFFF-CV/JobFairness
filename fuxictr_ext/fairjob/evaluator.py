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
from fuxictr_ext.fairjob.prediction_io import check_prediction_alignment, read_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_prediction_alignment(args.pred, args.meta)
    pred = pd.read_csv(args.pred)
    meta = read_meta(args.meta)
    metrics = compute_fairjob_metrics(pred, meta)

    # Prediction files are single-run artifacts, so the first row carries the
    # model/regime/mode labels used in the metric summary.
    first = pred.iloc[0].to_dict()
    payload = {
        "prediction_path": args.pred,
        "meta_path": args.meta,
        "model": first.get("model"),
        "regime": first.get("regime"),
        "mode": first.get("mode"),
        "expid": first.get("expid", Path(args.pred).stem),
        "hparams_source": first.get("hparams_source"),
        **metrics,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_out = out.with_suffix(".csv")
    pd.DataFrame([payload]).to_csv(csv_out, index=False)
    print(f"Wrote {out}")
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
