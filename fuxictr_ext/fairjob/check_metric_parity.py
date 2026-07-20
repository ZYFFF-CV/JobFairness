"""Compare local FairJob metrics with synthetic expectations and optional reference code.

The optional FairJob repository is loaded only for parity checks. Production
evaluation must use the local metric implementation so codex-related reference
files never become runtime dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.metrics import compute_fairjob_metrics
from fuxictr_ext.fairjob.prediction_io import read_meta


def synthetic_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a tiny deterministic frame covering DP, U, and U_TILDE branches."""

    meta = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4, 5],
            "click": [0, 1, 1, 0, 1, 0],
            "protected_attribute": [0, 1, 0, 1, 0, 1],
            "senior": [1, 1, 1, 1, 0, 0],
            "displayrandom": [1, 1, 1, 1, 1, 1],
            "impression_id": [10, 10, 11, 11, 11, 12],
            "product_id": [100, 100, 101, 101, 101, 102],
        }
    )
    pred = pd.DataFrame(
        {
            "row_id": meta["row_id"],
            "model": "SYN",
            "regime": "unaware",
            "mode": "smoke",
            "y_true": meta["click"],
            "y_pred": [0.1, 0.9, 0.2, 0.3, 0.1, 0.7],
        }
    )
    return pred, meta


def import_reference_functions(repo: Path):
    """Dynamically load official functions.py for one-off parity comparison."""

    functions_path = repo / "functions.py"
    if not functions_path.exists():
        return None
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("fairjob_reference_functions", functions_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_metrics(module, pred: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Adapt exported prediction/meta data to the official tensor interface."""

    import torch

    probs = np.column_stack([1.0 - pred["y_pred"].to_numpy(), pred["y_pred"].to_numpy()])
    probs_t = torch.tensor(probs, dtype=torch.float64)
    y_t = torch.tensor(meta["click"].to_numpy(), dtype=torch.long)
    a_t = torch.tensor(meta["protected_attribute"].to_numpy(), dtype=torch.long)
    s_t = torch.tensor(meta["senior"].to_numpy(), dtype=torch.long)
    d_t = torch.tensor(meta["displayrandom"].to_numpy(), dtype=torch.long)
    i_t = torch.tensor(meta["impression_id"].to_numpy(), dtype=torch.long)
    p_t = torch.tensor(meta["product_id"].to_numpy(), dtype=torch.long)
    return {
        "DP": float(module.demographic_parity(probs_t, a_t, s_t).item()),
        "U": float(module.utility(probs_t, y_t, a_t, i_t, d_t).item()),
        "U_TILDE": float(
            module.utility_product(probs_t, y_t, a_t, i_t, d_t, p_t, unbiased_ratio=True).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fairjob_repo", default=None)
    parser.add_argument("--pred", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    if args.pred and args.meta:
        pred = pd.read_csv(args.pred)
        meta = read_meta(args.meta)
    else:
        pred, meta = synthetic_frame()

    # Always run the local implementation first. Reference comparison is optional
    # and must not mask errors in the implementation that the integration uses.
    local = compute_fairjob_metrics(pred, meta)
    result = {
        "local": {k: local[k] for k in ["DP", "U", "U_TILDE"]},
        "reference_code_available": False,
    }

    if args.fairjob_repo:
        try:
            module = import_reference_functions(Path(args.fairjob_repo))
            if module is not None:
                official = official_metrics(module, pred, meta)
                diffs = {k: abs(local[k] - official[k]) for k in official}
                result.update(
                    {
                        "reference_code_available": True,
                        "official": official,
                        "absolute_diff": diffs,
                        "within_tolerance": all(v <= args.tolerance for v in diffs.values()),
                    }
                )
        except Exception as exc:
            result["reference_error"] = str(exc)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
