"""Run bounded FairJob outcome, selection, decomposition, and bootstrap diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.bootstrap import cluster_bootstrap
from fuxictr_ext.fairjob.dp_decomposition import decompose_dp
from fuxictr_ext.fairjob.metrics import binary_log_loss, roc_auc, utility, utility_product
from fuxictr_ext.fairjob.metrics_extended import (
    compute_extended_metrics,
    demographic_parity_details,
)
from fuxictr_ext.fairjob.prediction_io import check_prediction_alignment, read_meta
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest
from fuxictr_ext.fairjob.selection_protocols import select_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--context_columns", default="displayrandom,rank")
    parser.add_argument("--bootstrap_repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def scope_metrics(frame: pd.DataFrame) -> dict:
    """Compute outcome and calibration metrics on one declared row scope."""

    metrics = compute_extended_metrics(frame)
    metrics.update(
        {
            "U": utility(frame),
            "U_TILDE": utility_product(frame, unbiased_ratio=True),
        }
    )
    return metrics


def bootstrap_statistic(frame: pd.DataFrame) -> dict:
    """Return the bounded metric set used in screening confidence intervals."""

    dp = demographic_parity_details(frame)
    return {
        "DP_signed": dp["DP_signed"],
        "DP_abs": dp["DP_abs"],
        "AUC": roc_auc(frame["y_true"], frame["y_pred"]),
        "NLLH": binary_log_loss(frame["y_true"], frame["y_pred"]),
    }


def proxy_measurement_sensitivity(
    frame: pd.DataFrame,
    rates: tuple[float, ...] = (0.05, 0.1, 0.2),
    repeats: int = 20,
    seed: int = 2019,
) -> dict:
    """Perturb evaluator proxy labels without changing model predictions.

    Symmetric and one-direction flips test label-noise sensitivity. Random
    missingness evaluates complete cases after dropping the selected proxy
    labels. These are measurement diagnostics for a behavioral proxy, not a
    simulation of verified gender-label corruption.
    """

    if repeats < 1:
        raise ValueError("Proxy sensitivity repeats must be positive.")
    rng = np.random.default_rng(seed)
    original = frame["protected_attribute"].to_numpy(dtype=np.int8)
    modes = ("symmetric_flip", "group_0_to_1", "group_1_to_0", "random_missing")
    results = {}
    for mode in modes:
        mode_results = {}
        for rate in rates:
            if not 0.0 < rate < 1.0:
                raise ValueError("Proxy perturbation rates must be between zero and one.")
            values = []
            for _ in range(repeats):
                draw = rng.random(len(frame)) < rate
                perturbed = original.copy()
                keep = np.ones(len(frame), dtype=bool)
                if mode == "symmetric_flip":
                    perturbed[draw] = 1 - perturbed[draw]
                elif mode == "group_0_to_1":
                    selected = draw & (original == 0)
                    perturbed[selected] = 1
                elif mode == "group_1_to_0":
                    selected = draw & (original == 1)
                    perturbed[selected] = 0
                else:
                    keep = ~draw
                modified = frame.loc[keep].copy()
                modified["protected_attribute"] = perturbed[keep]
                dp = demographic_parity_details(modified)
                values.append((dp["DP_signed"], dp["DP_abs"], len(modified)))
            array = np.asarray(values, dtype=float)
            mode_results[str(rate)] = {
                "DP_signed_mean": float(array[:, 0].mean()),
                "DP_signed_std": float(array[:, 0].std(ddof=1))
                if repeats > 1
                else 0.0,
                "DP_abs_mean": float(array[:, 1].mean()),
                "rows_mean": float(array[:, 2].mean()),
            }
        results[mode] = mode_results
    return {
        "rates": list(rates),
        "repeats": repeats,
        "seed": seed,
        "perturbation_target": "behavioral_protected_proxy_evaluation_labels_only",
        "modes": results,
    }


def build_frame(pred_path: str | Path, meta_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Restore raw evaluator metadata in the canonical prediction row order."""

    alignment = check_prediction_alignment(pred_path, meta_path)
    pred = pd.read_csv(pred_path)
    frame = read_meta(meta_path)
    frame["y_true"] = pred["y_true"].to_numpy(dtype=float)
    frame["y_pred"] = pred["y_pred"].to_numpy(dtype=float)
    frame["click"] = frame["click"].to_numpy(dtype=float)
    labels = {}
    for key in ("model", "regime", "mode", "expid", "protocol", "seed"):
        if key not in pred.columns:
            continue
        value = pred.iloc[0].get(key)
        if pd.isna(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()
        labels[key] = value
    return frame, {"alignment": alignment, "labels": labels}


def run_diagnostics(
    frame: pd.DataFrame,
    context_columns: list[str],
    bootstrap_repeats: int,
    seed: int,
) -> dict:
    """Evaluate available selection scopes without inventing propensities."""

    all_logged = select_rows(frame, "all_logged")
    random_display = select_rows(frame, "random_display")
    senior = all_logged[all_logged["senior"].astype(int) == 1].copy()
    scopes = {
        "all_logged": scope_metrics(all_logged),
        "random_display": scope_metrics(random_display),
    }
    decomposition = {
        reference: decompose_dp(senior, context_columns, reference=reference)
        for reference in ("pooled", "group_average")
    }
    bootstraps = {
        "all_logged": cluster_bootstrap(
            all_logged,
            "impression_id",
            bootstrap_statistic,
            repeats=bootstrap_repeats,
            seed=seed,
        ),
        "random_display": cluster_bootstrap(
            random_display,
            "impression_id",
            bootstrap_statistic,
            repeats=bootstrap_repeats,
            seed=seed + 1,
        ),
    }
    position_status = "unavailable_missing_external_propensity"
    if "logging_propensity" in frame.columns:
        position_status = "unavailable_protocol_not_frozen"
    return {
        "selection_scopes": scopes,
        "dp_context_decomposition": decomposition,
        "cluster_bootstrap": bootstraps,
        "proxy_measurement_sensitivity": proxy_measurement_sensitivity(
            all_logged, repeats=bootstrap_repeats, seed=seed + 2
        ),
        "position_corrected": position_status,
        "claim_boundary": "descriptive_association_not_causal_effect",
    }


def main() -> None:
    args = parse_args()
    contexts = [value.strip() for value in args.context_columns.split(",") if value.strip()]
    frame, metadata = build_frame(args.pred, args.meta)
    payload = {
        "git_commit": git_commit(PROJECT_ROOT),
        "prediction_path": args.pred,
        "meta_path": args.meta,
        **metadata,
        **run_diagnostics(frame, contexts, args.bootstrap_repeats, args.seed),
    }
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
