"""Compare targeted representation masking with matched random masking.

This M3 diagnostic freezes the nonlinear probe family selected by the earlier
validation-only screen. Targeted units are ranked using protected-group
standardized mean differences on probe-train rows only. The fitted probe is
then held fixed while test representations are masked to their train means.
The result measures leakage sensitivity, not CTR outcome improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.prepare_probe_samples import load_probe_row_ids
from fuxictr_ext.fairjob.proxy_probe import candidate_models, sample_representation
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation_root", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--probe_sample", required=True)
    parser.add_argument("--baseline_result", required=True)
    parser.add_argument("--mask_fraction", type=float, default=0.1)
    parser.add_argument("--random_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def standardized_mean_difference(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank units by train-only protected-group separation.

    The pooled within-group scale prevents naturally high-variance units from
    dominating solely because of their numeric range. Zero-variance dimensions
    receive a finite score through the machine-epsilon floor.
    """

    group_0 = X[y == 0].astype(np.float64, copy=False)
    group_1 = X[y == 1].astype(np.float64, copy=False)
    if not len(group_0) or not len(group_1):
        raise ValueError("Both protected groups are required for unit ranking.")
    pooled_variance = (group_0.var(axis=0) + group_1.var(axis=0)) / 2.0
    scale = np.sqrt(np.maximum(pooled_variance, np.finfo(np.float64).eps))
    return np.abs(group_1.mean(axis=0) - group_0.mean(axis=0)) / scale


def mask_columns(
    X: np.ndarray, indices: np.ndarray, replacement: np.ndarray
) -> np.ndarray:
    """Return a copy with selected units set to their probe-train means."""

    masked = X.copy()
    masked[:, indices] = replacement[indices]
    return masked


def probe_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate a frozen protected-proxy probe on one representation matrix."""

    probability = model.predict_proba(X)[:, 1]
    return {
        "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, probability >= 0.5)
        ),
    }


def frozen_probe_from_result(
    result_path: str | Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[object, dict, dict]:
    """Refit the preregistered probe capacity and verify its baseline result."""

    reference_payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if len(reference_payload["probes"]) != 1:
        raise ValueError("Ablation requires one frozen probe family per result.")
    reference = reference_payload["probes"][0]
    selected_model = None
    for family, params, model in candidate_models(reference["family"], seed):
        if family == reference["family"] and params == reference["params"]:
            selected_model = model
            break
    if selected_model is None:
        raise ValueError("Reference probe parameters are not in the current grid.")

    selected_model.fit(X_train, y_train)
    validation = probe_metrics(selected_model, X_valid, y_valid)
    test = probe_metrics(selected_model, X_test, y_test)
    difference = test["auc"] - float(reference["test_auc"])
    if abs(difference) > 1e-8:
        raise ValueError(
            "Refitted frozen probe does not reproduce its baseline test AUC: "
            f"difference={difference}"
        )
    estimator = selected_model.steps[-1][1]
    fit_metadata = {
        "family": reference["family"],
        "params": reference["params"],
        "n_iter": int(np.max(np.atleast_1d(estimator.n_iter_))),
        "converged": bool(
            np.max(np.atleast_1d(estimator.n_iter_)) < estimator.max_iter
        ),
        "validation": validation,
        "test": test,
        "reference_test_auc": float(reference["test_auc"]),
        "reference_auc_difference": float(difference),
    }
    return selected_model, reference_payload, fit_metadata


def run_masking_intervention(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask_fraction: float,
    random_repeats: int,
    seed: int,
) -> dict:
    """Measure targeted leakage reduction against matched random masks."""

    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be between zero and one.")
    if random_repeats < 1:
        raise ValueError("random_repeats must be positive.")
    dimensions = X_train.shape[1]
    masked_dimensions = max(1, int(round(dimensions * mask_fraction)))
    replacement = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scores = standardized_mean_difference(X_train, y_train)
    targeted_indices = np.argsort(-scores, kind="stable")[:masked_dimensions]

    baseline = probe_metrics(model, X_test, y_test)
    targeted = probe_metrics(
        model,
        mask_columns(X_test, targeted_indices, replacement),
        y_test,
    )
    targeted_drop = baseline["auc"] - targeted["auc"]

    rng = np.random.default_rng(seed)
    random_results = []
    for repeat in range(random_repeats):
        indices = np.sort(
            rng.choice(dimensions, size=masked_dimensions, replace=False)
        )
        metrics = probe_metrics(
            model, mask_columns(X_test, indices, replacement), y_test
        )
        metrics.update(
            {
                "repeat": repeat,
                "auc_drop": float(baseline["auc"] - metrics["auc"]),
            }
        )
        random_results.append(metrics)

    random_drops = np.asarray(
        [result["auc_drop"] for result in random_results], dtype=np.float64
    )
    return {
        "mask_fraction": mask_fraction,
        "dimensions": dimensions,
        "masked_dimensions": masked_dimensions,
        "mask_value": "probe_train_mean",
        "ranking": "absolute_train_standardized_mean_difference",
        "targeted_indices": targeted_indices.tolist(),
        "targeted_score_min": float(scores[targeted_indices].min()),
        "baseline": baseline,
        "targeted": {
            **targeted,
            "auc_drop": float(targeted_drop),
        },
        "random": {
            "repeats": random_repeats,
            "auc_drop_mean": float(random_drops.mean()),
            "auc_drop_std": float(random_drops.std(ddof=1))
            if random_repeats > 1
            else 0.0,
            "auc_drop_min": float(random_drops.min()),
            "auc_drop_max": float(random_drops.max()),
            "results": random_results,
        },
        "targeted_extra_auc_drop": float(targeted_drop - random_drops.mean()),
        "empirical_one_sided_p": float(
            (1 + np.count_nonzero(random_drops >= targeted_drop))
            / (random_repeats + 1)
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.representation_root)
    sampled = {}
    for split in ("train", "valid", "test"):
        sampled[split] = sample_representation(
            root / split,
            args.representation,
            max_rows=sys.maxsize,
            seed=args.seed,
            selected_row_ids=load_probe_row_ids(args.probe_sample, split),
            validate_shards=False,
        )
    X_train, y_train, _ = sampled["train"]
    X_valid, y_valid, _ = sampled["valid"]
    X_test, y_test, _ = sampled["test"]
    model, reference_payload, frozen_probe = frozen_probe_from_result(
        args.baseline_result,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        args.seed,
    )
    intervention = run_masking_intervention(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        args.mask_fraction,
        args.random_repeats,
        args.seed,
    )
    payload = {
        "git_commit": git_commit(PROJECT_ROOT),
        "representation_root": str(root),
        "representation": args.representation,
        "probe_sample": args.probe_sample,
        "baseline_result": args.baseline_result,
        "baseline_result_git_commit": reference_payload["git_commit"],
        "seed": args.seed,
        "n_probe_train": int(len(y_train)),
        "n_probe_valid": int(len(y_valid)),
        "n_probe_test": int(len(y_test)),
        "frozen_probe": frozen_probe,
        "intervention": intervention,
        "scope": "representation_leakage_diagnostic_not_ctr_outcome_intervention",
    }
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
