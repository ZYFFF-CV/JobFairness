"""Fit leakage probes on frozen, sharded FairJob representations.

Probe selection uses only the representation train/validation splits. The test
split is loaded after model selection and is never used to choose capacity or
regularization. This separation is required before a layer AUC can be described
as out-of-sample behavioral-proxy leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.representation_io import validate_representation_directory
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation_root", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--probe_type", choices=["linear", "nonlinear", "both"], default="both")
    parser.add_argument("--max_rows_per_split", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--input_baseline_auc", type=float, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sample_representation(
    split_dir: str | Path,
    representation: str,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream shards and keep a deterministic uniform-priority sample.

    The bounded sampler avoids loading every full-split representation into RAM.
    Random priorities are generated independently of labels, and the retained
    sample is therefore suitable for both protected-proxy groups.
    """

    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    split_dir = Path(split_dir)
    manifest = validate_representation_directory(split_dir)
    if representation not in manifest["representation_shapes"]:
        raise KeyError(f"Representation {representation!r} is not in {split_dir}.")

    rng = np.random.default_rng(seed)
    kept_priority = np.empty(0, dtype=np.float64)
    kept_X = None
    kept_y = np.empty(0, dtype=np.int8)
    kept_row_id = np.empty(0, dtype=np.int64)

    for shard_info in manifest["shards"]:
        with np.load(split_dir / shard_info["path"]) as shard:
            X = np.asarray(shard[representation], dtype=np.float32)
            y = np.asarray(shard["protected_attribute"], dtype=np.int8)
            row_id = np.asarray(shard["row_id"], dtype=np.int64)
        priority = rng.random(len(y))
        if kept_X is None:
            combined_X = X
        else:
            combined_X = np.concatenate([kept_X, X], axis=0)
        combined_y = np.concatenate([kept_y, y])
        combined_row_id = np.concatenate([kept_row_id, row_id])
        combined_priority = np.concatenate([kept_priority, priority])
        if len(combined_y) > max_rows:
            indices = np.argpartition(combined_priority, max_rows - 1)[:max_rows]
            indices = indices[np.argsort(combined_priority[indices])]
            combined_X = combined_X[indices]
            combined_y = combined_y[indices]
            combined_row_id = combined_row_id[indices]
            combined_priority = combined_priority[indices]
        kept_X = combined_X
        kept_y = combined_y
        kept_row_id = combined_row_id
        kept_priority = combined_priority

    if kept_X is None or len(np.unique(kept_y)) != 2:
        raise ValueError(f"Probe split {split_dir} must contain both proxy groups.")
    return kept_X, kept_y, kept_row_id


def candidate_models(probe_type: str, seed: int) -> list[tuple[str, dict, object]]:
    """Return the bounded probe grid used for validation-only selection."""

    candidates = []
    if probe_type in {"linear", "both"}:
        for c_value in (0.1, 1.0, 10.0):
            params = {"C": c_value}
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            )
            candidates.append(("linear", params, model))
    if probe_type in {"nonlinear", "both"}:
        for hidden_units, alpha in (((64,), 1e-4), ((128, 64), 1e-3)):
            params = {"hidden_units": list(hidden_units), "alpha": alpha}
            model = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=hidden_units,
                    alpha=alpha,
                    batch_size=1024,
                    max_iter=100,
                    early_stopping=False,
                    random_state=seed,
                ),
            )
            candidates.append(("nonlinear", params, model))
    return candidates


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe_type: str,
    seed: int,
) -> list[dict]:
    """Select one model per probe family on validation AUC and report test once."""

    selected = []
    families = {"linear", "nonlinear"} if probe_type == "both" else {probe_type}
    for family in sorted(families):
        best = None
        for current_family, params, model in candidate_models(probe_type, seed):
            if current_family != family:
                continue
            model.fit(X_train, y_train)
            valid_prob = model.predict_proba(X_valid)[:, 1]
            valid_auc = float(roc_auc_score(y_valid, valid_prob))
            if best is None or valid_auc > best["valid_auc"]:
                best = {
                    "family": family,
                    "params": params,
                    "model": model,
                    "valid_auc": valid_auc,
                }
        test_prob = best.pop("model").predict_proba(X_test)[:, 1]
        best["test_auc"] = float(roc_auc_score(y_test, test_prob))
        best["test_balanced_accuracy"] = float(
            balanced_accuracy_score(y_test, test_prob >= 0.5)
        )
        selected.append(best)
    return selected


def main() -> None:
    args = parse_args()
    root = Path(args.representation_root)
    sampled = {}
    for index, split in enumerate(("train", "valid", "test")):
        sampled[split] = sample_representation(
            root / split,
            args.representation,
            args.max_rows_per_split,
            args.seed + index,
        )
    X_train, y_train, _ = sampled["train"]
    X_valid, y_valid, _ = sampled["valid"]
    X_test, y_test, _ = sampled["test"]
    probes = fit_probe(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        args.probe_type,
        args.seed,
    )
    for probe in probes:
        baseline = args.input_baseline_auc
        probe["input_baseline_auc"] = baseline
        probe["input_normalized_amplification"] = (
            float((probe["test_auc"] - baseline) / max(baseline - 0.5, 1e-12))
            if baseline is not None
            else None
        )

    payload = {
        "git_commit": git_commit(PROJECT_ROOT),
        "representation_root": str(root),
        "representation": args.representation,
        "seed": args.seed,
        "n_probe_train": int(len(y_train)),
        "n_probe_valid": int(len(y_valid)),
        "n_probe_test": int(len(y_test)),
        "probes": probes,
    }
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
