"""Measure behavioral-proxy leakage in raw FairJob model inputs.

Categorical values are target encoded with the protected proxy as the probe
target. The encoder is fit only on the probe training split; validation selects
logistic regularization, and test is read exactly once for final reporting.
This is a diagnostic leakage baseline, not a click-prediction model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, TargetEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.prepare_probe_samples import load_probe_row_ids
from fuxictr_ext.fairjob.proxy_probe import fit_probe
from fuxictr_ext.fairjob.protocols import features_for_protocol, load_protocols
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


SPLITS = ("train", "valid", "test")
ID_FEATURES = ("user_id", "product_id")
FEATURE_SETS = (
    "all_model_inputs",
    "ids_only",
    "non_id",
    "no_user_id",
    "no_product_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/FairJob/processed")
    parser.add_argument("--feature_schema", default="configs/fairjob/feature_schema.yaml")
    parser.add_argument("--task_protocols", default="configs/fairjob/task_protocols.yaml")
    parser.add_argument(
        "--fairness_protocols", default="configs/fairjob/fairness_protocols.yaml"
    )
    parser.add_argument("--protocol", default="pre_ranking")
    parser.add_argument("--regime", default="proxy_excluded")
    parser.add_argument("--feature_set", choices=FEATURE_SETS, default="all_model_inputs")
    parser.add_argument("--probe_type", choices=["linear", "nonlinear"], default="linear")
    parser.add_argument("--probe_sample", required=True)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def select_feature_set(features: list[str], feature_set: str) -> list[str]:
    """Apply the preregistered identity-feature control to model inputs."""

    if feature_set == "all_model_inputs":
        selected = features
    elif feature_set == "ids_only":
        selected = [name for name in features if name in ID_FEATURES]
    elif feature_set == "non_id":
        selected = [name for name in features if name not in ID_FEATURES]
    elif feature_set == "no_user_id":
        selected = [name for name in features if name != "user_id"]
    elif feature_set == "no_product_id":
        selected = [name for name in features if name != "product_id"]
    else:
        raise KeyError(f"Unknown input feature set: {feature_set}")
    if not selected:
        raise ValueError(f"Input feature set {feature_set!r} is empty.")
    return selected


def load_aligned_split(
    path: str | Path,
    row_ids: np.ndarray,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read raw values and restore the exact frozen probe row order."""

    columns = ["row_id", "protected_attribute", *feature_columns]
    # Keep one copy if a future proxy-included schema names the target directly.
    columns = list(dict.fromkeys(columns))
    frame = pd.read_csv(path, usecols=columns)
    if frame["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row_id values in {path}.")
    aligned = frame.set_index("row_id").reindex(row_ids)
    if aligned.isnull().all(axis=1).any():
        missing = row_ids[aligned.isnull().all(axis=1).to_numpy()][:5]
        raise ValueError(f"Frozen probe rows missing from {path}: {missing.tolist()}")
    y = aligned["protected_attribute"].to_numpy(dtype=np.int8)
    return aligned[feature_columns].reset_index(drop=True), y


def encode_inputs(
    frames: dict[str, pd.DataFrame],
    targets: dict[str, np.ndarray],
    categorical: list[str],
    numeric: list[str],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit train-only encoders and transform all three probe splits.

    ``TargetEncoder.fit_transform`` uses internal cross-fitting for the training
    rows. Validation and test receive only ``transform`` from that fitted train
    encoder, preventing protected-target leakage across data splits.
    """

    encoded_parts = {split: [] for split in SPLITS}
    metadata = {
        "categorical_features": categorical,
        "numeric_features": numeric,
        "target_encoder_fit_split": "train",
        "target_encoder_train_transform": "five_fold_cross_fit",
    }
    if categorical:
        values = {
            split: frames[split][categorical].astype("string").fillna("__NA__")
            for split in SPLITS
        }
        encoder = TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=5,
            shuffle=True,
            random_state=seed,
        )
        encoded_parts["train"].append(
            encoder.fit_transform(values["train"], targets["train"])
        )
        encoded_parts["valid"].append(encoder.transform(values["valid"]))
        encoded_parts["test"].append(encoder.transform(values["test"]))
    if numeric:
        scaler = StandardScaler()
        train_numeric = frames["train"][numeric].to_numpy(dtype=np.float64)
        encoded_parts["train"].append(scaler.fit_transform(train_numeric))
        encoded_parts["valid"].append(
            scaler.transform(frames["valid"][numeric].to_numpy(dtype=np.float64))
        )
        encoded_parts["test"].append(
            scaler.transform(frames["test"][numeric].to_numpy(dtype=np.float64))
        )
    encoded = {
        split: np.column_stack(encoded_parts[split]).astype(np.float32, copy=False)
        for split in SPLITS
    }
    if any(not np.isfinite(value).all() for value in encoded.values()):
        raise ValueError("Encoded raw inputs contain non-finite values.")
    metadata["encoded_dimensions"] = int(encoded["train"].shape[1])
    return encoded, metadata


def fit_input_probe(
    encoded: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    seed: int,
    probe_type: str = "linear",
) -> dict:
    """Select matched probe capacity on validation and report test once."""

    if probe_type == "nonlinear":
        # Reuse the exact hidden-representation MLP grid so amplification is not
        # an artifact of comparing a nonlinear layer probe with a linear input
        # baseline. Target encoding remains fit on train only in both cases.
        return fit_probe(
            encoded["train"],
            targets["train"],
            encoded["valid"],
            targets["valid"],
            encoded["test"],
            targets["test"],
            probe_type="nonlinear",
            seed=seed,
        )[0]

    best = None
    for c_value in (0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        model.fit(encoded["train"], targets["train"])
        valid_prob = model.predict_proba(encoded["valid"])[:, 1]
        valid_auc = float(roc_auc_score(targets["valid"], valid_prob))
        if best is None or valid_auc > best["valid_auc"]:
            best = {
                "family": "target_encoded_linear_input",
                "params": {"C": c_value},
                "valid_auc": valid_auc,
                "model": model,
            }
    test_prob = best.pop("model").predict_proba(encoded["test"])[:, 1]
    best["test_auc"] = float(roc_auc_score(targets["test"], test_prob))
    best["test_balanced_accuracy"] = float(
        balanced_accuracy_score(targets["test"], test_prob >= 0.5)
    )
    return best


def main() -> None:
    args = parse_args()
    schema = yaml.safe_load(Path(args.feature_schema).read_text(encoding="utf-8"))
    task, fairness = load_protocols(args.task_protocols, args.fairness_protocols)
    features, task_name, regime_name = features_for_protocol(
        schema, task, fairness, args.protocol, args.regime
    )
    features = select_feature_set(features, args.feature_set)
    categorical_universe = set(schema["categorical_id_features"])
    categorical_universe.update(schema["categorical_features"])
    categorical_universe.add(schema["protected_feature_for_model"])
    categorical = [name for name in features if name in categorical_universe]
    numeric = [name for name in features if name not in categorical_universe]

    data_dir = Path(args.data_dir)
    frames = {}
    targets = {}
    for split in SPLITS:
        row_ids = load_probe_row_ids(args.probe_sample, split)
        frames[split], targets[split] = load_aligned_split(
            data_dir / f"{split}.csv", row_ids, features
        )
        if len(np.unique(targets[split])) != 2:
            raise ValueError(f"Input probe split {split} must contain both groups.")
    encoded, encoding_metadata = encode_inputs(
        frames, targets, categorical, numeric, args.seed
    )
    probe = fit_input_probe(encoded, targets, args.seed, probe_type=args.probe_type)
    payload = {
        "git_commit": git_commit(PROJECT_ROOT),
        "data_dir": str(data_dir),
        "task_protocol": task_name,
        "proxy_regime": regime_name,
        "feature_set": args.feature_set,
        "features": features,
        "probe_sample": args.probe_sample,
        "seed": args.seed,
        "probe_type": args.probe_type,
        "n_probe_train": int(len(targets["train"])),
        "n_probe_valid": int(len(targets["valid"])),
        "n_probe_test": int(len(targets["test"])),
        "encoding": encoding_metadata,
        "probe": probe,
    }
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
