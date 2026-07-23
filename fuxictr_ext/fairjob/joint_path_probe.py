"""Fit preregistered Stage2 probes on node or concatenated path representations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.prepare_probe_samples import load_probe_row_ids
from fuxictr_ext.fairjob.proxy_probe import sample_representation
from fuxictr_ext.fairjob.representation_graph import load_representation_graph
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


FAMILIES = ("linear", "matched_capacity_nonlinear", "independent_bounded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation_root", required=True)
    parser.add_argument("--graph", default="configs/fairjob/stage2_representation_graph.yaml")
    parser.add_argument("--protocol", default="configs/fairjob/stage2_probe_protocol.yaml")
    parser.add_argument("--representation", default=None)
    parser.add_argument("--joint_path", default=None)
    parser.add_argument("--probe_sample", required=True)
    parser.add_argument("--max_rows_per_split", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--skip_shard_validation", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sample_joint_representation(
    split_dir: str | Path,
    members: list[str],
    max_rows: int,
    seed: int,
    selected_row_ids: np.ndarray,
    validate_shards: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned member arrays and concatenate them in frozen graph order."""

    arrays = []
    canonical_y = None
    canonical_rows = None
    for index, member in enumerate(members):
        X, y, row_ids = sample_representation(
            split_dir,
            member,
            max_rows=max_rows,
            seed=seed,
            selected_row_ids=selected_row_ids,
            validate_shards=validate_shards if index == 0 else False,
        )
        if canonical_y is None:
            canonical_y = y
            canonical_rows = row_ids
        elif not np.array_equal(canonical_y, y) or not np.array_equal(
            canonical_rows, row_ids
        ):
            raise ValueError(f"Joint member {member} is not row-aligned.")
        arrays.append(X)
    return np.concatenate(arrays, axis=1), canonical_y, canonical_rows


def _candidate_models(protocol: dict, seed: int):
    spec = protocol["probe_families"]
    for C in spec["linear"]["C"]:
        yield (
            "linear",
            {"C": float(C)},
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(C),
                    class_weight="balanced",
                    max_iter=int(spec["linear"]["max_iter"]),
                    random_state=seed,
                ),
            ),
        )
    nonlinear = spec["matched_capacity_nonlinear"]
    for hidden in nonlinear["hidden_layer_sizes"]:
        for alpha in nonlinear["alpha"]:
            yield (
                "matched_capacity_nonlinear",
                {
                    "hidden_layer_sizes": list(hidden),
                    "alpha": float(alpha),
                },
                make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=tuple(hidden),
                        alpha=float(alpha),
                        batch_size=1024,
                        max_iter=int(nonlinear["max_iter"]),
                        early_stopping=bool(nonlinear["early_stopping"]),
                        n_iter_no_change=20,
                        random_state=seed,
                    ),
                ),
            )
    independent = spec["independent_bounded"]
    for max_depth in independent["max_depth"]:
        yield (
            "independent_bounded",
            {"max_depth": int(max_depth)},
            ExtraTreesClassifier(
                n_estimators=int(independent["n_estimators"]),
                max_depth=int(max_depth),
                min_samples_leaf=int(independent["min_samples_leaf"]),
                class_weight="balanced",
                n_jobs=int(independent.get("n_jobs", 1)),
                random_state=seed,
            ),
        )


def fit_stage2_probes(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    protocol: dict,
    seed: int,
) -> list[dict]:
    """Select capacity and AUC orientation on validation, then read test once."""

    results = []
    for family in FAMILIES:
        best = None
        for current_family, params, model in _candidate_models(protocol, seed):
            if current_family != family:
                continue
            model.fit(X_train, y_train)
            valid_probability = model.predict_proba(X_valid)[:, 1]
            raw_valid_auc = float(roc_auc_score(y_valid, valid_probability))
            orientation = 1 if raw_valid_auc >= 0.5 else -1
            valid_auc = max(raw_valid_auc, 1.0 - raw_valid_auc)
            candidate = {
                "family": family,
                "params": params,
                "model": model,
                "orientation": orientation,
                "valid_auc": valid_auc,
                "raw_valid_auc": raw_valid_auc,
            }
            if best is None or (
                candidate["valid_auc"],
                json.dumps(params, sort_keys=True),
            ) > (
                best["valid_auc"],
                json.dumps(best["params"], sort_keys=True),
            ):
                best = candidate
        model = best.pop("model")
        test_probability = model.predict_proba(X_test)[:, 1]
        if best["orientation"] < 0:
            test_probability = 1.0 - test_probability
        best["test_auc"] = float(roc_auc_score(y_test, test_probability))
        best["test_balanced_accuracy"] = float(
            balanced_accuracy_score(y_test, test_probability >= 0.5)
        )
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(estimator, "n_iter_"):
            n_iter = int(np.max(np.atleast_1d(estimator.n_iter_)))
            best["n_iter"] = n_iter
            best["converged"] = n_iter < estimator.max_iter
        else:
            best["n_iter"] = None
            best["converged"] = True
        results.append(best)
    return results


def main() -> None:
    args = parse_args()
    if bool(args.representation) == bool(args.joint_path):
        raise ValueError("Pass exactly one of --representation or --joint_path.")
    graph = load_representation_graph(args.graph)
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    if args.joint_path:
        try:
            members = graph["joint_paths"][args.joint_path]["members"]
        except KeyError as error:
            raise KeyError(f"Unknown joint path: {args.joint_path}") from error
        target_name = f"joint_{args.joint_path}"
    else:
        members = [args.representation]
        target_name = args.representation

    sampled = {}
    root = Path(args.representation_root)
    for split_index, split in enumerate(("train", "valid", "test")):
        selected = load_probe_row_ids(args.probe_sample, split)
        sampled[split] = sample_joint_representation(
            root / split,
            members,
            max_rows=args.max_rows_per_split,
            seed=args.seed + split_index,
            selected_row_ids=selected,
            validate_shards=not args.skip_shard_validation,
        )
    X_train, y_train, train_rows = sampled["train"]
    X_valid, y_valid, valid_rows = sampled["valid"]
    X_test, y_test, test_rows = sampled["test"]
    probes = fit_stage2_probes(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        protocol,
        seed=args.seed,
    )
    payload = {
        "version": 1,
        "stage": "S2-P3",
        "git_commit": git_commit(PROJECT_ROOT),
        "representation_root": str(root),
        "target": target_name,
        "members": members,
        "probe_sample": args.probe_sample,
        "seed": args.seed,
        "rows": {
            "train": int(len(train_rows)),
            "valid": int(len(valid_rows)),
            "test": int(len(test_rows)),
        },
        "probes": probes,
        "claim_boundary": "bounded_probe_family_decodability",
    }
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
