import json
from pathlib import Path

import numpy as np

from fuxictr_ext.fairjob.joint_path_probe import (
    fit_stage2_probes,
    sample_joint_representation,
)
from fuxictr_ext.fairjob.representation_io import RepresentationShardWriter


def _write_representations(path: Path):
    writer = RepresentationShardWriter(path, shard_rows=6)
    rows = np.arange(60, dtype=np.int64)
    protected = (rows % 2).astype(np.int8)
    writer.add(
        {
            "row_id": rows,
            "y_true": (rows % 3 == 0).astype(np.float32),
            "protected_attribute": protected,
            "y_pred": np.full(60, 0.1, dtype=np.float32),
            "left": np.column_stack((protected, rows / 60)),
            "right": np.column_stack((1 - protected, rows % 5)),
        }
    )
    shards = writer.close()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "rows": 60,
                "representation_shapes": {"left": [2], "right": [2]},
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )


def test_joint_representation_preserves_frozen_member_order(tmp_path):
    split = tmp_path / "train"
    _write_representations(split)
    selected = np.array([1, 4, 9, 12, 31], dtype=np.int64)
    X, y, row_ids = sample_joint_representation(
        split,
        ["right", "left"],
        max_rows=10,
        seed=7,
        selected_row_ids=selected,
        validate_shards=False,
    )
    assert np.array_equal(row_ids, selected)
    assert np.array_equal(y, selected % 2)
    assert np.array_equal(X[:, 0], 1 - (selected % 2))
    assert np.array_equal(X[:, 2], selected % 2)


def test_stage2_probe_orientation_is_selected_on_validation_only():
    rng = np.random.default_rng(11)
    y_train = np.tile([0, 1], 100)
    y_valid = np.tile([0, 1], 50)
    y_test = np.tile([0, 1], 50)
    X_train = (-y_train + rng.normal(0, 0.01, len(y_train))).reshape(-1, 1)
    X_valid = (-y_valid + rng.normal(0, 0.01, len(y_valid))).reshape(-1, 1)
    X_test = (-y_test + rng.normal(0, 0.01, len(y_test))).reshape(-1, 1)
    protocol = {
        "probe_families": {
            "linear": {"C": [1.0], "max_iter": 200},
            "matched_capacity_nonlinear": {
                "hidden_layer_sizes": [[4]],
                "alpha": [0.001],
                "max_iter": 50,
                "early_stopping": False,
            },
            "independent_bounded": {
                "n_estimators": 10,
                "max_depth": [3],
                "min_samples_leaf": 2,
                "n_jobs": 1,
            },
        }
    }
    results = fit_stage2_probes(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        protocol,
        seed=7,
    )
    assert {result["family"] for result in results} == {
        "linear",
        "matched_capacity_nonlinear",
        "independent_bounded",
    }
    assert all(result["valid_auc"] >= 0.5 for result in results)
    assert all(result["test_auc"] >= 0.9 for result in results)
