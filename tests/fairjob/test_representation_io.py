from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuxictr_ext.fairjob.representation_io import (
    RepresentationShardWriter,
    selection_positions,
    validate_exported_predictions,
    validate_representation_directory,
)


def test_frozen_row_ids_map_to_ordered_source_positions():
    positions = selection_positions(
        np.array([10, 20, 30, 40]), np.array([10, 30, 40])
    )
    assert np.array_equal(positions, np.array([0, 2, 3]))
    with pytest.raises(ValueError, match="missing"):
        selection_positions(np.array([10, 20, 30]), np.array([10, 25]))


def test_representation_writer_splits_and_validates(tmp_path: Path):
    writer = RepresentationShardWriter(tmp_path / "test", shard_rows=3)
    writer.add(
        {
            "row_id": np.arange(2),
            "y_true": np.array([0, 1]),
            "protected_attribute": np.array([0, 1]),
            "y_pred": np.array([0.2, 0.8]),
            "embedding_flat": np.ones((2, 4)),
        }
    )
    writer.add(
        {
            "row_id": np.arange(2, 5),
            "y_true": np.array([0, 1, 0]),
            "protected_attribute": np.array([1, 0, 1]),
            "y_pred": np.array([0.3, 0.7, 0.4]),
            "embedding_flat": np.ones((3, 4)) * 2,
        }
    )
    shards = writer.close()
    assert [item["rows"] for item in shards] == [3, 2]

    manifest = {
        "version": 1,
        "split": "test",
        "rows": 5,
        "representation_shapes": {"embedding_flat": [4]},
        "shards": shards,
    }
    import json

    (tmp_path / "test" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    validated = validate_representation_directory(tmp_path / "test")
    assert validated["rows"] == 5


def test_representation_validator_accepts_sampled_increasing_ids(tmp_path: Path):
    writer = RepresentationShardWriter(tmp_path / "sample", shard_rows=2)
    writer.add(
        {
            "row_id": np.array([1, 4, 9]),
            "y_true": np.array([0, 1, 0]),
            "protected_attribute": np.array([0, 1, 1]),
            "y_pred": np.array([0.1, 0.8, 0.2]),
            "embedding_flat": np.ones((3, 2)),
        }
    )
    shards = writer.close()
    import json

    (tmp_path / "sample" / "manifest.json").write_text(
        json.dumps({"version": 1, "rows": 3, "shards": shards}), encoding="utf-8"
    )
    assert validate_representation_directory(tmp_path / "sample")["rows"] == 3


def test_exported_predictions_align_by_raw_row_id(tmp_path: Path):
    split_dir = tmp_path / "test"
    writer = RepresentationShardWriter(split_dir, shard_rows=2)
    writer.add(
        {
            "row_id": np.array([2, 7]),
            "y_true": np.array([0, 1], dtype=np.float32),
            "protected_attribute": np.array([1, 0], dtype=np.int8),
            "y_pred": np.array([0.25, 0.75], dtype=np.float32),
            "dcnv2_final": np.ones((2, 3), dtype=np.float32),
        }
    )
    shards = writer.close()
    import json

    (split_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "rows": 2, "shards": shards}),
        encoding="utf-8",
    )
    reference = tmp_path / "prediction.csv"
    pd.DataFrame(
        {
            "row_id": [7, 2, 9],
            "y_true": [1, 0, 0],
            "y_pred": [0.75, 0.25, 0.1],
        }
    ).to_csv(reference, index=False)

    result = validate_exported_predictions(split_dir, reference)
    assert result["rows"] == 2
    assert result["max_abs_prediction_diff"] == 0.0


def test_exported_prediction_validation_rejects_probability_drift(tmp_path: Path):
    split_dir = tmp_path / "test"
    writer = RepresentationShardWriter(split_dir)
    writer.add(
        {
            "row_id": np.array([3]),
            "y_true": np.array([1], dtype=np.float32),
            "protected_attribute": np.array([0], dtype=np.int8),
            "y_pred": np.array([0.8], dtype=np.float32),
        }
    )
    shards = writer.close()
    import json

    (split_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "rows": 1, "shards": shards}),
        encoding="utf-8",
    )
    reference = tmp_path / "prediction.csv"
    pd.DataFrame({"row_id": [3], "y_true": [1], "y_pred": [0.7]}).to_csv(
        reference, index=False
    )

    with pytest.raises(ValueError, match="changed predictions"):
        validate_exported_predictions(split_dir, reference)
