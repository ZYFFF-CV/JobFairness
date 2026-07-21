from pathlib import Path

import numpy as np

from fuxictr_ext.fairjob.representation_io import (
    RepresentationShardWriter,
    validate_representation_directory,
)


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
