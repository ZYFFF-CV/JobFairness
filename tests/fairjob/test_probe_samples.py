import json
from pathlib import Path

import numpy as np

from fuxictr_ext.fairjob.prepare_probe_samples import (
    load_probe_row_ids,
    select_row_ids,
    write_probe_sample,
)
from fuxictr_ext.fairjob.proxy_probe import sample_representation
from fuxictr_ext.fairjob.representation_io import RepresentationShardWriter


def test_probe_sample_round_trip_and_representation_alignment(tmp_path: Path):
    split_dir = tmp_path / "representations" / "train"
    writer = RepresentationShardWriter(split_dir, shard_rows=3)
    writer.add(
        {
            "row_id": np.array([1, 3, 4, 8, 10]),
            "y_true": np.array([0, 1, 0, 1, 0]),
            "protected_attribute": np.array([0, 1, 1, 0, 1]),
            "y_pred": np.linspace(0.1, 0.9, 5),
            "hidden": np.arange(10).reshape(5, 2),
        }
    )
    shards = writer.close()
    (split_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "rows": 5,
                "representation_shapes": {"hidden": [2]},
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    sample_path = tmp_path / "probe_rows.npz"
    selected = {
        "train": np.array([3, 8, 10]),
        "valid": np.array([0]),
        "test": np.array([0]),
    }
    write_probe_sample(sample_path, selected, {"test": True})
    row_ids = load_probe_row_ids(sample_path, "train")
    X, y, loaded_ids = sample_representation(
        split_dir,
        "hidden",
        max_rows=3,
        seed=7,
        selected_row_ids=row_ids,
        validate_shards=False,
    )
    assert np.array_equal(loaded_ids, selected["train"])
    assert np.array_equal(y, np.array([1, 0, 1]))
    assert np.array_equal(X, np.array([[2, 3], [6, 7], [8, 9]], dtype=np.float32))


def test_select_row_ids_is_deterministic_and_sorted():
    row_ids = np.arange(100, dtype=np.int64)
    first = select_row_ids(row_ids, max_rows=20, seed=11)
    second = select_row_ids(row_ids, max_rows=20, seed=11)
    assert np.array_equal(first, second)
    assert len(first) == 20
    assert np.all(np.diff(first) > 0)
