from pathlib import Path

import numpy as np
import pandas as pd

from fuxictr_ext.fairjob.prepare_probe_samples import write_probe_sample
from fuxictr_ext.fairjob.user_disjoint_probe import (
    freeze_user_disjoint_rows,
    stable_user_split,
)


def test_stable_user_split_is_deterministic_and_covers_declared_splits():
    ratios = {"train": 0.6, "valid": 0.2, "test": 0.2}
    first = [stable_user_split(f"user-{index}", 2019, ratios) for index in range(100)]
    second = [stable_user_split(f"user-{index}", 2019, ratios) for index in range(100)]
    assert first == second
    assert set(first) == {"train", "valid", "test"}


def test_user_disjoint_sample_uses_raw_ids_without_overlap(tmp_path: Path):
    ratios = {"train": 0.6, "valid": 0.2, "test": 0.2}
    users = {
        split: next(
            f"{split}-{index}"
            for index in range(10000)
            if stable_user_split(f"{split}-{index}", 7, ratios) == split
        )
        for split in ("train", "valid", "test")
    }
    selected = {}
    for split in ("train", "valid", "test"):
        frame = pd.DataFrame(
            {
                "row_id": np.arange(4),
                "user_id": [users[split], users[split], "shared", "shared"],
                "protected_attribute": [0, 1, 0, 1],
            }
        )
        frame.to_csv(tmp_path / f"{split}.csv", index=False)
        selected[split] = np.arange(4)
    row_sample = tmp_path / "row_sample.npz"
    write_probe_sample(row_sample, selected, {"test": True})
    result, metadata = freeze_user_disjoint_rows(
        processed_dir=tmp_path,
        identity_sources={
            split: f"{split}.csv" for split in ("train", "valid", "test")
        },
        row_sample=row_sample,
        ratios=ratios,
        seed=7,
        chunksize=2,
    )
    assert all(len(result[split]) >= 2 for split in result)
    assert sum(len(rows) for rows in result.values()) == 8
    assert set(metadata["user_overlaps"].values()) == {0}
    assert sorted(
        metadata["split_stats"][split]["selected_users"] for split in result
    ) == [1, 1, 2]
