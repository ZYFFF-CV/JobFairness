import json
from pathlib import Path

import numpy as np

from fuxictr_ext.fairjob.run_probe_matrix import probe_sample_is_current


def test_probe_sample_receipt_must_cover_exact_representation_roots(tmp_path: Path):
    workdir = tmp_path / "workdir"
    output = workdir / "probes" / "probe_rows.npz"
    output.parent.mkdir(parents=True)
    np.savez(output, train=np.array([1]), valid=np.array([1]), test=np.array([1]))
    matrix = {
        "workdir_root": str(workdir),
        "representation_roots": {"model_a": "training/model_a/representations"},
        "max_rows_per_split": 50000,
        "seed": 2019,
    }
    metadata = {
        "representation_roots": [
            str(workdir / "training/model_a/representations")
        ],
        "max_rows_per_split": 50000,
        "seed": 2019,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    assert probe_sample_is_current(matrix, output)

    matrix["representation_roots"]["model_b"] = "training/model_b/representations"
    assert not probe_sample_is_current(matrix, output)
