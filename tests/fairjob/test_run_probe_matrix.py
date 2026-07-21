import json
from pathlib import Path

import numpy as np

from fuxictr_ext.fairjob.run_probe_matrix import (
    ablation_commands,
    probe_sample_is_current,
    read_matrix,
)


ROOT = Path(__file__).resolve().parents[2]


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


def test_interaction_ablation_matrix_uses_frozen_probe_results():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_1_probe_matrix.yaml")
    commands = ablation_commands(matrix, "interaction_screening")
    assert len(commands) == 9
    for _, command, output in commands:
        assert "--baseline_result" in command
        assert "--probe_sample" in command
        assert command[command.index("--mask_fraction") + 1] == "0.1"
        assert command[command.index("--random_repeats") + 1] == "10"
        assert output.parent.name == "ablations"
