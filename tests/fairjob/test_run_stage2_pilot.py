import json
from pathlib import Path

import pytest

from fuxictr_ext.fairjob.run_stage2_pilot import (
    DATASET_ID,
    build_command,
    checkpoint_path,
    load_matrix,
    read_completion,
    select_matrix,
)


ROOT = Path(__file__).resolve().parents[2]


def _matrix():
    return load_matrix(ROOT / "configs/fairjob/stage2_ablation_matrix.yaml")


def test_default_selection_preserves_frozen_seed_and_method_order():
    seeds, methods = select_matrix(_matrix(), None, None)
    assert seeds == [2019, 2020, 2021]
    assert methods[0] == "baseline"
    assert methods[-1] == "matched_capacity_regularization"
    assert len(seeds) * len(methods) == 27


def test_subset_selection_cannot_reorder_or_extend_matrix():
    seeds, methods = select_matrix(
        _matrix(),
        [2021, 2019],
        ["structured_random_gate", "baseline"],
    )
    assert seeds == [2019, 2021]
    assert methods == ["baseline", "structured_random_gate"]
    with pytest.raises(ValueError):
        select_matrix(_matrix(), [99], ["baseline"])
    with pytest.raises(ValueError):
        select_matrix(_matrix(), [2019], ["unregistered"])


def test_dependent_command_uses_exact_same_seed_baseline(tmp_path):
    command = build_command(
        "python",
        tmp_path / "config",
        tmp_path / "pilot",
        0,
        2020,
        "multi_layer_path_gate_full",
    )
    checkpoint = checkpoint_path(tmp_path / "pilot", 2020)
    assert "--backbone_checkpoint" in command
    assert str(checkpoint) in command
    assert DATASET_ID in command

    baseline = build_command(
        "python",
        tmp_path / "config",
        tmp_path / "pilot",
        0,
        2020,
        "baseline",
    )
    assert "--backbone_checkpoint" not in baseline


def test_completion_requires_complete_status_and_exact_commit(tmp_path):
    assert read_completion(tmp_path, "abc") == "pending"
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps({"status": "running", "git_commit": "abc"}),
        encoding="utf-8",
    )
    assert read_completion(tmp_path, "abc") == "incomplete"
    manifest.write_text(
        json.dumps({"status": "complete", "git_commit": "old"}),
        encoding="utf-8",
    )
    assert read_completion(tmp_path, "abc") == "stale"
    manifest.write_text(
        json.dumps({"status": "complete", "git_commit": "abc"}),
        encoding="utf-8",
    )
    assert read_completion(tmp_path, "abc") == "complete"
