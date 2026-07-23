from pathlib import Path

import numpy as np
import pytest

from fuxictr_ext.fairjob.representation_graph import load_representation_graph
from fuxictr_ext.fairjob.stage2_checkpoint_probe import (
    build_target_plan,
    load_probe_protocol,
    load_protocol_rows,
    protocol_view,
    required_representation_names,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_sample(path, train, valid, test):
    np.savez(
        path,
        train=np.asarray(train, dtype=np.int64),
        valid=np.asarray(valid, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
    )


def test_target_plan_keeps_alias_visible_without_duplicate_source():
    graph = load_representation_graph(
        ROOT / "configs/fairjob/stage2_representation_graph.yaml"
    )
    protocol = load_probe_protocol(
        ROOT / "configs/fairjob/stage2_probe_protocol.yaml"
    )
    plan = build_target_plan(graph, protocol)
    names = required_representation_names(plan)

    assert plan["dcnv2_final"]["alias_of"] == "fusion_pre_logit"
    assert "fusion_pre_logit" in names
    assert "dcnv2_final" not in names
    assert plan["joint_cross_block_increments"]["members"] == [
        "cross_increment_0",
        "cross_increment_1",
        "cross_increment_2",
    ]
    assert "joint_legacy_suppression_components" not in plan


def test_user_disjoint_sample_must_be_subset_of_row_sample(tmp_path):
    row = tmp_path / "row.npz"
    user = tmp_path / "user.npz"
    _write_sample(row, [1, 2, 3], [4, 5], [6, 7])
    _write_sample(user, [1, 3], [5], [7])
    samples = load_protocol_rows(row, user)
    assert samples["user_disjoint"]["train"].tolist() == [1, 3]

    _write_sample(user, [1, 9], [5], [7])
    with pytest.raises(ValueError, match="not a subset"):
        load_protocol_rows(row, user)


def test_protocol_view_preserves_requested_raw_row_order():
    collected = {
        "row_id": np.asarray([10, 20, 30], dtype=np.int64),
        "y_true": np.asarray([0, 1, 0], dtype=np.int8),
        "protected_attribute": np.asarray([1, 0, 1], dtype=np.int8),
        "y_pred": np.asarray([0.1, 0.8, 0.2], dtype=np.float32),
        "representations": {
            "node": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        },
    }
    view = protocol_view(collected, np.asarray([10, 30], dtype=np.int64))
    assert view["row_id"].tolist() == [10, 30]
    assert view["protected_attribute"].tolist() == [1, 1]
    assert view["representations"]["node"].reshape(-1).tolist() == [1.0, 3.0]
