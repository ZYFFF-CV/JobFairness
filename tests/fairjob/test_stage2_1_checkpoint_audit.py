from pathlib import Path

import pytest

from fuxictr_ext.fairjob.stage2_1.checkpoint_audit import (
    frozen_target_names,
    load_stage2_1_protocol,
    max_family_endpoint,
    select_targets,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_protocol_declares_18_targets_and_bounded_selection():
    protocol = load_stage2_1_protocol(
        ROOT / "configs/fairjob/stage2_1_audit_protocol.yaml"
    )
    targets = frozen_target_names(protocol)
    assert len(targets) == 18
    assert targets[0] == "embedding_flat"
    assert targets[-1] == "joint_branch_outputs_with_fusion"
    assert select_targets(
        targets,
        ["joint_cross_dnn_outputs", "embedding_flat"],
    ) == ["embedding_flat", "joint_cross_dnn_outputs"]
    with pytest.raises(ValueError):
        select_targets(targets, ["not_registered"])


def test_max_family_endpoint_is_conservative_across_frozen_families():
    suite = {
        "families": {
            "linear": {
                "auc_selected": {
                    "test_auc": 0.61,
                    "valid_auc": 0.60,
                    "params": {"C": 0.1},
                }
            },
            "matched_capacity_nonlinear": {
                "auc_selected": {
                    "test_auc": 0.65,
                    "valid_auc": 0.64,
                    "params": {"alpha": 0.001},
                }
            },
        }
    }
    result = max_family_endpoint(suite)
    assert result["selected"]["family"] == "matched_capacity_nonlinear"
    assert result["selected"]["test_auc"] == 0.65
