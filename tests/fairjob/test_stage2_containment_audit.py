import copy

import pytest

from fuxictr_ext.fairjob.stage2_containment_audit import evaluate_early_stop


def _probe(family, test_auc):
    return {
        "family": family,
        "test_auc": test_auc,
        "valid_auc": test_auc - 0.01,
        "params": {},
    }


def _result(node_auc, joint_auc):
    protocol_node = {"probes": [_probe("linear", node_auc), _probe("tree", 0.55)]}
    protocol_joint = {
        "probes": [_probe("linear", joint_auc), _probe("tree", 0.54)]
    }
    return {
        "stage": "S2-P3",
        "status": "complete",
        "audit_git_commit": "audit",
        "training_git_commit": "train",
        "seed": 2019,
        "probe_seed": 2019,
        "graph_hash": "graph",
        "protocol_hash": "protocol",
        "row_sample_sha256": "row",
        "user_sample_sha256": "user",
        "prediction_integrity": {
            "max_abs_prediction_diff": 1e-16,
            "tolerance": 1e-6,
        },
        "targets": {
            "node": {
                "protocols": {
                    "row_disjoint": copy.deepcopy(protocol_node),
                    "user_disjoint": copy.deepcopy(protocol_node),
                }
            },
            "joint_path": {
                "protocols": {
                    "row_disjoint": copy.deepcopy(protocol_joint),
                    "user_disjoint": copy.deepcopy(protocol_joint),
                }
            },
        },
    }


def test_nonnegative_target_delta_is_irreversible_early_fail():
    baseline = _result(0.60, 0.60)
    primary = _result(0.61, 0.59)
    result = evaluate_early_stop(
        baseline,
        primary,
        target_nodes=["node"],
        joint_targets=["path"],
    )
    assert result["decision"] == "early_fail"
    assert result["remaining_checkpoint_probes"] == "not_approved"
    assert any("target_nonnegative" in item for item in result["irreversible_reasons"])


def test_all_negative_targets_and_nonincreasing_joints_continue():
    baseline = _result(0.60, 0.60)
    primary = _result(0.59, 0.599)
    result = evaluate_early_stop(
        baseline,
        primary,
        target_nodes=["node"],
        joint_targets=["path"],
    )
    assert result["decision"] == "continue"
    assert result["irreversible_reasons"] == []


def test_material_joint_increase_is_irreversible():
    baseline = _result(0.60, 0.60)
    primary = _result(0.59, 0.606)
    result = evaluate_early_stop(
        baseline,
        primary,
        target_nodes=["node"],
        joint_targets=["path"],
    )
    assert result["decision"] == "early_fail"
    assert any(
        "joint_material_increase" in item
        for item in result["irreversible_reasons"]
    )


def test_pair_identity_mismatch_is_rejected():
    baseline = _result(0.60, 0.60)
    primary = _result(0.59, 0.59)
    primary["row_sample_sha256"] = "different"
    with pytest.raises(ValueError, match="row_sample_sha256"):
        evaluate_early_stop(
            baseline,
            primary,
            target_nodes=["node"],
            joint_targets=["path"],
        )
