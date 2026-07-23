from copy import deepcopy
from pathlib import Path

import yaml

from fuxictr_ext.fairjob.stage2_pilot_gate import evaluate_pilot_gate


ROOT = Path(__file__).resolve().parents[2]


def _gate():
    return yaml.safe_load(
        (ROOT / "configs/fairjob/stage2_pilot_gate.yaml").read_text(
            encoding="utf-8"
        )
    )


def _evidence():
    seeds = {}
    for seed in (2019, 2020, 2021):
        seeds[str(seed)] = {
            "node_deltas": {
                "cross": -0.007,
                "fusion": -0.006,
                "embedding": 0.001,
                "logit": 0.001,
            },
            "joint_deltas": {"cross_dnn": -0.006},
            "auc_delta": -0.002,
            "calibrated_nllh_delta": 0.0004,
            "calibrated_brier_delta": 0.00004,
            "class_gap_ratio": 0.9,
            "pairwise_ranking_loss_ratio": 1.01,
            "collapse": False,
            "worst_group_material_degradation": False,
        }
    return {
        "target_nodes": ["cross", "fusion"],
        "joint_targets": ["cross_dnn"],
        "non_target_nodes": ["embedding", "logit"],
        "seeds": seeds,
        "control_superiority": {
            "final_only_no_gate": True,
            "multi_layer_no_gate": True,
            "final_only_path_gate": True,
            "structured_random_gate": True,
            "matched_capacity_regularization": True,
        },
    }


def test_full_pass_requires_containment_utility_and_controls():
    result = evaluate_pilot_gate(_evidence(), _gate())
    assert result["decision"] == "full_pass"
    assert result["material_migration_count"] == 0
    assert result["five_seed_expansion"] == "approved"


def test_one_edge_migration_is_partial_and_cannot_expand():
    evidence = _evidence()
    evidence["seeds"]["2019"]["node_deltas"]["embedding"] = 0.006
    evidence["seeds"]["2020"]["node_deltas"]["embedding"] = 0.006
    result = evaluate_pilot_gate(evidence, _gate())
    assert result["decision"] == "partial_pass"
    assert result["material_migration_count"] == 1
    assert result["five_seed_expansion"] == "not_approved"


def test_any_seed_collapse_forces_failure():
    evidence = deepcopy(_evidence())
    evidence["seeds"]["2020"]["collapse"] = True
    result = evaluate_pilot_gate(evidence, _gate())
    assert result["decision"] == "fail"
    assert result["collapse_seeds"] == [2020]
    assert "prediction_collapse" in result["failures"]


def test_missing_control_forces_failure():
    evidence = _evidence()
    evidence["control_superiority"]["structured_random_gate"] = False
    result = evaluate_pilot_gate(evidence, _gate())
    assert result["decision"] == "fail"
    assert "control_superiority" in result["failures"]
