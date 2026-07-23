from fuxictr_ext.fairjob.build_stage1_2_report import (
    evaluate_directions,
    frozen_stage2_matrix,
)


def test_direction_a_requires_converged_redistribution_without_clean_reduction():
    evidence = {
        "clean_leakage_reduction_methods": [],
        "redistribution_methods": ["global_suppression", "selective_suppression"],
        "prediction_collapse_seeds": {"adversarial": 3},
        "dp_role": "DP_near_floor_for_current_setting",
        "group_blind_calibration_matches_suppression": True,
        "all_probes_converged": True,
        "protocol_sign_consistent_runs": 16,
        "protocol_total_runs": 18,
    }
    result = evaluate_directions(evidence)
    assert result["selected_direction"] == "A"
    assert result["directions"]["A"]["selected"]
    assert not any(
        result["directions"][name]["selected"] for name in ("B", "C", "D", "E")
    )
    assert result["stage2_method_recovery"] == "not_approved"


def test_stage2_matrix_prevents_proxy_inference_dependency_and_early_expansion():
    matrix = frozen_stage2_matrix()
    assert matrix["inference_proxy_dependency"] is False
    assert matrix["expansion"]["five_seed_DCNv2"].startswith("conditional")
    assert "logging_policy_model" in matrix["not_approved"]
    assert "DP_abs" in matrix["secondary_endpoints"]
