import json

from fuxictr_ext.fairjob.build_m5_report import (
    METHODS,
    POSITION_STATUS,
    SEEDS,
    build_report,
    evaluate_gate,
)


def _payload(scale):
    scope = {
        "overall_AUC": 0.8 - scale,
        "overall_NLLH": 0.05 + scale,
        "overall_ECE": 0.003 + scale,
        "DP_abs": 0.01 + scale,
        "group_gap_ECE": 0.002 + scale,
        "U": 0.02 - scale,
        "U_TILDE": 0.03 - scale,
    }
    return {
        "git_commit": "abc123",
        "position_corrected": POSITION_STATUS,
        "selection_scopes": {
            "all_logged": scope,
            "random_display": scope,
        },
        "dp_context_decomposition": {
            "pooled": {"within_context_dp_signed": 0.005 + scale}
        },
    }


def test_m5_report_requires_complete_six_by_three_matrix(tmp_path):
    for method_index, method in enumerate(METHODS):
        for seed in SEEDS:
            (tmp_path / f"{method}_seed{seed}.json").write_text(
                json.dumps(_payload(method_index / 1000)), encoding="utf-8"
            )
    report = build_report(tmp_path)
    assert report["status"] == "complete_m5b_gate_failed"
    assert report["position_corrected"] == POSITION_STATUS
    assert report["m5b_gate"]["m5b_five_seed_expansion"] == "not_approved"


def test_gate_passes_only_joint_fairness_and_performance_improvement():
    metric_names = (
        "all_AUC",
        "all_NLLH",
        "all_ECE",
        "all_DP",
        "all_ECE_gap",
        "random_AUC",
        "random_NLLH",
        "random_ECE",
        "random_DP",
        "random_ECE_gap",
        "context_DP",
        "U",
        "U_TILDE",
    )
    baseline = {name: {"mean": 1.0, "std": 0.0} for name in metric_names}
    aggregates = {"baseline": baseline}
    for method in METHODS[1:]:
        aggregates[method] = {
            name: {"mean": value["mean"], "std": value["std"]}
            for name, value in baseline.items()
        }
    candidate = aggregates["selective_suppression"]
    for name in ("all_DP", "random_DP", "context_DP", "all_NLLH"):
        candidate[name]["mean"] = 0.9
    for name in ("all_AUC", "U", "U_TILDE"):
        candidate[name]["mean"] = 1.1
    gate = evaluate_gate(aggregates)
    assert gate["passed"] is True
    assert gate["passing_methods"] == ["selective_suppression"]
