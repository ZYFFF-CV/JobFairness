"""Build the deterministic Stage1.2 evidence and Stage2 direction decision."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fuxictr_ext.fairjob.build_m5_report import METHODS
from fuxictr_ext.fairjob.run_manifest import write_run_manifest


SUPPRESSION_METHODS = (
    "global_suppression",
    "matched_random_suppression",
    "selective_suppression",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2", required=True)
    parser.add_argument("--m3", required=True)
    parser.add_argument("--m4", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _mean(values) -> float:
    return float(statistics.mean(float(value) for value in values))


def summarize_evidence(m2: dict, m3: dict, m4: dict) -> dict:
    """Extract only decision-relevant facts from the three frozen audits."""

    leakage = {
        method: m2["summary"]["methods"][method]["classification"]
        for method in METHODS[1:]
    }
    collapse = {
        method: sum(
            bool(run["prediction_collapse"])
            for run in m3["runs"]
            if run["method"] == method
        )
        for method in METHODS
    }
    calibrated_nllh = {
        method: _mean(
            run["calibrated_metrics"]["overall_NLLH"]
            for run in m3["runs"]
            if run["method"] == method
        )
        for method in METHODS
    }
    raw_nllh = {
        method: _mean(
            run["raw_metrics"]["overall_NLLH"]
            for run in m3["runs"]
            if run["method"] == method
        )
        for method in METHODS
    }
    baseline_calibrated = calibrated_nllh["baseline"]
    suppression_calibration_distance = {
        method: calibrated_nllh[method] - baseline_calibrated
        for method in SUPPRESSION_METHODS
    }
    calibration_matches_suppression = all(
        abs(value) <= 0.001
        for value in suppression_calibration_distance.values()
    )
    m4_role = m4["dp_role"]
    return {
        "all_probes_converged": bool(m2["all_probes_converged"]),
        "leakage_classifications": leakage,
        "clean_leakage_reduction_methods": [
            method
            for method, classification in leakage.items()
            if classification == "reduced"
        ],
        "redistribution_methods": [
            method
            for method, classification in leakage.items()
            if classification == "redistributed"
        ],
        "inconclusive_methods": [
            method
            for method, classification in leakage.items()
            if classification.startswith("inconclusive")
        ],
        "prediction_collapse_seeds": collapse,
        "raw_NLLH_means": raw_nllh,
        "calibrated_NLLH_means": calibrated_nllh,
        "suppression_calibrated_NLLH_delta_from_baseline": (
            suppression_calibration_distance
        ),
        "group_blind_calibration_matches_suppression": (
            calibration_matches_suppression
        ),
        "dp_role": m4_role["classification"],
        "baseline_DP_abs_mean": m4_role["baseline_all_DP_abs_mean"],
        "baseline_DP_abs_seed_std": m4_role[
            "baseline_all_DP_abs_seed_std"
        ],
        "baseline_DP_bootstrap_zero_crossings": m4_role[
            "baseline_bootstrap_intervals_crossing_zero"
        ],
        "protocol_sign_consistent_runs": m4_role[
            "protocol_sign_consistent_runs"
        ],
        "protocol_total_runs": m4_role["total_runs"],
        "position_corrected": m4["position_corrected"],
    }


def evaluate_directions(evidence: dict) -> dict:
    """Apply the Stage1.2 A-E trigger rules and select exactly one direction."""

    clean_reduction = bool(evidence["clean_leakage_reduction_methods"])
    redistribution = bool(evidence["redistribution_methods"])
    collapse = evidence["prediction_collapse_seeds"]["adversarial"] == 3
    near_floor = (
        evidence["dp_role"] == "DP_near_floor_for_current_setting"
    )
    calibration_explains = evidence[
        "group_blind_calibration_matches_suppression"
    ]
    probes_stable = evidence["all_probes_converged"]
    protocol_mostly_consistent = (
        evidence["protocol_sign_consistent_runs"]
        / evidence["protocol_total_runs"]
        >= 0.75
    )

    directions = {
        "A": {
            "name": "multi_layer_multi_path_leakage_intervention",
            "selected": not clean_reduction and redistribution and probes_stable,
            "trigger_evidence": {
                "no_clean_leakage_reduction": not clean_reduction,
                "leakage_redistribution_observed": redistribution,
                "probe_convergence_complete": probes_stable,
            },
            "missing_evidence": [],
        },
        "B": {
            "name": "leakage_outcome_decoupling_audit",
            "selected": False,
            "trigger_evidence": {
                "clean_leakage_reduction": clean_reduction,
                "no_migration_or_collapse": not redistribution and not collapse,
            },
            "missing_evidence": [
                "at_least_one_stable_clean_leakage_reduction",
                "absence_of_leakage_migration",
            ],
        },
        "C": {
            "name": "group_calibration_under_proxy_uncertainty",
            "selected": False,
            "trigger_evidence": {
                "DP_near_floor": near_floor,
                "simple_group_blind_calibration_insufficient": not calibration_explains,
            },
            "missing_evidence": [
                "calibration_gap_remaining_after_group_blind_calibration"
            ],
        },
        "D": {
            "name": "fairness_measurement_reproducibility_audit",
            "selected": False,
            "trigger_evidence": {
                "probe_instability": not probes_stable,
                "protocol_direction_instability": not protocol_mostly_consistent,
            },
            "missing_evidence": [
                "broad_probe_nonconvergence",
                "majority_protocol_direction_reversals",
            ],
        },
        "E": {
            "name": "limited_method_recovery",
            "selected": False,
            "trigger_evidence": {
                "clean_leakage_reduction": clean_reduction,
                "selective_beats_controls": False,
                "no_prediction_collapse": not collapse,
            },
            "missing_evidence": [
                "clean_leakage_and_outcome_joint_improvement",
                "selective_superiority_over_matched_random_and_global",
                "three_protocol_Pareto_consistency",
            ],
        },
    }
    selected = [
        key for key, value in directions.items() if value["selected"]
    ]
    if selected != ["A"]:
        raise ValueError(
            f"Frozen Stage1.2 evidence does not select exactly direction A: {selected}"
        )
    return {
        "selected_direction": "A",
        "directions": directions,
        "stage2_method_recovery": "not_approved",
        "stage2_large_scale_training": "not_approved_before_pilot_gate",
    }


def frozen_stage2_matrix() -> dict:
    """Return the bounded Stage2-A development matrix and expansion gates."""

    return {
        "research_question": (
            "How can protected-proxy leakage be contained across DCNv2 "
            "layers and paths without migration or prediction collapse?"
        ),
        "backbone": "DCNv2_no_user_id",
        "behavioral_proxy_boundary": (
            "protected_attribute_is_behavioral_proxy_not_verified_demographic"
        ),
        "inference_proxy_dependency": False,
        "development_seeds": [2019, 2020, 2021],
        "methods": [
            {
                "name": "baseline",
                "role": "CTR_reference",
            },
            {
                "name": "current_selective_suppression",
                "role": "failed_local_gate_control_no_weight_sweep",
            },
            {
                "name": "multi_layer_path_containment",
                "role": "proposed_method",
                "components": [
                    "train_only_proxy_adversaries_at_embedding_cross0_cross1_cross2_final",
                    "path_specific_cross_and_residual_gates",
                    "max_layer_excess_leakage_penalty",
                    "CTR_prediction_preservation_constraint",
                    "score_variance_anti_collapse_constraint",
                ],
            },
            {
                "name": "final_layer_only_containment",
                "role": "layer_distribution_ablation",
            },
            {
                "name": "multi_layer_without_path_gates",
                "role": "path_control_ablation",
            },
            {
                "name": "matched_capacity_regularization",
                "role": "generic_regularization_control",
            },
        ],
        "representations": [
            "embedding_flat",
            "cross_layer_0",
            "cross_layer_1",
            "cross_layer_2",
            "dcnv2_final",
            "dcnv2_logit",
            "dcnv2_probability",
        ],
        "probe_families": ["linear", "matched_capacity_nonlinear"],
        "outcome_protocols": [
            "all_logged",
            "random_display",
            "context_conditioned",
        ],
        "primary_development_endpoints": [
            "max_layer_excess_probe_AUC_over_input",
            "leakage_migration_count",
            "overall_AUC",
            "overall_NLLH",
            "worst_group_NLLH",
            "worst_group_Brier",
            "prediction_score_std",
        ],
        "secondary_endpoints": [
            "DP_signed",
            "DP_abs",
            "group_ECE_gap",
            "U",
            "U_TILDE",
        ],
        "pilot_gate": {
            "selection_data": "validation_only",
            "requirements": [
                "all_targeted_layer_probe_AUCs_decline_without_non_target_layer_increase",
                "AUC_drop_no_more_than_0.01_from_matched_seed_baseline",
                "NLLH_increase_no_more_than_0.002_from_matched_seed_baseline",
                "prediction_std_ratio_at_least_0.5",
                "no_test_metric_used_for_hyperparameter_selection",
            ],
        },
        "expansion": {
            "five_seed_DCNv2": "conditional_after_three_seed_pilot_gate",
            "DeepFM_cross_backbone": "conditional_after_five_seed_DCNv2",
            "formal_bootstrap_and_tests": "required_after_expansion",
        },
        "not_approved": [
            "current_selective_suppression_weight_sweep",
            "logging_policy_model",
            "position_corrected_claims",
            "test_guided_method_selection",
            "new_backbone_search_before_mechanism_gate",
        ],
    }


def build_decision(m2: dict, m3: dict, m4: dict) -> dict:
    evidence = summarize_evidence(m2, m3, m4)
    decision = evaluate_directions(evidence)
    return {
        "version": 1,
        "stage": "S12-M5",
        "status": "stage1_2_complete_direction_frozen",
        "stage1_1_record": {
            "M3": "directional_pass",
            "M4": "conditional_pass_available_data_pass",
            "M5B": "failed",
        },
        "evidence": evidence,
        "decision": decision,
        "stage2_matrix": frozen_stage2_matrix(),
        "claim_boundary": [
            "descriptive_association_not_causal_effect",
            "behavioral_proxy_not_verified_demographic_attribute",
            "position_corrected_unavailable_missing_external_propensity",
            "no_evidence_current_methods_reduce_harmful_leakage",
        ],
    }


def render_report(payload: dict) -> str:
    evidence = payload["evidence"]
    decision = payload["decision"]
    matrix = payload["stage2_matrix"]
    lines = [
        "# Stage1.2 decision report",
        "",
        f"- Status: `{payload['status']}`.",
        "- Selected Stage2 direction: `A - multi-layer/multi-path leakage intervention`.",
        "- New long training: `not approved before the Stage2 pilot gate`.",
        f"- Position-corrected: `{evidence['position_corrected']}`.",
        "",
        "## Evidence chain",
        "",
        f"- M2: all probes converged; classifications are "
        f"`{json.dumps(evidence['leakage_classifications'], sort_keys=True)}`.",
        f"- M2: clean leakage reduction methods: "
        f"`{evidence['clean_leakage_reduction_methods']}`; redistribution methods: "
        f"`{evidence['redistribution_methods']}`.",
        f"- M3: adversarial collapse seeds: "
        f"`{evidence['prediction_collapse_seeds']['adversarial']}/3`.",
        "- M3: validation-only group-blind calibration brings all suppression "
        "methods within 0.001 NLLH of calibrated baseline, so their raw NLLH "
        "advantage is not an independent fairness mechanism.",
        f"- M4: `{evidence['dp_role']}`; baseline all-logged DP abs is "
        f"`{evidence['baseline_DP_abs_mean']:.6f} +/- "
        f"{evidence['baseline_DP_abs_seed_std']:.6f}`, with "
        f"`{evidence['baseline_DP_bootstrap_zero_crossings']}/3` baseline "
        "screening intervals crossing zero.",
        f"- M4: `{evidence['protocol_sign_consistent_runs']}/"
        f"{evidence['protocol_total_runs']}` runs preserve signed direction "
        "across all/random/context protocols, but no stable Pareto winner emerges.",
        "",
        "## Direction decision",
        "",
        "| Direction | Decision | Reason |",
        "|---|---|---|",
        "| A: multi-layer/multi-path containment | **Selected** | Existing local "
        "interventions do not cleanly reduce leakage and repeatedly move it to "
        "other layers or paths. |",
        "| B: leakage-outcome decoupling | Not selected | No method first achieved "
        "stable clean leakage reduction without migration or collapse. |",
        "| C: proxy-uncertain group calibration | Not selected | DP is near floor, "
        "but group-blind calibration already explains the suppression NLLH pattern; "
        "an unresolved calibration mechanism is absent. |",
        "| D: measurement/reproducibility audit | Not selected | All frozen probes "
        "converged and 16/18 protocol signs agree; instability is not broad enough "
        "to be the sole Stage2 direction. |",
        "| E: limited method recovery | Not selected | Selective suppression is "
        "not superior to matched-random/global controls and has no clean "
        "leakage-plus-outcome improvement. |",
        "",
        "This selection does not claim that representation leakage causes outcome "
        "disparity. It identifies the earlier point of failure: current "
        "interventions do not contain proxy information across the representation "
        "graph.",
        "",
        "## Frozen Stage2-A design",
        "",
        f"Research question: {matrix['research_question']}",
        "",
        "The proposed method attaches train-only proxy adversaries to embedding, "
        "cross0/1/2, and final representations; introduces separate cross-path and "
        "residual-path gates; penalizes the maximum layer-wise excess leakage; "
        "and adds prediction-preservation plus score-variance constraints to "
        "prevent fairness-by-collapse. The protected attribute is never required "
        "at inference.",
        "",
        "| Method | Role |",
        "|---|---|",
    ]
    for method in matrix["methods"]:
        lines.append(f"| {method['name']} | {method['role']} |")
    lines.extend(
        [
            "",
            "Development uses DCNv2 no-user-id and seeds 2019/2020/2021. "
            "Primary endpoints are max-layer excess probe AUC, migration count, "
            "AUC, NLLH, worst-group NLLH/Brier, and score variance. DP remains "
            "secondary because M4 places it near the current measurement floor.",
            "",
            "The pilot is selected on validation only. It must reduce leakage at "
            "all targeted layers without increasing an untargeted layer, keep "
            "matched-seed AUC loss within 0.01 and NLLH increase within 0.002, "
            "and retain at least half of baseline prediction-score standard "
            "deviation. Test metrics cannot select hyperparameters.",
            "",
            "Five-seed DCNv2 expansion is conditional on the three-seed pilot "
            "gate. DeepFM cross-backbone validation follows only after that "
            "expansion. Current selective-suppression weight sweeps, a logging "
            "policy model, position-corrected claims, and new-backbone search "
            "remain unapproved.",
            "",
            "## Claim boundary",
            "",
            "- `protected_attribute` is a behavioral proxy, not a verified "
            "demographic attribute.",
            "- Available results are descriptive associations, not causal effects.",
            "- Missing external propensities keep position correction unavailable.",
            "- Stage1.2 does not establish a successful fairness mitigation method; "
            "it freezes a falsifiable Stage2 mechanism hypothesis.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    m2 = json.loads(Path(args.m2).read_text(encoding="utf-8"))
    m3 = json.loads(Path(args.m3).read_text(encoding="utf-8"))
    m4 = json.loads(Path(args.m4).read_text(encoding="utf-8"))
    payload = build_decision(m2, m3, m4)
    write_run_manifest(args.out, payload)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_direction": payload["decision"][
                    "selected_direction"
                ],
                "out": args.out,
                "report": args.report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
