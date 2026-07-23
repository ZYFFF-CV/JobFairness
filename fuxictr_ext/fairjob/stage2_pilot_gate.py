"""Apply the frozen Stage2 three-seed containment and utility gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from fuxictr_ext.fairjob.run_manifest import write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", default="configs/fairjob/stage2_pilot_gate.yaml"
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _mean(values) -> float:
    return float(statistics.mean(float(value) for value in values))


def _seed_values(evidence: dict, key: str, seeds: list[int]) -> list[float]:
    values = []
    for seed in seeds:
        try:
            values.append(float(evidence["seeds"][str(seed)][key]))
        except KeyError as error:
            raise ValueError(f"Pilot evidence is missing {key} for seed {seed}.") from error
    return values


def evaluate_pilot_gate(evidence: dict, gate: dict) -> dict:
    """Return a deterministic full/partial/fail decision and every sub-gate."""

    if gate.get("version") != 1:
        raise ValueError("Unsupported Stage2 pilot gate version.")
    seeds = [int(seed) for seed in gate["seeds"]]
    if set(evidence.get("seeds", {})) != {str(seed) for seed in seeds}:
        raise ValueError("Pilot evidence must contain exactly the frozen seeds.")
    containment = gate["containment"]
    material = float(containment["material_delta"])
    target_names = list(evidence.get("target_nodes", []))
    joint_names = list(evidence.get("joint_targets", []))
    non_target_names = list(evidence.get("non_target_nodes", []))
    if not target_names or not joint_names:
        raise ValueError("Pilot evidence requires target nodes and joint targets.")

    target_checks = {}
    for name in target_names:
        values = [
            float(evidence["seeds"][str(seed)]["node_deltas"][name])
            for seed in seeds
        ]
        target_checks[name] = {
            "values": values,
            "mean": _mean(values),
            "mean_pass": _mean(values)
            <= float(containment["target_mean_delta_max"]),
            "direction_pass": all(value < 0 for value in values),
        }
    joint_checks = {}
    for name in joint_names:
        values = [
            float(evidence["seeds"][str(seed)]["joint_deltas"][name])
            for seed in seeds
        ]
        joint_checks[name] = {
            "values": values,
            "mean": _mean(values),
            "mean_pass": _mean(values)
            <= float(containment["joint_target_mean_delta_max"]),
            "no_seed_material_increase": all(value < material for value in values),
        }
    migration_checks = {}
    for name in non_target_names:
        values = [
            float(evidence["seeds"][str(seed)]["node_deltas"][name])
            for seed in seeds
        ]
        mean_value = _mean(values)
        material_seeds = sum(value >= material for value in values)
        migration_checks[name] = {
            "values": values,
            "mean": mean_value,
            "material_seed_count": material_seeds,
            "pass": (
                mean_value
                < float(containment["non_target_mean_delta_max_exclusive"])
                and material_seeds
                <= int(containment["non_target_material_seed_count_max"])
            ),
        }
    migration_count = sum(
        not item["pass"] for item in migration_checks.values()
    )

    utility = gate["utility"]
    auc = _seed_values(evidence, "auc_delta", seeds)
    nllh = _seed_values(evidence, "calibrated_nllh_delta", seeds)
    brier = _seed_values(evidence, "calibrated_brier_delta", seeds)
    class_gap = _seed_values(evidence, "class_gap_ratio", seeds)
    ranking = _seed_values(evidence, "pairwise_ranking_loss_ratio", seeds)
    utility_checks = {
        "auc": {
            "values": auc,
            "mean": _mean(auc),
            "pass": (
                _mean(auc) >= float(utility["auc"]["mean_delta_min"])
                and min(auc) >= float(utility["auc"]["per_seed_delta_min"])
            ),
        },
        "calibrated_nllh": {
            "values": nllh,
            "mean": _mean(nllh),
            "pass": (
                _mean(nllh)
                <= float(utility["calibrated_nllh"]["mean_delta_max"])
                and max(nllh)
                <= float(utility["calibrated_nllh"]["per_seed_delta_max"])
            ),
        },
        "calibrated_brier": {
            "values": brier,
            "mean": _mean(brier),
            "pass": (
                _mean(brier)
                <= float(utility["calibrated_brier"]["mean_delta_max"])
                and max(brier)
                <= float(utility["calibrated_brier"]["per_seed_delta_max"])
            ),
        },
        "class_gap": {
            "values": class_gap,
            "mean": _mean(class_gap),
            "pass": (
                _mean(class_gap)
                >= float(utility["class_gap_ratio"]["mean_min"])
                and min(class_gap)
                >= float(
                    utility["class_gap_ratio"]["per_seed_catastrophic_min"]
                )
            ),
        },
        "pairwise_ranking": {
            "values": ranking,
            "mean": _mean(ranking),
            "pass": (
                _mean(ranking)
                <= float(
                    utility["pairwise_ranking_loss_ratio"]["mean_max"]
                )
                and max(ranking)
                <= float(
                    utility["pairwise_ranking_loss_ratio"]["per_seed_max"]
                )
            ),
        },
    }
    collapse_seeds = [
        seed
        for seed in seeds
        if bool(evidence["seeds"][str(seed)].get("collapse", False))
    ]
    worst_group_pass = all(
        not bool(
            evidence["seeds"][str(seed)].get(
                "worst_group_material_degradation", False
            )
        )
        for seed in seeds
    )
    required_controls = gate["control_superiority"]["required_controls"]
    control_checks = {
        name: bool(evidence.get("control_superiority", {}).get(name, False))
        for name in required_controls
    }

    targets_pass = all(
        item["mean_pass"] and item["direction_pass"]
        for item in target_checks.values()
    )
    joints_pass = all(
        item["mean_pass"] and item["no_seed_material_increase"]
        for item in joint_checks.values()
    )
    utility_pass = all(item["pass"] for item in utility_checks.values())
    controls_pass = all(control_checks.values())
    no_collapse = not collapse_seeds
    full_pass = (
        targets_pass
        and joints_pass
        and migration_count
        == int(containment["full_pass_migration_count"])
        and utility_pass
        and no_collapse
        and worst_group_pass
        and controls_pass
    )
    partial_pass = (
        not full_pass
        and targets_pass
        and joints_pass
        and migration_count
        <= int(containment["partial_pass_migration_count"])
        and utility_pass
        and no_collapse
        and worst_group_pass
        and controls_pass
    )
    if full_pass:
        decision = "full_pass"
    elif partial_pass:
        decision = "partial_pass"
    else:
        decision = "fail"
    failures = []
    if not targets_pass:
        failures.append("target_containment")
    if not joints_pass:
        failures.append("joint_containment")
    if migration_count > int(containment["full_pass_migration_count"]):
        failures.append("material_migration")
    failures.extend(
        f"utility:{name}"
        for name, item in utility_checks.items()
        if not item["pass"]
    )
    if collapse_seeds:
        failures.append("prediction_collapse")
    if not worst_group_pass:
        failures.append("worst_group_quality")
    if not controls_pass:
        failures.append("control_superiority")
    return {
        "version": 1,
        "stage": "S2-P4",
        "decision": decision,
        "five_seed_expansion": (
            "approved" if decision == "full_pass" else "not_approved"
        ),
        "target_checks": target_checks,
        "joint_checks": joint_checks,
        "migration_checks": migration_checks,
        "material_migration_count": migration_count,
        "utility_checks": utility_checks,
        "collapse_seeds": collapse_seeds,
        "worst_group_quality_pass": worst_group_pass,
        "control_checks": control_checks,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    gate = yaml.safe_load(Path(args.gate).read_text(encoding="utf-8"))
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    result = evaluate_pilot_gate(evidence, gate)
    write_run_manifest(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
