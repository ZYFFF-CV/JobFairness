"""Build the Stage2.1 cross-seed redistribution replication decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest
from fuxictr_ext.fairjob.run_stage2_pilot import expid, run_dir
from fuxictr_ext.fairjob.stage2_1.checkpoint_audit import (
    load_stage2_1_protocol,
)
from fuxictr_ext.fairjob.stage2_1.redistribution_metrics import (
    protocol_consistency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="configs/fairjob/stage2_1_audit_protocol.yaml",
    )
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--pilot_workdir", default=None)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    return parser.parse_args()


def _load_audit(
    workdir: str | Path, seed: int, method: str
) -> dict:
    path = (
        Path(workdir)
        / "probes"
        / f"seed{seed}"
        / method
        / "checkpoint_audit.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Stage2.1 audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Stage2.1 audit is incomplete: {path}")
    return payload


def _max_auc(payload: dict, target: str, protocol: str) -> float:
    return float(
        payload["targets"][target]["protocols"][protocol][
            "max_family_endpoint"
        ]["selected"]["test_auc"]
    )


def _target_contrast(
    baseline: dict,
    primary: dict,
    target: str,
    protocol: str,
) -> dict:
    baseline_suite = baseline["targets"][target]["protocols"][protocol]
    primary_suite = primary["targets"][target]["protocols"][protocol]
    family_specific = {}
    fixed_capacity = {}
    for family in baseline_suite["families"]:
        baseline_family = baseline_suite["families"][family]
        primary_family = primary_suite["families"][family]
        family_specific[family] = (
            float(primary_family["auc_selected"]["test_auc"])
            - float(baseline_family["auc_selected"]["test_auc"])
        )
        fixed_capacity[family] = (
            float(primary_family["fixed_capacity"]["test_auc"])
            - float(baseline_family["fixed_capacity"]["test_auc"])
        )
    baseline_max = _max_auc(baseline, target, protocol)
    primary_max = _max_auc(primary, target, protocol)
    return {
        "baseline_max_auc": baseline_max,
        "primary_max_auc": primary_max,
        "max_family_delta": primary_max - baseline_max,
        "family_specific_delta": family_specific,
        "fixed_capacity_delta": fixed_capacity,
    }


def _gain_contrasts(
    baseline: dict,
    primary: dict,
    field: str,
) -> dict:
    result = {}
    for name, primary_item in primary.get(field, {}).items():
        baseline_item = baseline[field][name]
        result[name] = {}
        for protocol in primary_item["protocols"]:
            result[name][protocol] = {}
            for family, primary_family in primary_item["protocols"][
                protocol
            ].items():
                baseline_family = baseline_item["protocols"][protocol][family]
                result[name][protocol][family] = {}
                for role, primary_role in primary_family.items():
                    metric = (
                        "conditional_predictive_gain"
                        if field == "conditional_gains"
                        else "joint_predictive_gain"
                    )
                    result[name][protocol][family][role] = {
                        "baseline": float(baseline_family[role][metric]),
                        "primary": float(primary_role[metric]),
                        "delta": (
                            float(primary_role[metric])
                            - float(baseline_family[role][metric])
                        ),
                    }
    return result


def _outcome(
    pilot_workdir: str | Path,
    seed: int,
    baseline_method: str,
    primary_method: str,
    collapse_delta_min: float,
) -> dict:
    metrics = {}
    for label, method in (
        ("baseline", baseline_method),
        ("primary", primary_method),
    ):
        path = run_dir(pilot_workdir, seed, method) / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics[label] = {
            key: float(payload[key])
            for key in ("AUC", "NLLH", "overall_Brier", "DP", "U", "U_TILDE")
        }
    deltas = {
        key: metrics["primary"][key] - metrics["baseline"][key]
        for key in metrics["baseline"]
    }
    return {
        **metrics,
        "delta": deltas,
        "collapse": deltas["AUC"] < collapse_delta_min,
    }


def build_report(protocol: dict, workdir: str | Path, pilot_workdir: str | Path):
    """Pair confirmatory audits and apply the frozen replication labels."""

    seeds = [
        int(seed) for seed in protocol["seed_roles"]["internal_confirmatory"]
    ]
    baseline_method = protocol["methods"]["baseline"]
    primary_method = protocol["methods"]["primary"]
    targets = list(protocol["audit_targets"]["nodes"])
    targets.extend(
        f"joint_{name}"
        for name in protocol["audit_targets"]["joint_paths"]
    )
    material = float(protocol["material_delta"])
    seed_results = {}
    audit_commit = None
    protocol_hash = None
    for seed in seeds:
        baseline = _load_audit(workdir, seed, baseline_method)
        primary = _load_audit(workdir, seed, primary_method)
        for key in (
            "audit_git_commit",
            "stage2_1_protocol_sha256",
            "row_sample_sha256",
            "user_sample_sha256",
            "probe_seed",
        ):
            if baseline.get(key) != primary.get(key):
                raise ValueError(f"Seed {seed} paired audits differ on {key}.")
        if audit_commit is None:
            audit_commit = primary["audit_git_commit"]
            protocol_hash = primary["stage2_1_protocol_sha256"]
        elif (
            primary["audit_git_commit"] != audit_commit
            or primary["stage2_1_protocol_sha256"] != protocol_hash
        ):
            raise ValueError("Confirmatory seeds use different audit code/protocol.")

        contrasts = {
            protocol_name: {
                target: _target_contrast(
                    baseline,
                    primary,
                    target,
                    protocol_name,
                )
                for target in targets
            }
            for protocol_name in ("row_disjoint", "user_disjoint")
        }
        row_deltas = {
            name: item["max_family_delta"]
            for name, item in contrasts["row_disjoint"].items()
        }
        user_deltas = {
            name: item["max_family_delta"]
            for name, item in contrasts["user_disjoint"].items()
        }
        seed_results[str(seed)] = {
            "targets": contrasts,
            "redistribution": protocol_consistency(
                row_deltas,
                user_deltas,
                material_delta=material,
            ),
            "conditional_gain_contrasts": _gain_contrasts(
                baseline,
                primary,
                "conditional_gains",
            ),
            "joint_gain_contrasts": _gain_contrasts(
                baseline,
                primary,
                "joint_gains",
            ),
            "outcome": _outcome(
                pilot_workdir,
                seed,
                baseline_method,
                primary_method,
                collapse_delta_min=float(
                    protocol["confirmatory_gate"]["auc_delta_collapse_min"]
                ),
            ),
        }

    local_targets = protocol["audit_targets"]["local_reduction_targets"]
    amplification_targets = protocol["audit_targets"]["amplification_targets"]
    joint_targets = [
        f"joint_{name}"
        for name in protocol["audit_targets"]["joint_paths"]
    ]
    common_local_negative = [
        target
        for target in local_targets
        if all(
            seed_results[str(seed)]["targets"]["row_disjoint"][target][
                "max_family_delta"
            ]
            < 0
            for seed in seeds
        )
    ]
    common_material_amplified = [
        target
        for target in amplification_targets
        if all(
            seed_results[str(seed)]["targets"]["row_disjoint"][target][
                "max_family_delta"
            ]
            >= material
            for seed in seeds
        )
    ]
    common_positive_amplified = [
        target
        for target in amplification_targets
        if all(
            seed_results[str(seed)]["targets"]["row_disjoint"][target][
                "max_family_delta"
            ]
            > 0
            for seed in seeds
        )
    ]
    common_user_material_joint = [
        target
        for target in joint_targets
        if all(
            seed_results[str(seed)]["targets"]["user_disjoint"][target][
                "max_family_delta"
            ]
            >= material
            for seed in seeds
        )
    ]
    common_user_positive_joint = [
        target
        for target in joint_targets
        if all(
            seed_results[str(seed)]["targets"]["user_disjoint"][target][
                "max_family_delta"
            ]
            > 0
            for seed in seeds
        )
    ]
    no_collapse = all(
        not seed_results[str(seed)]["outcome"]["collapse"] for seed in seeds
    )
    gate = protocol["confirmatory_gate"]
    material_pass = (
        len(common_local_negative)
        >= int(gate["minimum_common_local_reductions"])
        and len(common_material_amplified)
        >= int(gate["minimum_common_material_amplifications"])
        and len(common_user_material_joint)
        >= int(gate["minimum_common_user_joint_amplifications"])
        and no_collapse
    )
    directional_pass = (
        len(common_local_negative)
        >= int(gate["minimum_common_local_reductions"])
        and len(common_positive_amplified)
        >= int(gate["minimum_common_material_amplifications"])
        and len(common_user_positive_joint)
        >= int(gate["minimum_common_user_joint_amplifications"])
        and no_collapse
    )
    if material_pass:
        decision = "replicated"
    elif directional_pass:
        decision = "partially_replicated"
    else:
        decision = "not_replicated"

    return {
        "version": 1,
        "stage": "S21-M3",
        "decision": decision,
        "stage2_1_code_commit": git_commit(PROJECT_ROOT),
        "audit_git_commit": audit_commit,
        "stage2_1_protocol_sha256": protocol_hash,
        "seeds": seed_results,
        "gate": {
            "material_delta": material,
            "common_local_negative": common_local_negative,
            "common_material_amplified": common_material_amplified,
            "common_positive_amplified": common_positive_amplified,
            "common_user_material_joint": common_user_material_joint,
            "common_user_positive_joint": common_user_positive_joint,
            "no_prediction_collapse": no_collapse,
            "material_pass": material_pass,
            "directional_pass": directional_pass,
        },
        "next_stage": (
            "S21-M4_component_attribution"
            if decision in ("replicated", "partially_replicated")
            else "S21-M6_internal_negative_result_closure"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }


def render_markdown(report: dict) -> str:
    """Render a compact decision report without hiding protocol disagreement."""

    lines = [
        "# Stage2.1 M3 Cross-Seed Replication Report",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Next stage: `{report['next_stage']}`",
        f"- Audit commit: `{report['audit_git_commit']}`",
        "",
        "## Confirmatory Gate",
        "",
        f"- Common local negative targets: "
        f"`{report['gate']['common_local_negative']}`",
        f"- Common material amplified targets: "
        f"`{report['gate']['common_material_amplified']}`",
        f"- Common raw-user material joint targets: "
        f"`{report['gate']['common_user_material_joint']}`",
        f"- No prediction collapse: "
        f"`{report['gate']['no_prediction_collapse']}`",
        "",
        "## Per-Seed Redistribution",
        "",
        "| Seed | Row reduction/migration | User reduction/migration | "
        "Row RI | User RI | AUC delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, item in report["seeds"].items():
        row = item["redistribution"]["row_disjoint"]
        user = item["redistribution"]["user_disjoint"]
        lines.append(
            f"| {seed} | {row['reduction_count']}/{row['migration_count']} | "
            f"{user['reduction_count']}/{user['migration_count']} | "
            f"{row['redistribution_index']:.6f} | "
            f"{user['redistribution_index']:.6f} | "
            f"{item['outcome']['delta']['AUC']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Target Max-Family Deltas",
            "",
            "| Seed | Protocol | Target | Delta | State |",
            "|---:|---|---|---:|---|",
        ]
    )
    for seed, item in report["seeds"].items():
        for protocol_name, targets in item["targets"].items():
            states = item["redistribution"][protocol_name]["states"]
            for target, contrast in targets.items():
                lines.append(
                    f"| {seed} | {protocol_name} | {target} | "
                    f"{contrast['max_family_delta']:+.6f} | "
                    f"{states[target]} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This is internal confirmation across training seeds on the same "
            "FairJob evaluation population. It is not independent dataset "
            "confirmation, mutual-information estimation, or a causal fairness "
            "result.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict:
    protocol = load_stage2_1_protocol(args.protocol)
    workdir = args.workdir or protocol["paths"]["stage2_1_workdir"]
    pilot_workdir = (
        args.pilot_workdir or protocol["paths"]["pilot_workdir"]
    )
    report = build_report(protocol, workdir, pilot_workdir)
    write_run_manifest(args.out_json, report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
