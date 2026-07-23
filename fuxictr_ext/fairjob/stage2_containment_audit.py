"""Apply the frozen containment stopping rule to paired P3 probe results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


PROTOCOLS = ("row_disjoint", "user_disjoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--primary_hparams", required=True)
    parser.add_argument("--material_delta", type=float, default=0.005)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _maximum_probe(target: dict, protocol: str) -> dict:
    """Return the strongest validation-selected held-out probe observation."""

    try:
        probes = target["protocols"][protocol]["probes"]
    except KeyError as error:
        raise ValueError(f"P3 target is missing protocol {protocol}.") from error
    if not probes:
        raise ValueError(f"P3 protocol {protocol} has no probe results.")
    selected = max(
        probes,
        key=lambda item: (
            float(item["test_auc"]),
            str(item["family"]),
        ),
    )
    return {
        "family": selected["family"],
        "test_auc": float(selected["test_auc"]),
        "valid_auc": float(selected["valid_auc"]),
        "params": selected["params"],
    }


def _validate_pair(baseline: dict, primary: dict) -> None:
    """Require the two P3 files to share every frozen audit input."""

    for label, payload in (("baseline", baseline), ("primary", primary)):
        if payload.get("status") != "complete":
            raise ValueError(f"{label} P3 result is not complete.")
        if payload.get("stage") != "S2-P3":
            raise ValueError(f"{label} is not a Stage2 P3 result.")
        prediction_check = payload.get("prediction_integrity", {})
        if float(prediction_check.get("max_abs_prediction_diff", 1.0)) > float(
            prediction_check.get("tolerance", 0.0)
        ):
            raise ValueError(f"{label} checkpoint does not reproduce P2 predictions.")
    immutable = (
        "audit_git_commit",
        "seed",
        "probe_seed",
        "graph_hash",
        "protocol_hash",
        "row_sample_sha256",
        "user_sample_sha256",
    )
    for key in immutable:
        if baseline.get(key) != primary.get(key):
            raise ValueError(f"Paired P3 results differ on {key}.")


def evaluate_early_stop(
    baseline: dict,
    primary: dict,
    target_nodes: list[str],
    joint_targets: list[str],
    material_delta: float = 0.005,
) -> dict:
    """Return a deterministic early-fail decision for one paired seed.

    The final gate requires every target-node delta to be negative in every
    seed. Therefore one nonnegative target delta is irreversible even before
    the other seeds run. A joint-path increase at or above the frozen material
    threshold is independently irreversible.
    """

    if material_delta <= 0:
        raise ValueError("material_delta must be positive.")
    if not target_nodes:
        raise ValueError("At least one trained target node is required.")
    _validate_pair(baseline, primary)

    protocol_results = {}
    irreversible_reasons = []
    for protocol in PROTOCOLS:
        node_checks = {}
        for name in target_nodes:
            if name not in baseline["targets"] or name not in primary["targets"]:
                raise ValueError(f"Paired P3 results are missing target node {name}.")
            baseline_max = _maximum_probe(baseline["targets"][name], protocol)
            primary_max = _maximum_probe(primary["targets"][name], protocol)
            delta = primary_max["test_auc"] - baseline_max["test_auc"]
            node_checks[name] = {
                "baseline_max_probe": baseline_max,
                "primary_max_probe": primary_max,
                "delta": delta,
                "negative_direction": delta < 0,
                "material_increase": delta >= material_delta,
            }
            if delta >= 0:
                irreversible_reasons.append(
                    f"{protocol}:target_nonnegative:{name}"
                )

        joint_checks = {}
        for name in joint_targets:
            target_name = f"joint_{name}"
            if (
                target_name not in baseline["targets"]
                or target_name not in primary["targets"]
            ):
                raise ValueError(
                    f"Paired P3 results are missing joint target {target_name}."
                )
            baseline_max = _maximum_probe(
                baseline["targets"][target_name], protocol
            )
            primary_max = _maximum_probe(
                primary["targets"][target_name], protocol
            )
            delta = primary_max["test_auc"] - baseline_max["test_auc"]
            joint_checks[name] = {
                "baseline_max_probe": baseline_max,
                "primary_max_probe": primary_max,
                "delta": delta,
                "material_increase": delta >= material_delta,
            }
            if delta >= material_delta:
                irreversible_reasons.append(
                    f"{protocol}:joint_material_increase:{name}"
                )
        protocol_results[protocol] = {
            "target_nodes": node_checks,
            "joint_targets": joint_checks,
        }

    decision = "early_fail" if irreversible_reasons else "continue"
    return {
        "version": 1,
        "stage": "S2-P3-early-stop",
        "decision": decision,
        "remaining_checkpoint_probes": (
            "not_approved" if decision == "early_fail" else "approved"
        ),
        "seed": int(primary["seed"]),
        "material_delta": material_delta,
        "target_nodes": target_nodes,
        "joint_targets": joint_targets,
        "protocols": protocol_results,
        "irreversible_reasons": irreversible_reasons,
        "rule": {
            "target_seed_direction": "all_negative",
            "joint_no_seed_material_increase": True,
            "probe_summary": "maximum_held_out_auc_across_frozen_families",
        },
    }


def execute(args: argparse.Namespace) -> dict:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    primary = json.loads(Path(args.primary).read_text(encoding="utf-8"))
    hparams = json.loads(
        Path(args.primary_hparams).read_text(encoding="utf-8")
    )["hparams"]
    target_nodes = list(hparams.get("stage2_adversary_nodes", []))
    joint_targets = list(hparams.get("stage2_joint_paths", {}))
    payload = evaluate_early_stop(
        baseline,
        primary,
        target_nodes=target_nodes,
        joint_targets=joint_targets,
        material_delta=args.material_delta,
    )
    payload.update(
        {
            "audit_git_commit": git_commit(PROJECT_ROOT),
            "probe_audit_git_commit": primary["audit_git_commit"],
            "training_git_commit": primary["training_git_commit"],
            "baseline_result": args.baseline,
            "primary_result": args.primary,
            "primary_hparams": args.primary_hparams,
        }
    )
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
