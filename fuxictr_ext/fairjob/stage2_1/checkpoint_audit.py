"""Audit one P2 checkpoint under the frozen Stage2.1 measurement protocol."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.representation_graph import (
    graph_hash,
    load_representation_graph,
)
from fuxictr_ext.fairjob.representation_io import sha256_file
from fuxictr_ext.fairjob.run_manifest import (
    git_commit,
    runtime_versions,
    stable_hash,
    write_run_manifest,
)
from fuxictr_ext.fairjob.stage2_checkpoint_probe import (
    PROTOCOL_NAMES,
    SPLITS,
    build_target_plan,
    collect_split_representations,
    load_model_and_data,
    load_probe_protocol,
    load_protocol_rows,
    protocol_view,
    required_representation_names,
    target_matrix,
    validate_reference_prediction,
)
from fuxictr_ext.fairjob.stage2_1.probe_audit import (
    conditional_gain,
    fit_probe_suite,
    joint_gain,
)


DEFAULT_PROTOCOL = Path("configs/fairjob/stage2_1_audit_protocol.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--expid", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--row_sample", default=None)
    parser.add_argument("--user_sample", default=None)
    parser.add_argument("--reference_prediction", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--probe_seed", type=int, default=2019)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=PROTOCOL_NAMES,
        default=None,
    )
    parser.add_argument("--skip_conditional", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_stage2_1_protocol(path: str | Path) -> dict:
    """Load and validate the frozen Stage2.1 protocol."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Stage2.1 protocol version.")
    if payload.get("status") != "frozen_before_confirmatory_probe_read":
        raise ValueError("Stage2.1 protocol is not frozen.")
    if payload.get("storage", {}).get("persist_representations") is not False:
        raise ValueError("Stage2.1 must not persist reconstructed representations.")
    return payload


def frozen_target_names(protocol: dict) -> list[str]:
    """Return node and joint targets in frozen protocol order."""

    targets = list(protocol["audit_targets"]["nodes"])
    targets.extend(
        f"joint_{name}"
        for name in protocol["audit_targets"]["joint_paths"]
    )
    if len(targets) != len(set(targets)):
        raise ValueError("Stage2.1 audit targets contain duplicates.")
    return targets


def select_targets(
    frozen_targets: list[str], requested: list[str] | None
) -> list[str]:
    """Select a smoke subset without changing frozen target order."""

    if requested is None:
        return list(frozen_targets)
    unknown = sorted(set(requested).difference(frozen_targets))
    if unknown:
        raise ValueError(f"Unknown Stage2.1 targets: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("Stage2.1 targets cannot contain duplicates.")
    return [name for name in frozen_targets if name in requested]


def max_family_endpoint(suite: dict) -> dict:
    """Return the conservative maximum held-out AUC across frozen families."""

    candidates = [
        {
            "family": family,
            "test_auc": float(result["auc_selected"]["test_auc"]),
            "valid_auc": float(result["auc_selected"]["valid_auc"]),
            "params": result["auc_selected"]["params"],
        }
        for family, result in suite["families"].items()
    ]
    selected = max(
        candidates,
        key=lambda item: (
            item["test_auc"],
            item["family"],
        ),
    )
    return {
        "selected": selected,
        "family_results": candidates,
        "selection_scope": "maximum_after_frozen_family_specific_selection",
    }


def _members_key(members: list[str]) -> str:
    return "concat__" + "__".join(members)


def _plan_target_for_members(plan: dict, members: list[str]) -> str | None:
    for name, spec in plan.items():
        if spec["members"] == members and not spec.get("alias_of"):
            return name
    return None


def _load_resume_payload(path: str | Path, identity: dict) -> dict:
    path = Path(path)
    if not path.exists():
        return {
            "version": 1,
            "stage": "S21-M1",
            "status": "running",
            **identity,
            "targets": {},
            "derived_suites": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"Cannot resume Stage2.1 audit with changed {key}.")
    payload["status"] = "running"
    payload.setdefault("targets", {})
    payload.setdefault("derived_suites", {})
    return payload


def execute(args: argparse.Namespace) -> dict:
    """Run one complete or bounded Stage2.1 checkpoint audit."""

    os.chdir(PROJECT_ROOT)
    stage2_1_protocol = load_stage2_1_protocol(args.protocol)
    graph_path = stage2_1_protocol["paths"]["graph"]
    probe_protocol_path = stage2_1_protocol["paths"]["probe_protocol"]
    graph = load_representation_graph(graph_path)
    probe_protocol = load_probe_protocol(probe_protocol_path)
    plan = build_target_plan(graph, probe_protocol)
    frozen_targets = frozen_target_names(stage2_1_protocol)
    unknown = sorted(set(frozen_targets).difference(plan))
    if unknown:
        raise ValueError(f"Stage2.1 protocol references unknown targets: {unknown}")
    selected_targets = select_targets(frozen_targets, args.targets)
    selected_protocols = list(args.protocols or PROTOCOL_NAMES)

    run_dir = Path(args.run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    hparams_path = run_dir / "hparams.json"
    if not run_manifest_path.exists() or not hparams_path.exists():
        raise FileNotFoundError(f"P2 run metadata is incomplete: {run_dir}")
    training_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if training_manifest.get("status") != "complete":
        raise ValueError(f"P2 run is not complete: {run_dir}")
    expected_training_commit = stage2_1_protocol["frozen_provenance"][
        "stage2_training_commit"
    ]
    if training_manifest.get("git_commit") != expected_training_commit:
        raise ValueError("P2 run does not match the frozen training commit.")
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))

    row_sample = Path(
        args.row_sample or stage2_1_protocol["paths"]["row_disjoint_sample"]
    )
    user_sample = Path(
        args.user_sample or stage2_1_protocol["paths"]["user_disjoint_sample"]
    )
    samples = load_protocol_rows(row_sample, user_sample)

    required_targets = list(selected_targets)
    if not args.skip_conditional:
        required_targets = list(frozen_targets)
    required_plan = {name: plan[name] for name in required_targets}
    required_names = required_representation_names(required_plan)
    if not args.skip_conditional:
        for spec in stage2_1_protocol["conditional_pairs"].values():
            for member in spec["parent"] + spec["child"]:
                if member not in required_names:
                    required_names.append(member)
        for spec in stage2_1_protocol["joint_gain"].values():
            for member in spec["left"] + spec["right"] + spec["joint"]:
                if member not in required_names:
                    required_names.append(member)

    model, params, checkpoint, generators = load_model_and_data(args, hparams)
    reference_prediction = Path(
        args.reference_prediction
        or run_dir / "predictions" / f"{args.expid}.csv"
    )
    identity = {
        "audit_git_commit": git_commit(PROJECT_ROOT),
        "training_git_commit": training_manifest["git_commit"],
        "stage2_decision_commit": stage2_1_protocol["frozen_provenance"][
            "stage2_decision_commit"
        ],
        "expid": args.expid,
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "probe_seed": args.probe_seed,
        "graph_hash": graph_hash(graph),
        "stage2_probe_protocol_hash": stable_hash(probe_protocol),
        "stage2_1_protocol_sha256": sha256_file(args.protocol),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "reference_prediction": str(reference_prediction),
        "reference_prediction_sha256": sha256_file(reference_prediction),
        "row_sample": str(row_sample),
        "row_sample_sha256": sha256_file(row_sample),
        "user_sample": str(user_sample),
        "user_sample_sha256": sha256_file(user_sample),
        "selected_targets": selected_targets,
        "selected_protocols": selected_protocols,
        "conditional_enabled": not args.skip_conditional,
        "storage_mode": "in_memory_checkpoint_reconstructable",
    }
    payload = _load_resume_payload(args.out, identity)
    payload["runtime"] = runtime_versions()
    payload["claim_boundary"] = stage2_1_protocol["claim_boundary"]
    write_run_manifest(args.out, payload)

    collected = {}
    for split in SPLITS:
        print(f"[S21] reconstruct split={split}", flush=True)
        collected[split] = collect_split_representations(
            model=model,
            data_generator=generators[split],
            meta_path=params["fairjob_split_meta_paths"][split],
            selected_row_ids=samples["row_disjoint"][split],
            required_names=required_names,
        )
    payload["prediction_integrity"] = validate_reference_prediction(
        collected["test"], reference_prediction
    )
    payload["source_rows"] = {
        split: collected[split]["source_rows"] for split in SPLITS
    }
    payload["representation_shapes"] = collected["train"][
        "representation_shapes"
    ]
    payload["protocol_rows"] = {
        protocol_name: {
            split: int(len(samples[protocol_name][split])) for split in SPLITS
        }
        for protocol_name in selected_protocols
    }
    views = {
        protocol_name: {
            split: protocol_view(
                collected[split], samples[protocol_name][split]
            )
            for split in SPLITS
        }
        for protocol_name in selected_protocols
    }
    fixed_capacity = stage2_1_protocol["fixed_capacity"]
    fixed_capacity = {
        key: value
        for key, value in fixed_capacity.items()
        if key != "rule"
    }
    epsilon = float(
        stage2_1_protocol["probe_endpoint"]["probability_clip_epsilon"]
    )
    write_run_manifest(args.out, payload)

    def fit_members(members: list[str], protocol_name: str) -> dict:
        split_views = views[protocol_name]
        matrices = {
            split: target_matrix(split_views[split], members)
            for split in SPLITS
        }
        labels = {
            split: split_views[split]["protected_attribute"]
            for split in SPLITS
        }
        suite = fit_probe_suite(
            matrices["train"],
            labels["train"],
            matrices["valid"],
            labels["valid"],
            matrices["test"],
            labels["test"],
            probe_protocol,
            fixed_capacity=fixed_capacity,
            seed=args.probe_seed,
            epsilon=epsilon,
        )
        suite["rows"] = {
            split: int(len(labels[split])) for split in SPLITS
        }
        suite["dimensions"] = int(matrices["train"].shape[1])
        suite["max_family_endpoint"] = max_family_endpoint(suite)
        del matrices
        gc.collect()
        return suite

    for target_name in selected_targets:
        spec = plan[target_name]
        if target_name not in payload["targets"]:
            payload["targets"][target_name] = {
                "kind": spec["kind"],
                "members": spec["members"],
                "alias_of": spec["alias_of"],
                "protocols": {},
            }
        target_result = payload["targets"][target_name]
        if spec["alias_of"]:
            canonical = spec["alias_of"]
            if canonical not in payload["targets"]:
                raise ValueError(
                    f"Alias {target_name} precedes canonical target {canonical}."
                )
            for protocol_name in selected_protocols:
                if protocol_name not in target_result["protocols"]:
                    target_result["protocols"][protocol_name] = copy.deepcopy(
                        payload["targets"][canonical]["protocols"][protocol_name]
                    )
            write_run_manifest(args.out, payload)
            continue
        for protocol_name in selected_protocols:
            if protocol_name in target_result["protocols"]:
                print(
                    f"[S21] skip target={target_name} protocol={protocol_name}",
                    flush=True,
                )
                continue
            print(
                f"[S21] fit target={target_name} protocol={protocol_name}",
                flush=True,
            )
            target_result["protocols"][protocol_name] = fit_members(
                spec["members"], protocol_name
            )
            write_run_manifest(args.out, payload)

    if not args.skip_conditional:
        for target_name in frozen_targets:
            if target_name not in payload["targets"]:
                raise ValueError(
                    f"Conditional audit requires frozen target {target_name}."
                )

        def ensure_suite(members: list[str], protocol_name: str) -> dict:
            target_name = _plan_target_for_members(plan, members)
            if (
                target_name is not None
                and target_name in payload["targets"]
                and protocol_name
                in payload["targets"][target_name]["protocols"]
            ):
                return payload["targets"][target_name]["protocols"][
                    protocol_name
                ]
            key = _members_key(members)
            derived = payload["derived_suites"].setdefault(
                key, {"members": members, "protocols": {}}
            )
            if protocol_name not in derived["protocols"]:
                print(
                    f"[S21] fit derived={key} protocol={protocol_name}",
                    flush=True,
                )
                derived["protocols"][protocol_name] = fit_members(
                    members, protocol_name
                )
                write_run_manifest(args.out, payload)
            return derived["protocols"][protocol_name]

        payload.setdefault("conditional_gains", {})
        for pair_name, spec in stage2_1_protocol["conditional_pairs"].items():
            pair_result = payload["conditional_gains"].setdefault(
                pair_name,
                {
                    "parent": spec["parent"],
                    "child": spec["child"],
                    "protocols": {},
                },
            )
            for protocol_name in selected_protocols:
                if protocol_name in pair_result["protocols"]:
                    continue
                parent_suite = ensure_suite(spec["parent"], protocol_name)
                combined_suite = ensure_suite(
                    spec["parent"] + spec["child"], protocol_name
                )
                pair_result["protocols"][protocol_name] = conditional_gain(
                    parent_suite, combined_suite
                )
                write_run_manifest(args.out, payload)

        payload.setdefault("joint_gains", {})
        for gain_name, spec in stage2_1_protocol["joint_gain"].items():
            gain_result = payload["joint_gains"].setdefault(
                gain_name,
                {
                    "left": spec["left"],
                    "right": spec["right"],
                    "joint": spec["joint"],
                    "protocols": {},
                },
            )
            for protocol_name in selected_protocols:
                if protocol_name in gain_result["protocols"]:
                    continue
                left_suite = ensure_suite(spec["left"], protocol_name)
                right_suite = ensure_suite(spec["right"], protocol_name)
                joint_suite = ensure_suite(spec["joint"], protocol_name)
                gain_result["protocols"][protocol_name] = joint_gain(
                    left_suite, right_suite, joint_suite
                )
                write_run_manifest(args.out, payload)

    payload["status"] = "complete"
    payload["completed_target_count"] = len(payload["targets"])
    payload["completed_derived_suite_count"] = len(payload["derived_suites"])
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
