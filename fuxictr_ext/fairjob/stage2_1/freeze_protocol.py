"""Freeze Stage2-A provenance and the Stage2.1 checkpoint-audit protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.representation_io import sha256_file
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest
from fuxictr_ext.fairjob.run_stage2_pilot import (
    DATASET_ID,
    expid,
    load_matrix,
    run_dir,
)
from fuxictr_ext.fairjob.stage2_1.checkpoint_audit import (
    load_stage2_1_protocol,
)


DEFAULT_MATRIX = Path("configs/fairjob/stage2_ablation_matrix.yaml")
DEFAULT_PROTOCOL = Path("configs/fairjob/stage2_1_audit_protocol.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--pilot_workdir", default=None)
    parser.add_argument("--row_sample", default=None)
    parser.add_argument("--user_sample", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def audit_p2_runs(
    pilot_workdir: str | Path,
    matrix: dict,
    expected_training_commit: str,
) -> dict:
    """Validate all frozen P2 runs and hash compact provenance artifacts."""

    runs = {}
    for seed in [int(value) for value in matrix["seeds"]]:
        runs[str(seed)] = {}
        for method in matrix["methods"]:
            source = run_dir(pilot_workdir, seed, method)
            manifest_path = source / "run_manifest.json"
            hparams_path = source / "hparams.json"
            metrics_path = source / "metrics.json"
            checkpoint = (
                source
                / "checkpoints"
                / DATASET_ID
                / f"{expid(method)}.model"
            )
            prediction = (
                source / "predictions" / f"{expid(method)}.csv"
            )
            required = (
                manifest_path,
                hparams_path,
                metrics_path,
                checkpoint,
                prediction,
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"P2 run seed={seed} method={method} is incomplete: {missing}"
                )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if manifest.get("status") != "complete":
                raise ValueError(f"P2 run is not complete: {source}")
            if manifest.get("git_commit") != expected_training_commit:
                raise ValueError(f"P2 run has a stale training commit: {source}")
            hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
            teacher = hparams.get("hparams", {}).get(
                "stage2_backbone_checkpoint"
            )
            if method == "baseline" and teacher:
                raise ValueError("Baseline unexpectedly declares a teacher.")
            if method != "baseline":
                expected_teacher = (
                    run_dir(pilot_workdir, seed, "baseline")
                    / "checkpoints"
                    / DATASET_ID
                    / f"{expid('baseline')}.model"
                )
                if Path(teacher or "") != expected_teacher:
                    raise ValueError(
                        f"Run {source} does not use its same-seed baseline."
                    )
            runs[str(seed)][method] = {
                "run_dir": str(source),
                "manifest_sha256": sha256_file(manifest_path),
                "hparams_sha256": sha256_file(hparams_path),
                "metrics_sha256": sha256_file(metrics_path),
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "prediction": str(prediction),
                "prediction_bytes": prediction.stat().st_size,
                "status": "complete",
            }
    return runs


def execute(args: argparse.Namespace) -> dict:
    protocol = load_stage2_1_protocol(args.protocol)
    matrix = load_matrix(args.matrix)
    paths = protocol["paths"]
    pilot_workdir = Path(args.pilot_workdir or paths["pilot_workdir"])
    row_sample = Path(args.row_sample or paths["row_disjoint_sample"])
    user_sample = Path(args.user_sample or paths["user_disjoint_sample"])
    source_protocol = (
        PROJECT_ROOT
        / "docs/fairjob/stage2_1/"
        "Stage2.1_FairJob_Representation_Graph_Proxy_Decodability_"
        "Redistribution_Audit.md"
    )
    expected_source_hash = protocol["frozen_provenance"][
        "stage2_1_source_protocol_sha256"
    ]
    if sha256_file(source_protocol) != expected_source_hash:
        raise ValueError("Archived Stage2.1 source protocol hash changed.")

    training_commit = protocol["frozen_provenance"][
        "stage2_training_commit"
    ]
    payload = {
        "version": 1,
        "stage": "S21-M0",
        "status": "frozen",
        "stage2_training_commit": training_commit,
        "stage2_decision_commit": protocol["frozen_provenance"][
            "stage2_decision_commit"
        ],
        "stage2_1_code_commit": git_commit(PROJECT_ROOT),
        "stage2_1_protocol": str(Path(args.protocol)),
        "stage2_1_protocol_sha256": sha256_file(args.protocol),
        "source_protocol": str(source_protocol),
        "source_protocol_sha256": expected_source_hash,
        "matrix": str(Path(args.matrix)),
        "matrix_sha256": sha256_file(args.matrix),
        "row_sample": str(row_sample),
        "row_sample_sha256": sha256_file(row_sample),
        "user_sample": str(user_sample),
        "user_sample_sha256": sha256_file(user_sample),
        "graph_sha256": sha256_file(paths["graph"]),
        "probe_protocol_sha256": sha256_file(paths["probe_protocol"]),
        "seed_roles": protocol["seed_roles"],
        "material_delta": protocol["material_delta"],
        "storage": protocol["storage"],
        "claim_boundary": protocol["claim_boundary"],
        "p2_runs": audit_p2_runs(
            pilot_workdir,
            matrix,
            expected_training_commit=training_commit,
        ),
    }
    payload["p2_run_count"] = sum(
        len(methods) for methods in payload["p2_runs"].values()
    )
    expected_count = len(matrix["seeds"]) * len(matrix["methods"])
    if payload["p2_run_count"] != expected_count:
        raise ValueError("Frozen P2 run count is incomplete.")
    write_run_manifest(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
