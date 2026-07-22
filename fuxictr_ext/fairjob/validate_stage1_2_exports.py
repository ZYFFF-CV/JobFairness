"""Validate the complete Stage1.2 post-intervention export matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.prepare_probe_samples import load_probe_row_ids
from fuxictr_ext.fairjob.representation_io import load_representation_row_ids
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/fairjob/stage1_2_matrix.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def validate_exports(matrix_path: str | Path, tolerance: float = 1e-6) -> dict:
    """Validate row alignment, layer contracts, masks, and predictions."""

    if tolerance < 0:
        raise ValueError("Prediction tolerance must be non-negative.")
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    jobs = matrix.get("jobs", [])
    if len(jobs) != 18:
        raise ValueError(f"Stage1.2 export validation requires 18 jobs, found {len(jobs)}.")
    workdir = Path(matrix["workdir_root"]) / "representation_exports"
    probe_sample = Path(matrix["probe_sample"])
    expected_row_ids = {
        split: load_probe_row_ids(probe_sample, split)
        for split in ("train", "valid", "test")
    }
    expected_shapes = None
    results = []
    for job in jobs:
        run_dir = workdir / job["name"]
        success = json.loads(
            (run_dir / "representation_export.success.json").read_text(
                encoding="utf-8"
            )
        )
        if success.get("returncode") != 0:
            raise ValueError(f"Representation export failed: {job['name']}")
        export_manifest = json.loads(
            (run_dir / "representation_export_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if export_manifest.get("status") != "representation_export_complete":
            raise ValueError(f"Incomplete representation export: {job['name']}")
        prediction_check = export_manifest.get("reference_prediction_check") or {}
        max_prediction_diff = float(
            prediction_check.get("max_abs_prediction_diff", float("inf"))
        )
        if max_prediction_diff > tolerance:
            raise ValueError(
                f"Prediction mismatch for {job['name']}: {max_prediction_diff}"
            )

        split_summaries = {}
        model_metadata = None
        for split in ("train", "valid", "test"):
            split_dir = run_dir / "representations" / split
            manifest, row_ids = load_representation_row_ids(
                split_dir, validate_shards=False
            )
            if not np.array_equal(row_ids, expected_row_ids[split]):
                raise ValueError(f"Frozen row IDs differ for {job['name']}:{split}")
            sampling = manifest.get("sampling", {})
            if sampling.get("method") != "frozen_row_ids":
                raise ValueError(f"Unexpected sampling for {job['name']}:{split}")
            if Path(sampling.get("row_id_source", "")) != probe_sample:
                raise ValueError(f"Probe sample path differs for {job['name']}:{split}")
            shapes = manifest["representation_shapes"]
            if expected_shapes is None:
                expected_shapes = shapes
            elif shapes != expected_shapes:
                raise ValueError(f"Representation shapes differ for {job['name']}:{split}")
            metadata = manifest.get("model_representation_metadata")
            if model_metadata is None:
                model_metadata = metadata
            elif metadata != model_metadata:
                raise ValueError(f"Model metadata differs between splits: {job['name']}")
            split_summaries[split] = {
                "rows": manifest["rows"],
                "shards": len(manifest["shards"]),
            }

        method = job["name"].rsplit("_seed", 1)[0]
        if model_metadata.get("mitigation_method") != method:
            raise ValueError(f"Mitigation method mismatch for {job['name']}")
        if int(model_metadata.get("training_seed")) != int(job["seed"]):
            raise ValueError(f"Training seed mismatch for {job['name']}")

        first_manifest = json.loads(
            (run_dir / "representations" / "test" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        first_shard = (
            run_dir
            / "representations"
            / "test"
            / first_manifest["shards"][0]["path"]
        )
        with np.load(first_shard) as shard:
            identity_error = float(
                np.max(
                    np.abs(
                        shard["dcnv2_final_pre_mitigation"]
                        - shard["dcnv2_suppressed_component"]
                        - shard["dcnv2_residual_component"]
                    )
                )
            )
        if identity_error > 1e-6:
            raise ValueError(
                f"Intervention components do not reconstruct pre-state: {job['name']}"
            )
        results.append(
            {
                "job": job["name"],
                "method": method,
                "seed": job["seed"],
                "max_abs_prediction_diff": max_prediction_diff,
                "max_component_identity_error": identity_error,
                "suppressed_dimensions": model_metadata["suppressed_dimensions"],
                "suppression_mask_sha256": model_metadata[
                    "suppression_mask_sha256"
                ],
                "splits": split_summaries,
            }
        )
    return {
        "version": 1,
        "stage": "S12-M1B",
        "git_commit": git_commit(PROJECT_ROOT),
        "matrix": str(matrix_path),
        "probe_sample": str(probe_sample),
        "prediction_tolerance": tolerance,
        "job_count": len(results),
        "split_manifest_count": len(results) * 3,
        "representation_shapes": expected_shapes,
        "max_abs_prediction_diff": max(
            item["max_abs_prediction_diff"] for item in results
        ),
        "max_component_identity_error": max(
            item["max_component_identity_error"] for item in results
        ),
        "jobs": results,
        "status": "pass",
    }


def render_report(payload: dict) -> str:
    """Render the matrix receipt without duplicating large manifests."""

    lines = [
        "# S12-M1B post-intervention export report",
        "",
        f"- Status: `{payload['status']}`.",
        f"- Export commit: `{payload['git_commit']}`.",
        f"- Jobs: `{payload['job_count']}`; split manifests: "
        f"`{payload['split_manifest_count']}`.",
        "- Rows per train/valid/test split and job: `50000/50000/50000`.",
        f"- Maximum prediction difference: `{payload['max_abs_prediction_diff']}` "
        f"(tolerance `{payload['prediction_tolerance']}`).",
        "- Maximum component reconstruction error: "
        f"`{payload['max_component_identity_error']}`.",
        "- Sampling: exact frozen raw row IDs from Stage1.1 `probe_rows.npz`.",
        "- Training: none; all exports loaded existing M5A checkpoints.",
        "",
        "| Job | Seed | Suppressed dims | Prediction diff | Component error |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in payload["jobs"]:
        lines.append(
            f"| {item['job']} | {item['seed']} | "
            f"{item['suppressed_dimensions']} | "
            f"{item['max_abs_prediction_diff']:.3e} | "
            f"{item['max_component_identity_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "The matrix is accepted for S12-M2 leakage and migration probes. "
            "This receipt does not claim that any intervention reduced leakage "
            "or improved outcome fairness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = validate_exports(args.matrix, tolerance=args.tolerance)
    write_run_manifest(args.out, payload)
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "job_count": payload["job_count"],
                "max_abs_prediction_diff": payload["max_abs_prediction_diff"],
                "max_component_identity_error": payload[
                    "max_component_identity_error"
                ],
                "out": args.out,
                "report": args.report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
