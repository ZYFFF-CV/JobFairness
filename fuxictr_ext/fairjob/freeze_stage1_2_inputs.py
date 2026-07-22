"""Freeze Stage1.1 M5A artifacts and hashes for Stage1.2 attribution work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.representation_io import sha256_file
from fuxictr_ext.fairjob.run_manifest import git_commit, runtime_versions, write_run_manifest


REQUIRED_DOCS = (
    "docs/fairjob/stage1_1/execution_plan.md",
    "docs/fairjob/stage1_1/m3_single_seed_diagnostic_report.md",
    "docs/fairjob/stage1_1/m3_multiseed_diagnostic_report.md",
    "docs/fairjob/stage1_1/m4_outcome_screening_report.md",
    "docs/fairjob/stage1_1/m5a_three_seed_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/fairjob/stage1_2_matrix.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage1.2 input artifact: {path}")
    return path


def _assert_under(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact escapes declared root {root}: {path}") from error


def _csv_rows(path: Path) -> int:
    """Count CSV data rows without materializing a full prediction file."""

    last_byte = b""
    with path.open("rb") as stream:
        lines = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        lines += 1
    return max(0, lines - 1)


def _fingerprint(path: Path, include_rows: bool = False) -> dict:
    payload = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if include_rows:
        payload["rows"] = _csv_rows(path)
    return payload


def freeze_inputs(matrix_path: str | Path) -> dict:
    """Validate and fingerprint the complete existing M5A artifact matrix."""

    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    jobs = matrix.get("jobs", [])
    if len(jobs) != 18:
        raise ValueError(f"Stage1.2 requires 18 frozen M5A jobs, found {len(jobs)}.")
    checkpoint_root = Path(matrix["checkpoint_workdir_root"])
    training_root = checkpoint_root / "training"
    expected_prediction_rows = int(matrix.get("expected_test_rows", 214446))
    artifacts = []
    training_commits = set()
    runtimes = []
    for job in jobs:
        run_dir = training_root / job["name"]
        _assert_under(run_dir, checkpoint_root)
        run_manifest_path = _require_file(run_dir / "run_manifest.json")
        success_path = _require_file(run_dir / "job.success.json")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if run_manifest.get("status") != "complete" or success.get("returncode") != 0:
            raise ValueError(f"M5A run is not complete: {run_dir}")
        if run_manifest.get("seed") != job.get("seed"):
            raise ValueError(f"Seed mismatch for {job['name']}.")
        training_commits.add(run_manifest["git_commit"])
        runtimes.append(run_manifest.get("runtime"))

        checkpoint = _require_file(
            run_dir
            / "checkpoints"
            / job["dataset_id"]
            / f"{job['expid']}.model"
        )
        prediction = _require_file(run_dir / "predictions" / f"{job['expid']}.csv")
        prediction_info = _fingerprint(prediction, include_rows=True)
        if prediction_info["rows"] != expected_prediction_rows:
            raise ValueError(
                f"Frozen prediction rows for {job['name']} are {prediction_info['rows']}."
            )
        artifacts.append(
            {
                "job": job["name"],
                "method": job["name"].rsplit("_seed", 1)[0],
                "seed": job["seed"],
                "expid": job["expid"],
                "dataset_id": job["dataset_id"],
                "training_commit": run_manifest["git_commit"],
                "config_hash": run_manifest["config_hash"],
                "checkpoint": _fingerprint(checkpoint),
                "prediction": prediction_info,
                "hparams": _fingerprint(_require_file(run_dir / "hparams.json")),
                "metrics": _fingerprint(_require_file(run_dir / "metrics.json")),
                "run_manifest": _fingerprint(run_manifest_path),
                "success_marker": _fingerprint(success_path),
            }
        )
    if len(training_commits) != 1:
        raise ValueError(f"M5A runs use multiple commits: {sorted(training_commits)}")
    if any(runtime != runtimes[0] for runtime in runtimes[1:]):
        raise ValueError("M5A runtime manifests are inconsistent across jobs.")

    data_dir = Path(matrix["data_dir"])
    data_files = {
        name: _fingerprint(_require_file(data_dir / name))
        for name in ("feature_manifest.json", "split_manifest.json", "test_meta.csv")
    }
    probe_sample = Path(matrix["probe_sample"])
    probe_files = {
        "sample": _fingerprint(_require_file(probe_sample)),
        "manifest": _fingerprint(_require_file(probe_sample.with_suffix(".json"))),
    }
    documents = {
        path: _fingerprint(_require_file(PROJECT_ROOT / path)) for path in REQUIRED_DOCS
    }
    return {
        "version": 1,
        "stage": "Stage1.2",
        "stage1_2_baseline_commit": git_commit(PROJECT_ROOT),
        "m5a_training_commit": next(iter(training_commits)),
        "environment_versions_at_freeze": runtime_versions(),
        "m5a_training_runtime": runtimes[0],
        "expected_prediction_rows": expected_prediction_rows,
        "stage1_1_status": {
            "m3_interaction_leakage_screen": "directional_pass",
            "m4_available_data_selection_screen": "conditional_pass",
            "position_corrected": "unavailable_missing_external_propensity",
            "m5b_fairness_utility_gate": "failed",
            "selective_suppression_stage2_approval": "not_approved",
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "data_files": data_files,
        "probe_sample_files": probe_files,
        "documents": documents,
    }


def render_report(payload: dict) -> str:
    """Render a concise closure report backed by the machine manifest."""

    lines = [
        "# Stage1.1 closure for Stage1.2",
        "",
        f"- Stage1.2 baseline commit: `{payload['stage1_2_baseline_commit']}`.",
        f"- M5A training commit: `{payload['m5a_training_commit']}`.",
        f"- Frozen M5A runs: `{payload['artifact_count']}`.",
        "- Frozen prediction rows per run: "
        f"`{payload['expected_prediction_rows']}`.",
        "- Position-corrected: `unavailable_missing_external_propensity`.",
        "",
        "| Job | Seed | Checkpoint SHA-256 | Prediction SHA-256 | Config hash |",
        "|---|---:|---|---|---|",
    ]
    for item in payload["artifacts"]:
        lines.append(
            f"| {item['job']} | {item['seed']} | "
            f"`{item['checkpoint']['sha256']}` | `{item['prediction']['sha256']}` | "
            f"`{item['config_hash']}` |"
        )
    lines.extend(
        [
            "",
            "All checkpoints, predictions, configurations, manifests, and probe "
            "samples remain outside the source checkout under the declared "
            "FairJob data/workdir roots.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = freeze_inputs(args.matrix)
    write_run_manifest(args.out, payload)
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_count": payload["artifact_count"],
                "m5a_training_commit": payload["m5a_training_commit"],
                "out": args.out,
                "report": args.report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
