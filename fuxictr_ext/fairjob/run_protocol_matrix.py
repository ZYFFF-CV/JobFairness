"""Run or detach an idempotent Stage1.1 FairJob experiment matrix.

The detached mode starts a new server-side process session with logs redirected
to the Stage1.1 workdir. Completed jobs receive a success marker and are skipped
with ``--resume``; this is experiment-level recovery and does not claim to
restore optimizer state inside a partially completed FuxiCTR epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "fuxictr_ext/fairjob/run_fuxictr_model.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/fairjob/stage1_1_matrix.yaml")
    parser.add_argument("--group", required=True)
    parser.add_argument(
        "--mode",
        choices=["print", "foreground", "detach", "status"],
        default="print",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--representations_only",
        action="store_true",
        help="Re-export aligned representations from completed checkpoints.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected_commit", default=None)
    return parser.parse_args()


def read_matrix(path: str | Path) -> dict:
    """Load and minimally validate the matrix document."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("jobs"), list):
        raise ValueError("Unsupported or malformed Stage1.1 matrix.")
    return payload


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_for_job(
    job: dict,
    matrix: dict,
    dry_run: bool,
    representations_only: bool = False,
) -> tuple[list[str], Path]:
    """Build a shell-free Python command and its isolated run directory."""

    expid = job["expid"]
    dataset_id = job["dataset_id"]
    phase = "dry_run" if dry_run else "training"
    if dry_run and representations_only:
        raise ValueError("Representation-only export cannot be a dry run.")
    if dry_run:
        if not expid.endswith("_full") or not dataset_id.endswith("_full"):
            raise ValueError("Dry-run matrix jobs must point to full experiment IDs.")
        expid = expid[:-5] + "_smoke"
        dataset_id = dataset_id[:-5] + "_smoke"
    run_dir = Path(matrix["workdir_root"]) / phase / job["name"]
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config_dir",
        str(matrix.get("config_dir", PROJECT_ROOT / "configs/fairjob")),
        "--expid",
        expid,
        "--dataset_id",
        dataset_id,
        "--gpu",
        str(matrix.get("gpu", 0)),
        "--run_dir",
        str(run_dir),
    ]
    if representations_only:
        command.extend(
            [
                "--export_representations_only",
                "--representation_out",
                str(run_dir / "representations_aligned"),
                "--representation_splits",
                "train,valid,test",
                "--representation_max_rows_per_split",
                "200000",
                "--representation_seed",
                str(matrix.get("representation_seed", 2019)),
            ]
        )
    elif dry_run:
        command.append("--dry_run")
    elif job.get("export_representations", False):
        command.extend(
            [
                "--representation_out",
                str(run_dir / "representations"),
                "--representation_splits",
                "train,valid,test",
                "--representation_max_rows_per_split",
                "200000",
            ]
        )
    if "seed" in job:
        command.extend(["--seed", str(job["seed"])])
    return command, run_dir


def select_jobs(matrix: dict, group: str) -> list[dict]:
    jobs = [job for job in matrix["jobs"] if job.get("group") == group]
    if not jobs:
        raise ValueError(f"No Stage1.1 jobs found for group {group!r}.")
    return jobs


def run_streaming(command: list[str], cwd: Path, log_path: Path) -> int:
    """Mirror child output to the terminal and an on-disk log in real time."""

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        if process.stdout is None:
            raise RuntimeError("Unable to capture Stage1.1 job output.")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def run_foreground(args: argparse.Namespace, matrix: dict) -> None:
    """Run jobs sequentially with live terminal output and durable logs."""

    commit = current_commit()
    if args.expected_commit and commit != args.expected_commit:
        raise ValueError(f"Expected commit {args.expected_commit}, found {commit}.")
    for job in select_jobs(matrix, args.group):
        command, run_dir = command_for_job(
            job, matrix, args.dry_run, args.representations_only
        )
        artifact_prefix = (
            "representation_export" if args.representations_only else "job"
        )
        success_path = run_dir / f"{artifact_prefix}.success.json"
        if args.resume and success_path.exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            run_dir / f"{artifact_prefix}.command.json",
            {"command": command, "git_commit": commit, "job": job},
        )
        started = datetime.now(timezone.utc).isoformat()
        returncode = run_streaming(
            command, PROJECT_ROOT, run_dir / f"{artifact_prefix}.runner.log"
        )
        payload = {
            "job": job["name"],
            "git_commit": commit,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
        }
        if returncode != 0:
            atomic_json(run_dir / f"{artifact_prefix}.failed.json", payload)
            raise RuntimeError(f"Stage1.1 job failed: {job['name']}")
        atomic_json(success_path, payload)


def detach(args: argparse.Namespace, matrix: dict) -> None:
    """Launch the foreground supervisor in a new process session."""

    root = Path(matrix["workdir_root"]) / "supervisors"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = root / f"{args.group}-{stamp}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--matrix",
        str(Path(args.matrix).resolve()),
        "--group",
        args.group,
        "--mode",
        "foreground",
        "--resume",
    ]
    if args.dry_run:
        command.append("--dry_run")
    if args.representations_only:
        command.append("--representations_only")
    if args.expected_commit:
        command.extend(["--expected_commit", args.expected_commit])
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log.close()
    payload = {
        "pid": process.pid,
        "command": command,
        "log_path": str(log_path),
        "git_commit": current_commit(),
    }
    atomic_json(root / f"{args.group}-{stamp}.pid.json", payload)
    print(json.dumps(payload, indent=2))


def print_status(args: argparse.Namespace, matrix: dict) -> None:
    """Print machine-readable completion state for every selected job."""

    status = []
    for job in select_jobs(matrix, args.group):
        _, run_dir = command_for_job(
            job, matrix, args.dry_run, args.representations_only
        )
        artifact_prefix = (
            "representation_export" if args.representations_only else "job"
        )
        state = "pending"
        marker = None
        if (run_dir / f"{artifact_prefix}.success.json").exists():
            state = "complete"
            marker = run_dir / f"{artifact_prefix}.success.json"
        elif (run_dir / f"{artifact_prefix}.failed.json").exists():
            state = "failed"
            marker = run_dir / f"{artifact_prefix}.failed.json"
        elif (run_dir / f"{artifact_prefix}.command.json").exists():
            state = "started"
            marker = run_dir / f"{artifact_prefix}.command.json"
        status.append(
            {
                "job": job["name"],
                "state": state,
                "run_dir": str(run_dir),
                "marker": str(marker) if marker else None,
            }
        )
    print(json.dumps(status, indent=2))


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    matrix = read_matrix(args.matrix)
    if args.mode == "print":
        for job in select_jobs(matrix, args.group):
            command, _ = command_for_job(
                job, matrix, args.dry_run, args.representations_only
            )
            print(json.dumps(command))
    elif args.mode == "foreground":
        run_foreground(args, matrix)
    elif args.mode == "detach":
        detach(args, matrix)
    else:
        print_status(args, matrix)


if __name__ == "__main__":
    main()
