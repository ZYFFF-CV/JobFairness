"""Run the frozen Stage2 P3 checkpoint audits in the foreground.

Each child process handles one checkpoint and releases its in-memory
representations before the next job starts. Completed results from the current
audit commit are skipped, while stale or partial results remain explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from fuxictr_ext.fairjob.run_stage2_pilot import (
    DATASET_ID,
    expid,
    load_matrix,
    run_dir,
    select_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDITOR = PROJECT_ROOT / "fuxictr_ext/fairjob/stage2_checkpoint_probe.py"
DEFAULT_MATRIX = PROJECT_ROOT / "configs/fairjob/stage2_ablation_matrix.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_dir",
        default="/root/autodl-tmp/workdirs/JobFairness/stage1_1/runtime_config",
    )
    parser.add_argument(
        "--pilot_workdir",
        default="/root/autodl-tmp/workdirs/JobFairness/stage2/pilot",
    )
    parser.add_argument(
        "--probe_workdir",
        default="/root/autodl-tmp/workdirs/JobFairness/stage2/probes",
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument(
        "--graph", default="configs/fairjob/stage2_representation_graph.yaml"
    )
    parser.add_argument(
        "--protocol", default="configs/fairjob/stage2_probe_protocol.yaml"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--probe_seed", type=int, default=2019)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def output_path(probe_workdir: str | Path, seed: int, method: str) -> Path:
    """Return the compact per-checkpoint P3 result path."""

    return (
        Path(probe_workdir)
        / f"seed{seed}"
        / method
        / "checkpoint_probe.json"
    )


def read_probe_completion(path: str | Path, audit_commit: str) -> str:
    """Classify a P3 output without silently accepting stale audit code."""

    path = Path(path)
    if not path.exists():
        return "pending"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return "incomplete"
    if payload.get("audit_git_commit") != audit_commit:
        return "stale"
    return "complete"


def build_command(
    python: str,
    config_dir: str | Path,
    pilot_workdir: str | Path,
    probe_workdir: str | Path,
    graph: str | Path,
    protocol: str | Path,
    gpu: int,
    probe_seed: int,
    seed: int,
    method: str,
) -> list[str]:
    """Build one shell-free P3 checkpoint command."""

    source_run = run_dir(pilot_workdir, seed, method)
    return [
        str(python),
        "-u",
        str(AUDITOR),
        "--config_dir",
        str(config_dir),
        "--expid",
        expid(method),
        "--dataset_id",
        DATASET_ID,
        "--run_dir",
        str(source_run),
        "--graph",
        str(graph),
        "--protocol",
        str(protocol),
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
        "--probe_seed",
        str(probe_seed),
        "--out",
        str(output_path(probe_workdir, seed, method)),
    ]


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def execute(args: argparse.Namespace) -> None:
    """Run or print the selected P3 jobs in frozen matrix order."""

    matrix = load_matrix(args.matrix)
    seeds, methods = select_matrix(matrix, args.seeds, args.methods)
    audit_commit = current_commit()
    print(
        json.dumps(
            {
                "stage": "S2-P3",
                "audit_git_commit": audit_commit,
                "seeds": seeds,
                "methods": methods,
                "jobs": len(seeds) * len(methods),
                "storage_mode": "in_memory_checkpoint_reconstructable",
                "foreground": True,
            },
            indent=2,
        ),
        flush=True,
    )

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    for seed in seeds:
        for method in methods:
            source_run = run_dir(args.pilot_workdir, seed, method)
            training_manifest = source_run / "run_manifest.json"
            if not args.dry_run:
                if not training_manifest.exists():
                    raise FileNotFoundError(
                        f"P2 manifest is unavailable: {training_manifest}"
                    )
                training = json.loads(
                    training_manifest.read_text(encoding="utf-8")
                )
                if training.get("status") != "complete":
                    raise ValueError(f"P2 run is not complete: {source_run}")

            destination = output_path(args.probe_workdir, seed, method)
            status = read_probe_completion(destination, audit_commit)
            if not args.dry_run and status == "complete":
                print(
                    f"[P3] skip complete seed={seed} method={method}",
                    flush=True,
                )
                continue
            if not args.dry_run and status == "stale":
                raise RuntimeError(
                    f"Refusing stale P3 result at {destination}."
                )
            command = build_command(
                args.python,
                args.config_dir,
                args.pilot_workdir,
                args.probe_workdir,
                args.graph,
                args.protocol,
                args.gpu,
                args.probe_seed,
                seed,
                method,
            )
            print(
                f"[P3] start seed={seed} method={method} "
                f"previous_status={status}",
                flush=True,
            )
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                env=environment,
            )
            if read_probe_completion(destination, audit_commit) != "complete":
                raise RuntimeError(
                    f"P3 job returned without a complete result: {destination}"
                )
            print(
                f"[P3] complete seed={seed} method={method}",
                flush=True,
            )


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
