"""Run the frozen Stage2.1 Tier-1 checkpoint audits in the foreground."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from fuxictr_ext.fairjob.representation_io import sha256_file
from fuxictr_ext.fairjob.run_stage2_pilot import (
    DATASET_ID,
    expid,
    run_dir,
)
from fuxictr_ext.fairjob.stage2_1.checkpoint_audit import (
    load_stage2_1_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDITOR = PROJECT_ROOT / "fuxictr_ext/fairjob/stage2_1/checkpoint_audit.py"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/fairjob/stage2_1_audit_protocol.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--closure_manifest",
        default=(
            "/root/autodl-tmp/workdirs/JobFairness/stage2_1/"
            "manifests/stage2_1_closure.json"
        ),
    )
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--pilot_workdir", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--probe_seed", type=int, default=2019)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def output_path(workdir: str | Path, seed: int, method: str) -> Path:
    return (
        Path(workdir)
        / "probes"
        / f"seed{seed}"
        / method
        / "checkpoint_audit.json"
    )


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_closure(
    path: str | Path,
    protocol_path: str | Path,
    audit_commit: str,
) -> dict:
    """Require an M0 manifest frozen by the exact audit commit and protocol."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stage2.1 closure manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("stage") != "S21-M0":
        raise ValueError("Stage2.1 closure manifest is not frozen.")
    if payload.get("stage2_1_code_commit") != audit_commit:
        raise ValueError("Stage2.1 closure manifest has a stale code commit.")
    if payload.get("stage2_1_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Stage2.1 closure manifest has a stale protocol hash.")
    if payload.get("p2_run_count") != 27:
        raise ValueError("Stage2.1 closure manifest does not contain 27 P2 runs.")
    return payload


def read_completion(
    path: str | Path,
    audit_commit: str,
    protocol_sha256: str,
) -> str:
    path = Path(path)
    if not path.exists():
        return "pending"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return "incomplete"
    if payload.get("audit_git_commit") != audit_commit:
        return "stale"
    if payload.get("stage2_1_protocol_sha256") != protocol_sha256:
        return "stale"
    return "complete"


def build_command(
    python: str,
    protocol: str | Path,
    config_dir: str | Path,
    pilot_workdir: str | Path,
    workdir: str | Path,
    gpu: int,
    probe_seed: int,
    seed: int,
    method: str,
) -> list[str]:
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
        "--protocol",
        str(protocol),
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
        "--probe_seed",
        str(probe_seed),
        "--out",
        str(output_path(workdir, seed, method)),
    ]


def execute(args: argparse.Namespace) -> None:
    """Run the four frozen Tier-1 audits with job-level resume."""

    protocol = load_stage2_1_protocol(args.protocol)
    paths = protocol["paths"]
    config_dir = args.config_dir or paths["runtime_config"]
    pilot_workdir = args.pilot_workdir or paths["pilot_workdir"]
    workdir = args.workdir or paths["stage2_1_workdir"]
    frozen_seeds = [
        int(seed) for seed in protocol["seed_roles"]["internal_confirmatory"]
    ]
    seeds = frozen_seeds if args.seeds is None else args.seeds
    unknown = sorted(set(seeds).difference(frozen_seeds))
    if unknown or len(seeds) != len(set(seeds)):
        raise ValueError(f"Seeds are outside frozen Tier 1: {unknown}")
    seeds = [seed for seed in frozen_seeds if seed in seeds]
    methods = [
        protocol["methods"]["baseline"],
        protocol["methods"]["primary"],
    ]
    audit_commit = current_commit()
    closure = validate_closure(
        args.closure_manifest,
        args.protocol,
        audit_commit,
    )
    protocol_sha256 = closure["stage2_1_protocol_sha256"]

    print(
        json.dumps(
            {
                "stage": "S21-M3",
                "audit_git_commit": audit_commit,
                "seeds": seeds,
                "methods": methods,
                "jobs": len(seeds) * len(methods),
                "foreground": True,
                "new_training": False,
                "storage_mode": protocol["storage"]["mode"],
            },
            indent=2,
        ),
        flush=True,
    )
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    for seed in seeds:
        for method in methods:
            source_run = run_dir(pilot_workdir, seed, method)
            manifest_path = source_run / "run_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"P2 run is unavailable: {source_run}")
            training = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                training.get("status") != "complete"
                or training.get("git_commit")
                != closure["stage2_training_commit"]
            ):
                raise ValueError(f"P2 run is stale or incomplete: {source_run}")

            destination = output_path(workdir, seed, method)
            status = read_completion(
                destination,
                audit_commit,
                protocol_sha256,
            )
            if status == "complete":
                print(
                    f"[S21] skip complete seed={seed} method={method}",
                    flush=True,
                )
                continue
            if status == "stale":
                raise RuntimeError(
                    f"Refusing stale Stage2.1 result at {destination}."
                )
            command = build_command(
                args.python,
                args.protocol,
                config_dir,
                pilot_workdir,
                workdir,
                args.gpu,
                args.probe_seed,
                seed,
                method,
            )
            print(
                f"[S21] start seed={seed} method={method} "
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
            if (
                read_completion(
                    destination,
                    audit_commit,
                    protocol_sha256,
                )
                != "complete"
            ):
                raise RuntimeError(
                    f"Stage2.1 job returned incomplete: {destination}"
                )
            print(
                f"[S21] complete seed={seed} method={method}",
                flush=True,
            )


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
