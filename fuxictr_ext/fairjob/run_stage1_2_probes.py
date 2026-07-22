"""Run frozen post-intervention leakage probes for Stage1.2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = PROJECT_ROOT / "fuxictr_ext/fairjob/proxy_probe.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default="configs/fairjob/stage1_2_probe_matrix.yaml"
    )
    parser.add_argument("--group", choices=["smoke", "full"], required=True)
    parser.add_argument(
        "--mode", choices=["print", "foreground", "status"], default="print"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_matrix(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Stage1.2 probe matrix version.")
    return payload


def probe_commands(matrix: dict, group: str) -> list[tuple[str, list[str], Path]]:
    """Build one resumable command per checkpoint and representation."""

    export_matrix = yaml.safe_load(
        Path(matrix["export_matrix"]).read_text(encoding="utf-8")
    )
    workdir = Path(matrix["workdir_root"])
    commands = []
    for job in export_matrix["jobs"]:
        representations = matrix["representations"]
        if group == "smoke":
            if job["name"] != matrix["smoke"]["job"]:
                continue
            representations = [matrix["smoke"]["representation"]]
        for representation in representations:
            name = f"{job['name']}__{representation}"
            output = workdir / "probes" / group / f"{name}.json"
            root = (
                workdir
                / "representation_exports"
                / job["name"]
                / "representations"
            )
            command = [
                sys.executable,
                str(PROBE_SCRIPT),
                "--representation_root",
                str(root),
                "--representation",
                representation,
                "--probe_type",
                "both",
                "--max_rows_per_split",
                str(matrix["max_rows_per_split"]),
                "--seed",
                str(matrix["probe_seed"]),
                "--probe_sample",
                str(matrix["probe_sample"]),
                "--skip_shard_validation",
                "--linear_max_iter",
                str(matrix["linear_max_iter"]),
                "--nonlinear_max_iter",
                str(matrix["nonlinear_max_iter"]),
                "--out",
                str(output),
            ]
            if matrix.get("nonlinear_early_stopping", False):
                command.append("--nonlinear_early_stopping")
            commands.append((name, command, output))
    if not commands:
        raise ValueError(f"No Stage1.2 probes found for group {group!r}.")
    return commands


def run_foreground(commands: list[tuple[str, list[str], Path]], resume: bool) -> None:
    """Run probes sequentially with visible output and per-probe receipts."""

    for index, (name, command, output) in enumerate(commands, start=1):
        if resume and output.exists():
            print(f"SKIP [{index}/{len(commands)}] {name}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"START [{index}/{len(commands)}] {name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        print(f"DONE [{index}/{len(commands)}] {name}", flush=True)


def main() -> None:
    args = parse_args()
    matrix = read_matrix(args.matrix)
    commands = probe_commands(matrix, args.group)
    if args.mode == "status":
        print(
            json.dumps(
                [
                    {"name": name, "complete": output.exists(), "output": str(output)}
                    for name, _, output in commands
                ],
                indent=2,
            )
        )
    elif args.mode == "print":
        for _, command, _ in commands:
            print(json.dumps(command))
    else:
        run_foreground(commands, args.resume)


if __name__ == "__main__":
    main()
