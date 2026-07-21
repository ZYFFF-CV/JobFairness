"""Run the bounded FairJob Stage1.1 proxy-probe matrix in foreground.

The matrix first freezes one fully validated row sample. Input controls and
layer probes then reuse those exact split-local row IDs. Each completed command
has an independent JSON result, so ``--resume`` can continue without rerunning
successful probes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SCRIPT = PROJECT_ROOT / "fuxictr_ext/fairjob/prepare_probe_samples.py"
INPUT_SCRIPT = PROJECT_ROOT / "fuxictr_ext/fairjob/input_proxy_probe.py"
LAYER_SCRIPT = PROJECT_ROOT / "fuxictr_ext/fairjob/proxy_probe.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default="configs/fairjob/stage1_1_probe_matrix.yaml"
    )
    parser.add_argument(
        "--group",
        choices=[
            "input_controls",
            "layer_screening",
            "nonlinear_screening",
            "positive_control",
        ],
    )
    parser.add_argument("--mode", choices=["print", "foreground", "status"], default="print")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_matrix(path: str | Path) -> dict:
    """Load the probe matrix and reject unsupported revisions."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Stage1.1 probe matrix version.")
    return payload


def resolve_layout(matrix: dict) -> tuple[Path, Path, dict[str, Path]]:
    """Resolve all server paths below the isolated Stage1.1 work directory."""

    workdir = Path(matrix["workdir_root"])
    probe_dir = workdir / "probes"
    roots = {
        name: workdir / relative
        for name, relative in matrix["representation_roots"].items()
    }
    return workdir, probe_dir, roots


def sample_command(matrix: dict) -> tuple[list[str], Path]:
    """Build the one-time validation and row-freezing command."""

    _, probe_dir, roots = resolve_layout(matrix)
    output = probe_dir / "probe_rows.npz"
    command = [sys.executable, str(SAMPLE_SCRIPT)]
    for root in roots.values():
        command.extend(["--representation_root", str(root)])
    command.extend(
        [
            "--max_rows_per_split",
            str(matrix["max_rows_per_split"]),
            "--seed",
            str(matrix["seed"]),
            "--out",
            str(output),
        ]
    )
    return command, output


def input_commands(matrix: dict, group: str) -> list[tuple[str, list[str], Path]]:
    """Build raw-input probe commands for one preregistered group."""

    _, probe_dir, _ = resolve_layout(matrix)
    sample_path = probe_dir / "probe_rows.npz"
    commands = []
    for job in matrix["input_jobs"]:
        if job["group"] != group:
            continue
        output = probe_dir / "input" / f"{job['name']}.json"
        command = [
            sys.executable,
            str(INPUT_SCRIPT),
            "--data_dir",
            str(matrix["data_dir"]),
            "--regime",
            job["regime"],
            "--feature_set",
            job["feature_set"],
            "--probe_sample",
            str(sample_path),
            "--seed",
            str(matrix["seed"]),
            "--out",
            str(output),
        ]
        commands.append((job["name"], command, output))
    return commands


def layer_commands(matrix: dict, group: str) -> list[tuple[str, list[str], Path]]:
    """Build one command per frozen representation and linear probe."""

    _, probe_dir, roots = resolve_layout(matrix)
    sample_path = probe_dir / "probe_rows.npz"
    commands = []
    for job in matrix["representation_jobs"]:
        if job["group"] != group:
            continue
        baseline_result = probe_dir / "input" / f"{job['input_baseline']}.json"
        for representation in job["representations"]:
            name = f"{job['name']}__{representation}"
            output = probe_dir / "layers" / f"{name}.json"
            command = [
                sys.executable,
                str(LAYER_SCRIPT),
                "--representation_root",
                str(roots[job["root"]]),
                "--representation",
                representation,
                "--probe_type",
                job.get("probe_type", "linear"),
                "--max_rows_per_split",
                str(matrix["max_rows_per_split"]),
                "--seed",
                str(matrix["seed"]),
                "--probe_sample",
                str(sample_path),
                "--skip_shard_validation",
                "--input_baseline_result",
                str(baseline_result),
                "--out",
                str(output),
            ]
            commands.append((name, command, output))
    return commands


def commands_for_group(matrix: dict, group: str) -> list[tuple[str, list[str], Path]]:
    """Return commands in dependency order for the requested probe group."""

    commands = input_commands(matrix, group)
    if group in {"layer_screening", "nonlinear_screening", "positive_control"}:
        commands.extend(layer_commands(matrix, group))
    if not commands:
        raise ValueError(f"No probe jobs found for group {group!r}.")
    return commands


def ensure_probe_sample(matrix: dict, mode: str, resume: bool) -> None:
    """Create the shared sample before any probe consumes it."""

    command, output = sample_command(matrix)
    if output.exists() and resume:
        return
    if mode == "print":
        print(json.dumps(command))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_foreground(commands: list[tuple[str, list[str], Path]], resume: bool) -> None:
    """Run probes sequentially while preserving live child output."""

    for name, command, output in commands:
        if resume and output.exists():
            print(f"SKIP complete probe: {name}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"START probe: {name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        print(f"DONE probe: {name}", flush=True)


def main() -> None:
    args = parse_args()
    matrix = read_matrix(args.matrix)
    if args.mode == "status":
        commands = commands_for_group(matrix, args.group)
        status = [
            {"name": name, "complete": output.exists(), "output": str(output)}
            for name, _, output in commands
        ]
        print(json.dumps(status, indent=2))
        return
    ensure_probe_sample(matrix, args.mode, args.resume)
    commands = commands_for_group(matrix, args.group)
    if args.mode == "print":
        for _, command, _ in commands:
            print(json.dumps(command))
    else:
        run_foreground(commands, args.resume)


if __name__ == "__main__":
    main()
