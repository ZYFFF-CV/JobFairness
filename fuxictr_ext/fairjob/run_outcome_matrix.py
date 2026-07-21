"""Run the bounded Stage1.1 outcome-diagnostic matrix in foreground."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_SCRIPT = PROJECT_ROOT / "fuxictr_ext/fairjob/outcome_diagnostics.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default="configs/fairjob/stage1_1_outcome_matrix.yaml"
    )
    parser.add_argument("--mode", choices=["print", "foreground", "status"], default="print")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_matrix(path: str | Path) -> dict:
    """Read one supported outcome matrix revision."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not payload.get("jobs"):
        raise ValueError("Unsupported or empty outcome matrix.")
    return payload


def commands(matrix: dict) -> list[tuple[str, list[str], Path]]:
    """Build one isolated diagnostic command per prediction artifact."""

    root = Path(matrix["workdir_root"])
    output_root = root / "outcome_diagnostics"
    result = []
    for job in matrix["jobs"]:
        output = output_root / f"{job['name']}.json"
        command = [
            sys.executable,
            str(DIAGNOSTIC_SCRIPT),
            "--pred",
            str(root / job["prediction"]),
            "--meta",
            str(matrix["meta_path"]),
            "--context_columns",
            str(matrix.get("context_columns", "displayrandom,rank")),
            "--bootstrap_repeats",
            str(matrix.get("bootstrap_repeats", 20)),
            "--seed",
            str(matrix.get("seed", 2019)),
            "--out",
            str(output),
        ]
        result.append((job["name"], command, output))
    return result


def main() -> None:
    args = parse_args()
    matrix = read_matrix(args.matrix)
    jobs = commands(matrix)
    if args.mode == "status":
        print(
            json.dumps(
                [
                    {"name": name, "complete": output.exists(), "output": str(output)}
                    for name, _, output in jobs
                ],
                indent=2,
            )
        )
        return
    for name, command, output in jobs:
        if args.mode == "print":
            print(json.dumps(command))
            continue
        if args.resume and output.exists():
            print(f"SKIP complete diagnostic: {name}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"START diagnostic: {name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        print(f"DONE diagnostic: {name}", flush=True)


if __name__ == "__main__":
    main()
