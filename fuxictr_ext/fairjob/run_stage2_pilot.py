"""Run the frozen Stage2 P2 matrix in a foreground, resumable process.

This scheduler launches the existing FairJob FuxiCTR entry point one run at a
time. It does not daemonize, redirect logs, tune weights, or read test metrics
for control flow. A same-seed baseline is always completed before dependent
methods receive its checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "fuxictr_ext/fairjob/run_fuxictr_model.py"
CONFIG_GENERATOR = PROJECT_ROOT / "configs/fairjob/make_model_config.py"
DEFAULT_MATRIX = PROJECT_ROOT / "configs/fairjob/stage2_ablation_matrix.yaml"
DATASET_ID = "fairjob_m5_pre_ranking_no_user_id_proxy_excluded_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_dir",
        default="/root/autodl-tmp/workdirs/JobFairness/stage1_1/runtime_config",
    )
    parser.add_argument(
        "--workdir",
        default="/root/autodl-tmp/workdirs/JobFairness/stage2/pilot",
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the frozen command sequence without starting training.",
    )
    return parser.parse_args()


def load_matrix(path: str | Path) -> dict:
    """Load the frozen P2 seed and method order."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported Stage2 ablation matrix version.")
    if payload.get("backbone") != "DCNv2":
        raise ValueError("The Stage2 P2 scheduler requires the DCNv2 matrix.")
    methods = payload.get("methods", {})
    if not methods or next(iter(methods)) != "baseline":
        raise ValueError("The frozen Stage2 matrix must start with baseline.")
    if payload.get("execution", {}).get("p2_pilot", {}).get(
        "requires_same_seed_baseline_teacher"
    ) is not True:
        raise ValueError("P2 must require a same-seed baseline teacher.")
    return payload


def select_matrix(
    matrix: dict,
    requested_seeds: list[int] | None,
    requested_methods: list[str] | None,
) -> tuple[list[int], list[str]]:
    """Select a preregistered subset without changing frozen order."""

    frozen_seeds = [int(seed) for seed in matrix["seeds"]]
    frozen_methods = list(matrix["methods"])
    seeds = frozen_seeds if requested_seeds is None else requested_seeds
    methods = (
        frozen_methods if requested_methods is None else requested_methods
    )
    unknown_seeds = sorted(set(seeds).difference(frozen_seeds))
    unknown_methods = sorted(set(methods).difference(frozen_methods))
    if unknown_seeds or unknown_methods:
        raise ValueError(
            "Requested jobs are outside the frozen matrix: "
            f"seeds={unknown_seeds}, methods={unknown_methods}"
        )
    if len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods):
        raise ValueError("Seeds and methods cannot contain duplicates.")
    # CLI order cannot silently change method ordering or baseline pairing.
    seeds = [seed for seed in frozen_seeds if seed in seeds]
    methods = [method for method in frozen_methods if method in methods]
    return seeds, methods


def expid(method: str) -> str:
    return f"Stage2DCNv2_{method}_full"


def run_dir(workdir: str | Path, seed: int, method: str) -> Path:
    return Path(workdir) / f"seed{seed}" / method


def checkpoint_path(workdir: str | Path, seed: int) -> Path:
    """Return the exact same-seed baseline checkpoint expected by P2."""

    return (
        run_dir(workdir, seed, "baseline")
        / "checkpoints"
        / DATASET_ID
        / f"{expid('baseline')}.model"
    )


def read_completion(path: str | Path, expected_commit: str) -> str:
    """Classify an existing run as pending, incomplete, complete, or stale."""

    manifest_path = Path(path) / "run_manifest.json"
    if not manifest_path.exists():
        return "pending"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return "incomplete"
    if payload.get("git_commit") != expected_commit:
        return "stale"
    return "complete"


def build_command(
    python: str,
    config_dir: str | Path,
    workdir: str | Path,
    gpu: int,
    seed: int,
    method: str,
) -> list[str]:
    """Build one shell-free foreground training command."""

    command = [
        str(python),
        "-u",
        str(RUNNER),
        "--config_dir",
        str(config_dir),
        "--expid",
        expid(method),
        "--dataset_id",
        DATASET_ID,
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
        "--run_dir",
        str(run_dir(workdir, seed, method)),
    ]
    if method != "baseline":
        command.extend(
            [
                "--backbone_checkpoint",
                str(checkpoint_path(workdir, seed)),
            ]
        )
    return command


def validate_runtime_config(
    config_dir: str | Path, methods: list[str]
) -> None:
    """Verify that generated model and dataset entries cover this P2 run."""

    config_dir = Path(config_dir)
    model_path = config_dir / "model_config.yaml"
    dataset_path = config_dir / "dataset_config.yaml"
    if not model_path.exists() or not dataset_path.exists():
        raise FileNotFoundError(
            f"Runtime config is incomplete: {config_dir}"
        )
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    dataset_config = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    missing = [
        expid(method)
        for method in methods
        if expid(method) not in model_config
    ]
    if missing:
        raise ValueError(f"Runtime model config is missing P2 entries: {missing}")
    if DATASET_ID not in dataset_config:
        raise ValueError(
            f"Runtime dataset config is missing {DATASET_ID}."
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


def execute(args: argparse.Namespace) -> None:
    """Run or print the selected P2 matrix in frozen seed/method order."""

    matrix = load_matrix(args.matrix)
    seeds, methods = select_matrix(matrix, args.seeds, args.methods)
    commit = current_commit()
    config_dir = Path(args.config_dir)
    workdir = Path(args.workdir)

    if not args.dry_run:
        subprocess.run(
            [
                args.python,
                str(CONFIG_GENERATOR),
                "--out",
                str(config_dir / "model_config.yaml"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    validate_runtime_config(config_dir, methods)

    print(
        json.dumps(
            {
                "stage": "S2-P2",
                "git_commit": commit,
                "seeds": seeds,
                "methods": methods,
                "jobs": len(seeds) * len(methods),
                "workdir": str(workdir),
                "foreground": True,
                "test_driven_control_flow": False,
            },
            indent=2,
        ),
        flush=True,
    )

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    for seed in seeds:
        selected = list(methods)
        if not args.dry_run and "baseline" not in selected:
            baseline_status = read_completion(
                run_dir(workdir, seed, "baseline"), commit
            )
            if baseline_status != "complete":
                raise RuntimeError(
                    f"Seed {seed} baseline is {baseline_status}; run it first."
                )
        for method in selected:
            destination = run_dir(workdir, seed, method)
            status = read_completion(destination, commit)
            if not args.dry_run and status == "complete":
                print(
                    f"[P2] skip complete seed={seed} method={method}",
                    flush=True,
                )
                continue
            if not args.dry_run and status == "stale":
                raise RuntimeError(
                    f"Refusing stale completed run at {destination}."
                )
            if not args.dry_run and method != "baseline":
                baseline = checkpoint_path(workdir, seed)
                baseline_status = read_completion(
                    run_dir(workdir, seed, "baseline"), commit
                )
                if baseline_status != "complete" or not baseline.exists():
                    raise RuntimeError(
                        f"Seed {seed} baseline checkpoint is unavailable."
                    )

            command = build_command(
                args.python,
                config_dir,
                workdir,
                args.gpu,
                seed,
                method,
            )
            print(
                f"[P2] start seed={seed} method={method} "
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
            if read_completion(destination, commit) != "complete":
                raise RuntimeError(
                    f"Run returned without a complete manifest: {destination}"
                )
            print(
                f"[P2] complete seed={seed} method={method}",
                flush=True,
            )


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
