"""Reproducibility manifests for FairJob server runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def stable_hash(payload: dict) -> str:
    """Hash a mapping after deterministic JSON serialization."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit(project_root: str | Path) -> str:
    """Return the exact source revision used by a run."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def runtime_versions() -> dict:
    """Collect versions that can change preprocessing or model numerics."""

    packages = {}
    for name in ("fuxictr", "numpy", "pandas", "scikit-learn", "torch", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        runtime["cuda_available"] = torch.cuda.is_available()
        runtime["cuda_version"] = torch.version.cuda
        runtime["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        runtime["cuda_available"] = False
        runtime["cuda_version"] = None
        runtime["gpu_name"] = None
    return runtime


def write_run_manifest(path: str | Path, payload: dict) -> None:
    """Atomically replace a JSON run manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
