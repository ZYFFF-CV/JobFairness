"""Build an external FairJob runtime config directory for one machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_runtime_config(processed_dir: Path, data_root: Path, out_dir: Path) -> dict:
    """Generate dataset paths and copy immutable model/protocol definitions."""

    processed_dir = processed_dir.resolve()
    data_root = data_root.resolve()
    out_dir = out_dir.resolve()
    manifest_path = processed_dir / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prepared feature manifest: {manifest_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_out = out_dir / "dataset_config.yaml"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "configs/fairjob/make_dataset_config.py"),
        "--feature_manifest",
        str(manifest_path),
        "--processed_dir",
        str(processed_dir),
        "--data_root",
        str(data_root),
        "--out",
        str(dataset_out),
        "--task_protocols",
        str(PROJECT_ROOT / "configs/fairjob/task_protocols.yaml"),
        "--fairness_protocols",
        str(PROJECT_ROOT / "configs/fairjob/fairness_protocols.yaml"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    source_files = {
        "model_config.yaml": PROJECT_ROOT / "configs/fairjob/model_config.yaml",
        "task_protocols.yaml": PROJECT_ROOT / "configs/fairjob/task_protocols.yaml",
        "fairness_protocols.yaml": PROJECT_ROOT / "configs/fairjob/fairness_protocols.yaml",
    }
    for name, source in source_files.items():
        shutil.copy2(source, out_dir / name)

    files = [dataset_out, *(out_dir / name for name in source_files)]
    payload = {
        "version": 1,
        "processed_dir": str(processed_dir),
        "data_root": str(data_root),
        "files": {path.name: sha256_file(path) for path in files},
    }
    (out_dir / "runtime_config_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = build_runtime_config(
        Path(args.processed_dir), Path(args.data_root), Path(args.out_dir)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
