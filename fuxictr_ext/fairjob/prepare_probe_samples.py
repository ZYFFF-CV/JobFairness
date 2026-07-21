"""Freeze aligned row samples for FairJob Stage1.1 leakage probes.

Every raw-input and hidden-representation probe must use the same examples.
This command validates each exported representation set, verifies that their
row IDs agree, and writes one deterministic sample per data split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.representation_io import load_representation_row_ids
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation_root",
        action="append",
        required=True,
        help="Repeat for every model whose exported rows must match.",
    )
    parser.add_argument("--max_rows_per_split", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def select_row_ids(row_ids: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Select a sorted uniform sample without changing split membership."""

    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(row_ids) <= max_rows:
        return row_ids.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_ids, size=max_rows, replace=False))


def freeze_probe_samples(
    representation_roots: list[str | Path],
    max_rows_per_split: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Validate model exports and return identical sampled IDs for all probes."""

    if not representation_roots:
        raise ValueError("At least one representation root is required.")
    roots = [Path(root) for root in representation_roots]
    selected = {}
    source_counts = {}
    for split_index, split in enumerate(SPLITS):
        canonical = None
        for root in roots:
            _, row_ids = load_representation_row_ids(root / split, validate_shards=True)
            if canonical is None:
                canonical = row_ids
            elif not np.array_equal(canonical, row_ids):
                raise ValueError(
                    f"Representation row IDs differ for split {split}: {root}"
                )
        source_counts[split] = int(len(canonical))
        selected[split] = select_row_ids(
            canonical, max_rows_per_split, seed + split_index
        )
    metadata = {
        "git_commit": git_commit(PROJECT_ROOT),
        "representation_roots": [str(root) for root in roots],
        "max_rows_per_split": max_rows_per_split,
        "seed": seed,
        "source_counts": source_counts,
        "sample_counts": {split: int(len(ids)) for split, ids in selected.items()},
        "validation": "full_shard_hash_schema_finiteness_and_row_alignment",
    }
    return selected, metadata


def write_probe_sample(path: str | Path, selected: dict, metadata: dict) -> None:
    """Write row IDs to NPZ and an adjacent auditable JSON manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **selected)
    write_run_manifest(path.with_suffix(".json"), metadata)


def load_probe_row_ids(path: str | Path, split: str) -> np.ndarray:
    """Load and validate one split from a frozen probe sample."""

    if split not in SPLITS:
        raise KeyError(f"Unknown probe split: {split}")
    with np.load(path) as payload:
        row_ids = np.asarray(payload[split], dtype=np.int64)
    if len(row_ids) > 1 and not np.all(np.diff(row_ids) > 0):
        raise ValueError(f"Probe row IDs are not strictly increasing for {split}.")
    return row_ids


def main() -> None:
    args = parse_args()
    selected, metadata = freeze_probe_samples(
        args.representation_root, args.max_rows_per_split, args.seed
    )
    write_probe_sample(args.out, selected, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
