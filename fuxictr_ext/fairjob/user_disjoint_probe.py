"""Freeze Stage2 probe rows with disjoint raw FairJob user identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.prepare_probe_samples import (
    SPLITS,
    load_probe_row_ids,
    write_probe_sample,
)
from fuxictr_ext.fairjob.representation_io import sha256_file
from fuxictr_ext.fairjob.run_manifest import git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", default="configs/fairjob/stage2_probe_protocol.yaml"
    )
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--row_sample", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--chunksize", type=int, default=100000)
    return parser.parse_args()


def stable_user_split(
    user_id, seed: int, ratios: dict[str, float]
) -> str:
    """Assign one raw user ID to a deterministic disjoint probe split."""

    if tuple(ratios) != SPLITS:
        raise ValueError(f"User split ratios must follow {SPLITS}.")
    values = np.asarray(list(ratios.values()), dtype=float)
    if np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("User split ratios must be positive and sum to one.")
    token = f"{seed}\0{user_id}".encode("utf-8")
    digest = hashlib.blake2b(token, digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    cumulative = 0.0
    for split, ratio in ratios.items():
        cumulative += ratio
        if value < cumulative:
            return split
    return SPLITS[-1]


def load_selected_identity_rows(
    source: str | Path,
    selected_row_ids: np.ndarray,
    chunksize: int = 100000,
) -> pd.DataFrame:
    """Read raw user IDs only for a frozen, split-local row sample."""

    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    source = Path(source)
    selected_row_ids = np.asarray(selected_row_ids, dtype=np.int64)
    wanted = set(selected_row_ids.tolist())
    parts = []
    for chunk in pd.read_csv(
        source,
        usecols=["row_id", "user_id", "protected_attribute"],
        chunksize=chunksize,
    ):
        selected = chunk[chunk["row_id"].isin(wanted)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError(f"No frozen row IDs were found in {source}.")
    frame = pd.concat(parts, ignore_index=True).sort_values("row_id")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"Identity source contains duplicate row IDs: {source}")
    loaded = frame["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(loaded, selected_row_ids):
        missing = np.setdiff1d(selected_row_ids, loaded)
        raise ValueError(
            f"Identity source is missing {len(missing)} frozen rows: {source}"
        )
    if frame["user_id"].isna().any():
        raise ValueError(f"Identity source has missing raw user IDs: {source}")
    return frame


def freeze_user_disjoint_rows(
    processed_dir: str | Path,
    identity_sources: dict[str, str],
    row_sample: str | Path,
    ratios: dict[str, float],
    seed: int,
    chunksize: int = 100000,
) -> tuple[dict[str, np.ndarray], dict]:
    """Filter frozen row samples through one stable raw-user partition."""

    processed_dir = Path(processed_dir)
    selected = {}
    split_stats = {}
    users_by_split = {}
    source_hashes = {}
    for split in SPLITS:
        source = processed_dir / identity_sources[split]
        source_hashes[split] = sha256_file(source)
        row_ids = load_probe_row_ids(row_sample, split)
        frame = load_selected_identity_rows(source, row_ids, chunksize=chunksize)
        assignment = frame["user_id"].map(
            lambda value: stable_user_split(value, seed, ratios)
        )
        kept = frame[assignment == split].copy()
        if kept.empty:
            raise ValueError(f"User-disjoint split {split} has no rows.")
        kept_ids = kept["row_id"].to_numpy(dtype=np.int64)
        selected[split] = kept_ids
        users = {str(value) for value in kept["user_id"].unique()}
        users_by_split[split] = users
        group_counts = (
            kept["protected_attribute"].astype(int).value_counts().to_dict()
        )
        if not {0, 1}.issubset(group_counts):
            raise ValueError(
                f"User-disjoint split {split} lacks one protected proxy group."
            )
        split_stats[split] = {
            "source_rows": int(len(frame)),
            "selected_rows": int(len(kept)),
            "selected_users": int(len(users)),
            "group_0_rows": int(group_counts[0]),
            "group_1_rows": int(group_counts[1]),
        }
    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = users_by_split[left].intersection(users_by_split[right])
            overlaps[f"{left}:{right}"] = len(overlap)
            if overlap:
                raise ValueError(
                    f"User-disjoint splits {left}/{right} overlap by {len(overlap)} users."
                )
    metadata = {
        "version": 1,
        "stage": "S2-P0.3",
        "git_commit": git_commit(PROJECT_ROOT),
        "assignment": {
            "method": "blake2b_stable_hash",
            "seed": seed,
            "ratios": ratios,
            "identity": "raw_user_id_not_tokenizer_id",
        },
        "processed_dir": str(processed_dir),
        "row_sample": str(row_sample),
        "row_sample_sha256": sha256_file(row_sample),
        "identity_source_sha256": source_hashes,
        "split_stats": split_stats,
        "user_overlaps": overlaps,
    }
    return selected, metadata


def main() -> None:
    args = parse_args()
    protocol = yaml.safe_load(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    if protocol.get("version") != 1:
        raise ValueError("Unsupported Stage2 probe protocol version.")
    processed_dir = args.processed_dir or protocol["processed_dir"]
    row_sample = args.row_sample or protocol["row_disjoint_sample"]
    out = args.out or protocol["user_disjoint_sample"]
    selected, metadata = freeze_user_disjoint_rows(
        processed_dir=processed_dir,
        identity_sources=protocol["identity_sources"],
        row_sample=row_sample,
        ratios=protocol["user_assignment"]["ratios"],
        seed=int(protocol["user_assignment"]["seed"]),
        chunksize=args.chunksize,
    )
    write_probe_sample(out, selected, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
