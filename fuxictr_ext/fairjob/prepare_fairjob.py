"""Prepare FairJob CSV splits for FuxiCTR and FairJob-specific evaluation.

This script keeps the original FairJob protocol visible in generated artifacts:
the full test split is always the final 20% of the raw file, evaluator metadata
is exported before FuxiCTR tokenization, and the protected attribute is copied to
a model-facing feature only for the unfair regime. The raw reference repository
is intentionally not imported here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


FIXED_COLUMNS = [
    "click",
    "protected_attribute",
    "senior",
    "displayrandom",
    "rank",
    "user_id",
    "impression_id",
    "product_id",
]

# Evaluators must group by the raw IDs from this file, not by the FuxiCTR
# tokenized feature values produced later by FeatureProcessor.
META_COLUMNS = [
    "row_id",
    "click",
    "protected_attribute",
    "senior",
    "displayrandom",
    "impression_id",
    "product_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_path", default="data/FairJob/raw/fairjob.csv.gz")
    parser.add_argument("--out_dir", default="data/FairJob/processed")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--valid_rows", type=int, default=50000)
    parser.add_argument("--smoke_rows", type=int, default=50000)
    parser.add_argument("--format", choices=["csv"], default="csv")
    parser.add_argument(
        "--feature_schema",
        default="configs/fairjob/feature_schema.yaml",
        help="Path to write a human-readable feature schema.",
    )
    return parser.parse_args()


def natural_feature_order(columns: Iterable[str]) -> list[str]:
    """Sort cat*/num* columns in numeric suffix order.

    The FairJob task cards require dynamic header inspection instead of assuming
    a fixed final num column. Natural ordering keeps generated configs stable
    across repeated runs and across raw files with different feature counts.
    """

    def key(name: str) -> tuple[str, int | str]:
        prefix = "".join(ch for ch in name if not ch.isdigit())
        suffix = name[len(prefix) :]
        return prefix, int(suffix) if suffix.isdigit() else suffix

    return sorted(columns, key=key)


def detect_features(df: pd.DataFrame) -> dict[str, list[str] | str]:
    """Build the feature manifest that drives all downstream configs.

    The manifest is the single source of truth for unaware/unfair feature lists.
    Unaware never receives any protected attribute column; unfair receives the
    explicit copy ``protected_attribute_feat`` so evaluator code can reserve
    ``protected_attribute`` for raw metadata only.
    """

    missing = [col for col in FIXED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required FairJob columns: {missing}")

    cat_features = natural_feature_order(
        col for col in df.columns if col.startswith("cat")
    )
    num_features = ["rank"] + natural_feature_order(
        col for col in df.columns if col.startswith("num")
    )

    return {
        "label": "click",
        "protected_feature": "protected_attribute",
        "protected_feature_for_model": "protected_attribute_feat",
        "categorical_id_features": ["user_id", "impression_id", "product_id"],
        "categorical_features": cat_features,
        "binary_context_features": ["senior", "displayrandom"],
        "numeric_features": num_features,
    }


def assert_binary(df: pd.DataFrame, columns: list[str]) -> None:
    """Fail early if FairJob's binary context columns are malformed."""

    for col in columns:
        values = set(df[col].dropna().unique().tolist())
        if not values.issubset({0, 1, 0.0, 1.0, "0", "1"}):
            raise ValueError(f"Column {col} is not binary: {sorted(values)[:10]}")


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize binary fields and create the model-facing protected feature.

    ``protected_attribute_feat`` deliberately duplicates the raw protected
    attribute. FuxiCTR dataset configs may include this column for unfair runs,
    while metric code always reads the original value from the meta CSV.
    """

    df = df.copy()
    for col in ["click", "protected_attribute", "senior", "displayrandom"]:
        df[col] = df[col].astype(int)
    df["protected_attribute_feat"] = df["protected_attribute"].astype(int)
    return df


def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the 0-based split-local row id used for strict prediction joins."""

    df = df.reset_index(drop=True).copy()
    df.insert(0, "row_id", range(len(df)))
    return df


def split_train_valid(
    train_candidate: pd.DataFrame, valid_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    """Cut a deterministic validation tail from the train candidate rows.

    The FairJob paper protocol fixes the outer train/test split. This inner
    train/valid split is only for FuxiCTR monitoring and is recorded in the
    split manifest so it is auditable.
    """

    if len(train_candidate) < 2:
        raise ValueError("Need at least two rows in train candidate split.")
    capped_valid_rows = min(valid_rows, max(1, int(len(train_candidate) * 0.1)))
    train = train_candidate.iloc[: len(train_candidate) - capped_valid_rows].copy()
    valid = train_candidate.iloc[len(train_candidate) - capped_valid_rows :].copy()
    policy = {
        "valid_policy": "tail_slice_from_train_candidate",
        "requested_valid_rows": valid_rows,
        "actual_valid_rows": len(valid),
    }
    return train, valid, policy


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} rows={len(df)}")


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_schema(path: Path, feature_manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(feature_manifest, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"Wrote {path}")


def prepare_split(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    test_ratio: float,
    valid_rows: int,
) -> dict:
    """Write one full or smoke split and return its manifest summary.

    For the full split, ``train_full.csv`` preserves the complete first 80% train
    candidate block. ``train.csv``/``valid.csv`` are derived from that block for
    FuxiCTR training. The test file and meta file keep identical row order.
    """

    cut = int(len(df) * (1.0 - test_ratio))
    if cut <= 0 or cut >= len(df):
        raise ValueError(f"Invalid split cut={cut} for rows={len(df)}")

    train_candidate = df.iloc[:cut].copy()
    test = add_row_id(df.iloc[cut:].copy())
    train, valid, valid_policy = split_train_valid(train_candidate, valid_rows)

    if prefix:
        train_name = f"{prefix}_train.csv"
        valid_name = f"{prefix}_valid.csv"
        test_name = f"{prefix}_test.csv"
        meta_name = f"{prefix}_test_meta.csv"
    else:
        write_csv(add_row_id(train_candidate), out_dir / "train_full.csv")
        train_name = "train.csv"
        valid_name = "valid.csv"
        test_name = "test.csv"
        meta_name = "test_meta.csv"

    write_csv(add_row_id(train), out_dir / train_name)
    write_csv(add_row_id(valid), out_dir / valid_name)
    write_csv(test, out_dir / test_name)
    write_csv(test[META_COLUMNS], out_dir / meta_name)

    return {
        "prefix": prefix or "full",
        "input_rows": len(df),
        "train_candidate_rows": len(train_candidate),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "test_start_original_offset": cut,
        **valid_policy,
    }


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_path)
    out_dir = Path(args.out_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"FairJob raw data not found: {raw_path}. "
            "Place fairjob.csv.gz there or pass --raw_path."
        )

    df = pd.read_csv(raw_path)
    df = normalize_types(df)
    assert_binary(df, ["click", "protected_attribute", "senior", "displayrandom"])
    feature_manifest = detect_features(df)

    # The split manifest is part of the reproducibility contract: report builders
    # and reviewers can verify exactly which rows were used without opening the
    # large CSV files again.
    split_manifest = {
        "raw_path": str(raw_path),
        "out_dir": str(out_dir),
        "format": args.format,
        "test_ratio": args.test_ratio,
        "full": prepare_split(df, out_dir, "", args.test_ratio, args.valid_rows),
    }

    smoke_rows = min(args.smoke_rows, len(df))
    smoke_df = df.iloc[:smoke_rows].copy()
    split_manifest["smoke"] = prepare_split(
        smoke_df,
        out_dir,
        "smoke",
        args.test_ratio,
        min(args.valid_rows, max(1, int(smoke_rows * 0.1))),
    )

    feature_manifest["unaware_features"] = (
        feature_manifest["categorical_id_features"]
        + feature_manifest["categorical_features"]
        + feature_manifest["binary_context_features"]
        + feature_manifest["numeric_features"]
    )
    feature_manifest["unfair_features"] = (
        feature_manifest["unaware_features"]
        + [feature_manifest["protected_feature_for_model"]]
    )

    write_manifest(out_dir / "split_manifest.json", split_manifest)
    write_manifest(out_dir / "feature_manifest.json", feature_manifest)
    write_schema(Path(args.feature_schema), feature_manifest)


if __name__ == "__main__":
    main()
