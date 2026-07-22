"""Generate FuxiCTR dataset_config.yaml entries from FairJob feature_manifest.

FuxiCTR's loader expects a dataset id to resolve to concrete data paths and
feature columns. This generator keeps those entries derived from the prepared
manifest so the unaware/unfair split cannot drift from the data-preparation
logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.protocols import features_for_protocol, load_protocols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature_manifest",
        default="data/FairJob/processed/feature_manifest.json",
    )
    parser.add_argument("--processed_dir", default="data/FairJob/processed")
    parser.add_argument("--data_root", default="data/FairJob/fuxictr")
    parser.add_argument("--out", default="configs/fairjob/dataset_config.yaml")
    parser.add_argument(
        "--task_protocols",
        default="configs/fairjob/task_protocols.yaml",
    )
    parser.add_argument(
        "--fairness_protocols",
        default="configs/fairjob/fairness_protocols.yaml",
    )
    return parser.parse_args()


def feature_cols(
    manifest: dict,
    features: list[str],
    include_fairness_meta: bool = False,
) -> list[dict]:
    """Return FuxiCTR feature column specs for one fairness regime.

    Binary context fields are modeled as categorical features to match the
    reference FairJob setup, while rank/num* fields remain numeric and are
    normalized by FuxiCTR's existing StandardScaler path.
    """

    categorical_candidates = (
        manifest["categorical_id_features"]
        + manifest["categorical_features"]
        + manifest["binary_context_features"]
        + [manifest["protected_feature_for_model"]]
    )
    categorical = [name for name in categorical_candidates if name in features]
    numeric = [name for name in manifest["numeric_features"] if name in features]

    cols = []
    if categorical:
        cols.append(
            {
                "name": categorical,
                "active": True,
                "dtype": "str",
                "type": "categorical",
            }
        )
    if numeric:
        cols.append(
            {
                "name": numeric,
                "active": True,
                "dtype": "float",
                "type": "numeric",
                "normalizer": "StandardScaler",
            }
        )
    if include_fairness_meta:
        # These aliases retain raw binary values in each training batch while
        # FuxiCTR's BaseModel.get_inputs() excludes type=meta columns from the
        # embedding stack. They are loss metadata, never model input features.
        cols.extend(
            [
                {
                    "name": "protected_attribute_meta",
                    "active": True,
                    "dtype": "int",
                    "type": "meta",
                    "remap": False,
                    "preprocess": "copy_from(protected_attribute)",
                },
                {
                    "name": "senior_meta",
                    "active": True,
                    "dtype": "int",
                    "type": "meta",
                    "remap": False,
                    "preprocess": "copy_from(senior)",
                },
            ]
        )
    return cols


def make_entry(
    manifest: dict,
    processed_dir: Path,
    data_root: str,
    features: list[str],
    regime: str,
    regime_name: str,
    protocol: str,
    protocol_name: str,
    mode: str,
    include_fairness_meta: bool = False,
) -> dict:
    """Build one dataset_config entry for a regime/mode pair.

    The four dataset ids are the stable public interface used by model configs:
    fairjob_unaware_smoke, fairjob_unfair_smoke, fairjob_unaware_full, and
    fairjob_unfair_full. The extra fairjob_* fields are ignored by FuxiCTR core
    code but consumed by the FairJob wrapper scripts for prediction alignment.
    """

    if mode == "smoke":
        train_data = processed_dir / "smoke_train.csv"
        valid_data = processed_dir / "smoke_valid.csv"
        test_data = processed_dir / "smoke_test.csv"
        meta_path = processed_dir / "smoke_test_meta.csv"
    else:
        # The validation tail must not also appear in the training file. A
        # separate final-refit protocol may use train_full.csv after selection,
        # but monitored Stage1.1 runs use disjoint train.csv and valid.csv.
        train_data = processed_dir / "train.csv"
        valid_data = processed_dir / "valid.csv"
        test_data = processed_dir / "test.csv"
        meta_path = processed_dir / "test_meta.csv"

    prefix = "smoke_" if mode == "smoke" else ""
    split_meta_paths = {
        "train": processed_dir / f"{prefix}train_meta.csv",
        "valid": processed_dir / f"{prefix}valid_meta.csv",
        "test": meta_path,
    }

    return {
        "data_root": data_root,
        "data_format": "csv",
        "train_data": str(train_data).replace("\\", "/"),
        "valid_data": str(valid_data).replace("\\", "/"),
        "test_data": str(test_data).replace("\\", "/"),
        "min_categr_count": 1,
        "feature_cols": feature_cols(
            manifest, features, include_fairness_meta=include_fairness_meta
        ),
        "label_col": {"name": "click", "dtype": "float"},
        "fairjob_regime": regime,
        "fairjob_regime_name": regime_name,
        "fairjob_protocol": protocol,
        "fairjob_protocol_name": protocol_name,
        "fairjob_mode": mode,
        "fairjob_meta_path": str(meta_path).replace("\\", "/"),
        "fairjob_split_meta_paths": {
            key: str(path).replace("\\", "/") for key, path in split_meta_paths.items()
        },
        "fairjob_processed_dir": str(processed_dir).replace("\\", "/"),
    }


def load_manifest(path: Path) -> dict:
    """Load a generated JSON manifest or the checked-in YAML schema."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Feature manifest must be a mapping: {path}")
    return payload


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.feature_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing feature manifest: {manifest_path}. Run prepare_fairjob.py first."
        )
    manifest = load_manifest(manifest_path)
    processed_dir = Path(args.processed_dir)
    task_protocols, fairness_protocols = load_protocols(
        args.task_protocols, args.fairness_protocols
    )

    config = {}
    # Preserve Stage1.0 IDs for historical anchor commands. New scientific runs
    # use explicit task and proxy-regime names below.
    for regime in ["unaware", "unfair"]:
        for mode in ["smoke", "full"]:
            dataset_id = f"fairjob_{regime}_{mode}"
            features, protocol_name, regime_name = features_for_protocol(
                manifest,
                task_protocols,
                fairness_protocols,
                "legacy_integration",
                regime,
            )
            config[dataset_id] = make_entry(
                manifest=manifest,
                processed_dir=processed_dir,
                data_root=args.data_root,
                features=features,
                regime=regime,
                regime_name=regime_name,
                protocol="legacy_integration",
                protocol_name=protocol_name,
                mode=mode,
            )

    # Every non-legacy task protocol receives dataset entries. This includes
    # identity controls added after M3 without duplicating feature-list logic.
    for protocol in task_protocols["protocols"]:
        if protocol == "legacy_integration":
            continue
        for regime in fairness_protocols["proxy_regimes"]:
            for mode in ["smoke", "full"]:
                features, protocol_name, regime_name = features_for_protocol(
                    manifest,
                    task_protocols,
                    fairness_protocols,
                    protocol,
                    regime,
                )
                dataset_id = f"fairjob_{protocol}_{regime}_{mode}"
                config[dataset_id] = make_entry(
                    manifest=manifest,
                    processed_dir=processed_dir,
                    data_root=args.data_root,
                    features=features,
                    regime=regime,
                    regime_name=regime_name,
                    protocol=protocol,
                    protocol_name=protocol_name,
                    mode=mode,
                )

    # M5 uses the stable no-user-id backbone selected by M3. Raw protected and
    # senior values are carried as meta aliases for training-only mitigation
    # losses, without changing the model-facing proxy-excluded feature list.
    features, protocol_name, regime_name = features_for_protocol(
        manifest,
        task_protocols,
        fairness_protocols,
        "pre_ranking_no_user_id",
        "proxy_excluded",
    )
    for mode in ("smoke", "full"):
        dataset_id = f"fairjob_m5_pre_ranking_no_user_id_proxy_excluded_{mode}"
        config[dataset_id] = make_entry(
            manifest=manifest,
            processed_dir=processed_dir,
            data_root=args.data_root,
            features=features,
            regime="proxy_excluded",
            regime_name=regime_name,
            protocol="pre_ranking_no_user_id",
            protocol_name=protocol_name,
            mode=mode,
            include_fairness_meta=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
