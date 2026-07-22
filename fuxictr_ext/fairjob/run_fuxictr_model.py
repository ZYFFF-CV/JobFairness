"""Run a FuxiCTR model on FairJob with aligned diagnostic artifacts.

The training path mirrors ``experiment/run_expid.py``. FairJob-specific work is
kept in this extension: scientific protocol labels, strict prediction export,
run manifests, and optional sharded representation export. ``--dry_run`` builds
the data/model stack and checks one forward batch without taking an optimizer
step, so server readiness can be established before long training starts.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr.features import FeatureMap
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.utils import load_config, print_to_json, set_logger

import model_zoo
from fuxictr_ext.fairjob.hparams import write_hparams
from fuxictr_ext.fairjob.evaluator import evaluate_prediction_file, write_evaluation_outputs
from fuxictr_ext.fairjob.models import FAIRJOB_MODELS, resolve_model_class
from fuxictr_ext.fairjob.prediction_io import write_prediction_csv
from fuxictr_ext.fairjob.representation_io import (
    export_representations,
    validate_exported_predictions,
)
from fuxictr_ext.fairjob.run_manifest import (
    git_commit,
    runtime_versions,
    stable_hash,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", default="configs/fairjob")
    parser.add_argument("--expid", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--prediction_out", default=None)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hparams_out", default=None)
    parser.add_argument("--metrics_out", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--representation_out", default=None)
    parser.add_argument(
        "--representation_splits",
        default="train,valid,test",
        help="Comma-separated subset of train,valid,test.",
    )
    parser.add_argument("--representation_shard_rows", type=int, default=10000)
    parser.add_argument("--representation_max_rows_per_split", type=int, default=None)
    parser.add_argument(
        "--representation_seed",
        type=int,
        default=None,
        help="Sampling seed for bounded representation exports; defaults to model seed.",
    )
    parser.add_argument(
        "--export_representations_only",
        action="store_true",
        help="Load the best checkpoint and export representations without retraining.",
    )
    parser.add_argument(
        "--reference_prediction",
        default=None,
        help="Canonical test prediction used to verify checkpoint-only exports.",
    )
    return parser.parse_args()


def infer_model_regime_mode(expid: str, params: dict) -> tuple[str, str, str, str]:
    """Return model, scientific regime, mode, and task protocol labels."""

    model = expid.split("_", 1)[0]
    regime = params.get("fairjob_regime_name") or params.get("fairjob_regime")
    mode = params.get("fairjob_mode")
    protocol = params.get("fairjob_protocol_name") or params.get(
        "fairjob_protocol", "legacy-integration"
    )
    if not regime:
        regime = "proxy-included" if "_unfair_" in expid else "proxy-excluded"
    if regime == "unaware":
        regime = "proxy-excluded"
    elif regime == "unfair":
        regime = "proxy-included"
    if not mode:
        mode = "smoke" if expid.endswith("_smoke") else "full"
    return model, regime, mode, protocol


def model_class_for_name(model_name: str):
    """Resolve extension adapters first, then native model-zoo classes."""

    if model_name in FAIRJOB_MODELS:
        return resolve_model_class(model_name)
    return getattr(model_zoo, model_name)


def configure_paths(args: argparse.Namespace, params: dict) -> tuple[Path | None, str | None]:
    """Apply external workdir defaults without changing checked-in configs."""

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if args.model_root:
        params["model_root"] = args.model_root
    elif run_dir:
        params["model_root"] = str(run_dir / "checkpoints")

    prediction_out = args.prediction_out
    if prediction_out is None and run_dir and not args.dry_run:
        prediction_out = str(run_dir / "predictions" / f"{args.expid}.csv")
    return run_dir, prediction_out


def validate_forward_adapter(model, batch_data) -> dict:
    """Check that representation export reproduces native model probabilities."""

    if not hasattr(model, "forward_with_representations"):
        output = model(batch_data)["y_pred"].detach().cpu().numpy()
        return {"rows": int(len(output)), "representation_shapes": {}}

    was_training = model.training
    model.eval()
    native = model(batch_data)["y_pred"].detach().cpu().numpy()
    diagnostic = model.forward_with_representations(batch_data)
    adapted = diagnostic["y_pred"].detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(native - adapted)))
    if max_abs_diff > 1e-6:
        raise ValueError(
            f"Representation adapter changed predictions: max_abs_diff={max_abs_diff}"
        )
    shapes = {
        name: list(value.shape)
        for name, value in diagnostic["representations"].items()
    }
    model.train(was_training)
    return {
        "rows": int(len(native)),
        "max_abs_prediction_diff": max_abs_diff,
        "representation_shapes": shapes,
    }


def export_requested_representations(model, feature_map, params: dict, args) -> dict:
    """Export declared splits using non-shuffled iterators and raw metadata."""

    if not args.representation_out:
        return {}
    if not hasattr(model, "forward_with_representations"):
        raise ValueError("The selected model does not expose diagnostic representations.")
    requested = {name.strip() for name in args.representation_splits.split(",") if name.strip()}
    if not requested.issubset({"train", "valid", "test"}):
        raise ValueError(f"Unknown representation splits: {sorted(requested)}")
    meta_paths = params.get("fairjob_split_meta_paths")
    if not meta_paths:
        raise ValueError("Dataset config does not declare fairjob_split_meta_paths.")

    export_params = dict(params)
    export_params["shuffle"] = False
    train_gen, valid_gen, test_gen = RankDataLoader(
        feature_map, stage="both", **export_params
    ).make_iterator()
    generators = {"train": train_gen, "valid": valid_gen, "test": test_gen}
    # Model initialization and bounded export sampling are separate sources of
    # randomness. A fixed export seed keeps cross-seed representation probes on
    # identical rows while preserving independent model-training seeds.
    representation_seed = (
        args.representation_seed
        if args.representation_seed is not None
        else params["seed"]
    )
    manifests = {}
    for split in ("train", "valid", "test"):
        if split in requested:
            manifests[split] = export_representations(
                model=model,
                data_generator=generators[split],
                meta_path=meta_paths[split],
                out_dir=args.representation_out,
                split=split,
                shard_rows=args.representation_shard_rows,
                max_rows=args.representation_max_rows_per_split,
                sample_seed=representation_seed
                + {"train": 0, "valid": 1, "test": 2}[split],
            )
    return manifests


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    params = load_config(args.config_dir, args.expid)
    if params["dataset_id"] != args.dataset_id:
        raise ValueError(
            f"Config dataset_id={params['dataset_id']} does not match "
            f"--dataset_id={args.dataset_id}"
        )
    params["gpu"] = args.gpu
    if args.seed is not None:
        params["seed"] = args.seed
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.export_representations_only and not args.representation_out:
        raise ValueError("--export_representations_only requires --representation_out.")
    if args.export_representations_only and args.dry_run:
        raise ValueError("Representation-only export cannot be combined with --dry_run.")
    if args.reference_prediction and not args.export_representations_only:
        raise ValueError("--reference_prediction requires --export_representations_only.")
    run_dir, prediction_out = configure_paths(args, params)

    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params["seed"])

    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
        feature_encoder, **params
    )
    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = model_class_for_name(params["model"])
    model = model_class(feature_map, **params)
    model.count_parameters()
    train_gen, valid_gen = RankDataLoader(
        feature_map, stage="train", **params
    ).make_iterator()

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "git_commit": git_commit(PROJECT_ROOT),
        "expid": args.expid,
        "dataset_id": args.dataset_id,
        "config_hash": stable_hash(params),
        "runtime": runtime_versions(),
        "seed": params["seed"],
        "gpu": args.gpu,
    }
    manifest_name = (
        "representation_export_manifest.json"
        if args.export_representations_only
        else "run_manifest.json"
    )
    manifest_path = run_dir / manifest_name if run_dir else None
    if manifest_path:
        write_run_manifest(manifest_path, manifest)

    first_batch = next(iter(train_gen))
    if args.export_representations_only:
        if not Path(model.checkpoint).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model.checkpoint}")
        model.load_weights(model.checkpoint)
    forward_check = validate_forward_adapter(model, first_batch)
    logging.info("Forward adapter check: " + print_to_json(forward_check))
    if args.dry_run:
        manifest.update({"status": "dry_run_complete", "forward_check": forward_check})
        if manifest_path:
            write_run_manifest(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.export_representations_only:
        del train_gen, valid_gen
        gc.collect()
        representation_manifests = export_requested_representations(
            model, feature_map, params, args
        )
        prediction_check = None
        if args.reference_prediction:
            if "test" not in representation_manifests:
                raise ValueError("Reference prediction validation requires test export.")
            prediction_check = validate_exported_predictions(
                Path(args.representation_out) / "test",
                args.reference_prediction,
            )
        manifest.update(
            {
                "status": "representation_export_complete",
                "checkpoint": model.checkpoint,
                "representation_seed": (
                    args.representation_seed
                    if args.representation_seed is not None
                    else params["seed"]
                ),
                "representation_splits": {
                    split: item["rows"]
                    for split, item in representation_manifests.items()
                },
                "reference_prediction_check": prediction_check,
            }
        )
        if manifest_path:
            write_run_manifest(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if not prediction_out:
        raise ValueError("Training runs require --prediction_out or --run_dir.")
    model.fit(train_gen, validation_data=valid_gen, **params)
    logging.info("****** Validation evaluation ******")
    model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    logging.info("******** Test prediction ********")
    test_gen = RankDataLoader(feature_map, stage="test", **params).make_iterator()
    y_pred = model.predict(test_gen)
    model_name, regime, mode, protocol = infer_model_regime_mode(args.expid, params)
    hparams_source = (
        "stage1_1_m5a_fixed"
        if params.get("mitigation_method")
        else "stage1_1_fixed_screening"
    )
    meta_path = params["fairjob_meta_path"]
    write_prediction_csv(
        out_path=prediction_out,
        meta_path=meta_path,
        y_pred=y_pred,
        model=model_name,
        regime=regime,
        mode=mode,
        expid=args.expid,
        hparams_source=hparams_source,
        seed=params.get("seed"),
        protocol=protocol,
    )
    metrics_out = args.metrics_out or (
        str(run_dir / "metrics.json")
        if run_dir
        else f"results/fairjob/metrics/{args.expid}.json"
    )
    evaluation = evaluate_prediction_file(prediction_out, meta_path)
    write_evaluation_outputs(evaluation, metrics_out)

    hparams_out = args.hparams_out or (
        str(run_dir / "hparams.json")
        if run_dir
        else f"results/fairjob/hparams/{args.expid}.json"
    )
    tracked_hparams = {
        key: params.get(key)
        for key in (
            "learning_rate",
            "batch_size",
            "epochs",
            "optimizer",
            "embedding_dim",
            "hidden_units",
            "model_structure",
            "parallel_dnn_hidden_units",
            "num_cross_layers",
            "net_dropout",
            "embedding_regularizer",
            "net_regularizer",
            "mitigation_method",
            "suppression_fraction",
            "suppression_seed_offset",
            "selective_indices_config",
            "fairness_weight",
            "adversarial_weight",
            "adversary_hidden_units",
            "gradient_reversal_scale",
        )
        if key in params
    }
    write_hparams(
        hparams_out,
        tracked_hparams,
        hparams_source,
        extra={
            "expid": args.expid,
            "dataset_id": args.dataset_id,
            "protocol": protocol,
            "regime": regime,
            "git_commit": manifest["git_commit"],
        },
    )
    representation_manifests = export_requested_representations(
        model, feature_map, params, args
    )
    manifest.update(
        {
            "status": "complete",
            "prediction_out": prediction_out,
            "hparams_out": hparams_out,
            "metrics_out": metrics_out,
            "representation_splits": {
                split: item["rows"] for split, item in representation_manifests.items()
            },
        }
    )
    if manifest_path:
        write_run_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
