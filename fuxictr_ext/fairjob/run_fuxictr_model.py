"""Run a FuxiCTR model on FairJob and export aligned predictions.

This wrapper mirrors the native FuxiCTR ``experiment/run_expid.py`` flow while
adding FairJob-specific prediction export. Keeping this as an extension avoids
changing FuxiCTR core training code just to satisfy FairJob row-alignment and
reporting requirements.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

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
from fuxictr_ext.fairjob.prediction_io import write_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", default="configs/fairjob")
    parser.add_argument("--expid", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--prediction_out", required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--hparams_out", default=None)
    parser.add_argument(
        "--model_root",
        default=None,
        help=(
            "Optional checkpoint/log root override. Use this to keep large "
            "runtime artifacts outside the source checkout."
        ),
    )
    return parser.parse_args()


def infer_model_regime_mode(expid: str, params: dict) -> tuple[str, str, str]:
    """Infer report labels from config metadata with expid as a fallback."""

    model = expid.split("_", 1)[0]
    regime = params.get("fairjob_regime")
    mode = params.get("fairjob_mode")
    if not regime:
        regime = "unfair" if "_unfair_" in expid else "unaware"
    if not mode:
        mode = "smoke" if expid.endswith("_smoke") else "full"
    return model, regime, mode


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    params = load_config(args.config_dir, args.expid)
    if params["dataset_id"] != args.dataset_id:
        raise ValueError(
            f"Config dataset_id={params['dataset_id']} does not match --dataset_id={args.dataset_id}"
        )
    params["gpu"] = args.gpu
    # FuxiCTR uses model_root for both checkpoints and experiment logs. An
    # explicit runtime override lets shared servers isolate those artifacts
    # without changing the checked-in model configuration.
    if args.model_root:
        params["model_root"] = args.model_root
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params["seed"])

    # Reuse FuxiCTR's standard preprocessing path so LR smoke/integration runs
    # exercise the same FeatureProcessor, feature_map, and dataloader stack as
    # normal model_zoo experiments.
    feature_encoder = FeatureProcessor(**params)
    params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
        feature_encoder, **params
    )

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    model.count_parameters()

    train_gen, valid_gen = RankDataLoader(feature_map, stage="train", **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info("****** Validation evaluation ******")
    model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    logging.info("******** Test prediction ********")
    test_gen = RankDataLoader(feature_map, stage="test", **params).make_iterator()
    y_pred = model.predict(test_gen)
    model_name, regime, mode = infer_model_regime_mode(args.expid, params)
    hparams_source = "manual_reference_yaml"
    meta_path = params["fairjob_meta_path"]

    # The evaluator only accepts probabilities joined to the raw meta rows by
    # row_id. Exporting here prevents later scripts from relying on tokenized IDs
    # inside the FuxiCTR parquet files.
    write_prediction_csv(
        out_path=args.prediction_out,
        meta_path=meta_path,
        y_pred=y_pred,
        model=model_name,
        regime=regime,
        mode=mode,
        expid=args.expid,
        hparams_source=hparams_source,
        seed=params.get("seed"),
    )

    hparams_out = args.hparams_out or f"results/fairjob/hparams/{args.expid}.json"
    # FuxiCTR native LR does not expose every hyperparameter used by the official
    # FairJob PyTorch LR. Recording unsupported fields keeps reports explicit
    # about what was not mapped.
    write_hparams(
        hparams_out,
        {
            "learning_rate": params.get("learning_rate"),
            "regularizer": params.get("regularizer"),
            "batch_size": params.get("batch_size"),
            "epochs": params.get("epochs"),
            "optimizer": params.get("optimizer"),
        },
        hparams_source,
        extra={
            "expid": args.expid,
            "dataset_id": args.dataset_id,
            "unsupported_official_lr_params": [
                "emb_size",
                "scheduler_step_size",
                "scheduler_gamma",
            ],
        },
    )

    print(
        json.dumps(
            {
                "prediction_out": args.prediction_out,
                "meta_path": meta_path,
                "hparams_out": hparams_out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
