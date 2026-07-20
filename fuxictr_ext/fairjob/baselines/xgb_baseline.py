"""XGBoost baseline using FairJob splits and the shared prediction format.

XGBoost is kept as a FuxiCTR-compatible baseline rather than a model_zoo model:
it uses the same prepared splits, feature regimes, prediction CSV schema, and
evaluator as LR, but it does not depend on FuxiCTR neural-model internals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.hparams import resolve_xgb_hparams, write_hparams
from fuxictr_ext.fairjob.prediction_io import write_prediction_csv


class MeanTargetEncoder:
    """Small fallback for environments without sklearn.preprocessing.TargetEncoder.

    The fallback fits category means on train only and maps unseen categories to
    the global train mean. It is intentionally simple so missing sklearn support
    does not block smoke validation.
    """

    def fit(self, X, y):
        X = pd.DataFrame(X).astype(str)
        y = pd.Series(y, dtype=float)
        self.global_mean_ = float(y.mean())
        self.maps_ = []
        for col in X.columns:
            mapping = y.groupby(X[col]).mean().to_dict()
            self.maps_.append(mapping)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(str)
        cols = []
        for idx, col in enumerate(X.columns):
            cols.append(X[col].map(self.maps_[idx]).fillna(self.global_mean_).to_numpy())
        return np.column_stack(cols) if cols else np.empty((len(X), 0))


def make_target_encoder():
    """Prefer sklearn's binary TargetEncoder, falling back to a local encoder."""

    try:
        from sklearn.preprocessing import TargetEncoder

        return TargetEncoder(target_type="binary"), "sklearn.preprocessing.TargetEncoder"
    except Exception:
        print("Using fallback MeanTargetEncoder")
        return MeanTargetEncoder(), "fallback_mean_target_encoder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/FairJob/processed")
    parser.add_argument("--regime", choices=["unaware", "unfair"], required=True)
    parser.add_argument("--mode", choices=["smoke", "integration", "full"], required=True)
    parser.add_argument("--hparams", default="configs/fairjob/xgb_reference_hparams.yaml")
    parser.add_argument("--feature_manifest", default=None)
    parser.add_argument("--reference_repo", default=None)
    parser.add_argument("--tune_trials", type=int, default=0)
    parser.add_argument("--out_pred", required=True)
    parser.add_argument("--out_hparams", default=None)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_paths(data_dir: Path, mode: str) -> tuple[Path, Path, Path]:
    """Return the train/test/meta files for the requested validation scale."""

    if mode == "smoke":
        return (
            data_dir / "smoke_train.csv",
            data_dir / "smoke_test.csv",
            data_dir / "smoke_test_meta.csv",
        )
    return data_dir / "train_full.csv", data_dir / "test.csv", data_dir / "test_meta.csv"


def feature_names(manifest: dict, regime: str) -> tuple[list[str], list[str]]:
    """Select features according to the same unaware/unfair contract as FuxiCTR."""

    categorical = (
        manifest["categorical_id_features"]
        + manifest["categorical_features"]
        + manifest["binary_context_features"]
    )
    if regime == "unfair":
        categorical = categorical + [manifest["protected_feature_for_model"]]
    numeric = manifest["numeric_features"]
    return categorical, numeric


def build_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    categorical: list[str],
    numeric: list[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Encode train/test matrices without leaking test labels.

    TargetEncoder is fitted only on the train split. Test rows are transformed
    with that fitted encoder, preserving the same split discipline required by
    the task cards and avoiding the reference-code ambiguity around XGBoost
    encoded versus unencoded matrices.
    """

    pieces_train = []
    pieces_test = []
    encoder_name = "none"
    if categorical:
        encoder, encoder_name = make_target_encoder()
        train_cat = train_df[categorical].astype(str)
        test_cat = test_df[categorical].astype(str)
        encoder.fit(train_cat, y_train)
        pieces_train.append(encoder.transform(train_cat))
        pieces_test.append(encoder.transform(test_cat))
    if numeric:
        pieces_train.append(train_df[numeric].astype(float).fillna(0.0).to_numpy())
        pieces_test.append(test_df[numeric].astype(float).fillna(0.0).to_numpy())
    if not pieces_train:
        raise ValueError("No features selected for XGBoost.")
    return np.hstack(pieces_train), np.hstack(pieces_test), encoder_name


def fit_xgb(X_train, y_train, params: dict):
    """Fit a CPU-friendly XGBClassifier with automatic class weighting."""

    import xgboost as xgb

    params = params.copy()
    params.pop("scale_pos_weight", None)
    if "n_jobs" not in params:
        params["n_jobs"] = -1
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model, scale_pos_weight


def maybe_tune(X_train, y_train, params: dict, tune_trials: int) -> dict:
    """Optionally run a tiny Optuna search, never the paper-level 100 trials."""

    if tune_trials == 0:
        return params
    if tune_trials not in (5, 10):
        raise ValueError("Only --tune_trials 5 or 10 is allowed for this integration task.")
    import optuna
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    # Tuning is intentionally limited to a deterministic tail validation split.
    # This is a local integration aid, not the official FairJob paper search.
    def objective(trial):
        trial_params = params.copy()
        trial_params.update(
            {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 100, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
                "gamma": trial.suggest_float("gamma", 1e-3, 100, log=True),
            }
        )
        trial_params.pop("scale_pos_weight", None)
        scale_pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        model = xgb.XGBClassifier(
            **trial_params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        model.fit(X_tr, y_tr)
        pred = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=tune_trials)
    tuned = params.copy()
    tuned.update(study.best_trial.params)
    return tuned


def main() -> None:
    args = parse_args()
    mode = "smoke" if args.mode == "smoke" else "full"
    data_dir = Path(args.data_dir)
    manifest_path = Path(args.feature_manifest) if args.feature_manifest else data_dir / "feature_manifest.json"
    manifest = load_manifest(manifest_path)
    train_path, test_path, meta_path = split_paths(data_dir, mode)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_train = train_df["click"].astype(int).to_numpy()
    categorical, numeric = feature_names(manifest, args.regime)
    X_train, X_test, encoder_name = build_matrix(
        train_df, test_df, y_train, categorical, numeric
    )

    # Hyperparameter source precedence is resolved in one place so reports can
    # distinguish official CSVs, manual references, quick trials, and fallbacks.
    params, source = resolve_xgb_hparams(
        args.hparams, reference_repo=args.reference_repo, tune_trials=args.tune_trials
    )
    params = maybe_tune(X_train, y_train, params, args.tune_trials)
    model, scale_pos_weight = fit_xgb(X_train, y_train, params)
    y_pred = model.predict_proba(X_test)[:, 1]

    model_name = "XGB"
    expid = f"XGB_fairjob_{args.regime}_{'smoke' if mode == 'smoke' else 'full'}"
    # Export the exact same prediction schema as LR so evaluator/report code does
    # not need model-specific branches.
    write_prediction_csv(
        out_path=args.out_pred,
        meta_path=meta_path,
        y_pred=y_pred,
        model=model_name,
        regime=args.regime,
        mode=mode,
        expid=expid,
        hparams_source=source,
        seed=params.get("random_state"),
    )
    out_hparams = args.out_hparams or f"results/fairjob/hparams/{expid}.json"
    write_hparams(
        out_hparams,
        params,
        source,
        extra={
            "expid": expid,
            "encoder": encoder_name,
            "scale_pos_weight": scale_pos_weight,
            "n_features": int(X_train.shape[1]),
        },
    )


if __name__ == "__main__":
    main()
