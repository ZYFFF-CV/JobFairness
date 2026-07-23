"""Diagnose prediction collapse and validation-only calibration in Stage1.2."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.metrics_extended import compute_extended_metrics
from fuxictr_ext.fairjob.prediction_io import check_prediction_alignment, read_meta
from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


EPS = 1e-7
CALIBRATORS = ("identity", "temperature", "platt", "beta", "isotonic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/fairjob/stage1_2_matrix.yaml")
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _clip(probability) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)


def _logit(probability) -> np.ndarray:
    probability = _clip(probability)
    return np.log(probability) - np.log1p(-probability)


def _sigmoid(logit) -> np.ndarray:
    logit = np.asarray(logit, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


def _nllh(y_true, probability) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probability = _clip(probability)
    return float(
        -np.mean(y_true * np.log(probability) + (1.0 - y_true) * np.log1p(-probability))
    )


def fit_calibrator(name: str, probability, y_true):
    """Fit one group-blind calibrator on validation rows only."""

    probability = _clip(probability)
    y_true = np.asarray(y_true, dtype=int)
    logit = _logit(probability)
    if name == "identity":
        return lambda values: _clip(values)
    if name == "temperature":
        result = minimize_scalar(
            lambda log_temperature: _nllh(
                y_true, _sigmoid(logit / np.exp(log_temperature))
            ),
            bounds=(-4.0, 4.0),
            method="bounded",
        )
        temperature = float(np.exp(result.x))
        return lambda values: _sigmoid(_logit(values) / temperature)
    if name == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probability, y_true)
        return lambda values: _clip(model.predict(_clip(values)))
    if name == "platt":
        features = logit.reshape(-1, 1)
    elif name == "beta":
        features = np.column_stack((np.log(probability), -np.log1p(-probability)))
    else:
        raise ValueError(f"Unknown calibrator: {name}")
    model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    model.fit(features, y_true)

    def predict(values):
        values = _clip(values)
        if name == "platt":
            transformed = _logit(values).reshape(-1, 1)
        else:
            transformed = np.column_stack(
                (np.log(values), -np.log1p(-values))
            )
        return _clip(model.predict_proba(transformed)[:, 1])

    return predict


def select_calibrator(probability, y_true, seed: int) -> dict:
    """Select on a held-out validation half, then refit on all validation rows."""

    probability = _clip(probability)
    y_true = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(y_true))
    midpoint = len(permutation) // 2
    fit_indices = permutation[:midpoint]
    select_indices = permutation[midpoint:]
    candidates = {}
    for name in CALIBRATORS:
        model = fit_calibrator(name, probability[fit_indices], y_true[fit_indices])
        candidates[name] = _nllh(
            y_true[select_indices], model(probability[select_indices])
        )
    selected = min(CALIBRATORS, key=lambda name: (candidates[name], CALIBRATORS.index(name)))
    return {
        "selected": selected,
        "selection_nllh": candidates,
        "fit_rows": int(len(fit_indices)),
        "selection_rows": int(len(select_indices)),
        "model": fit_calibrator(selected, probability, y_true),
    }


def load_validation_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned validation labels and probabilities from representation shards."""

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    labels = []
    predictions = []
    for shard_info in manifest["shards"]:
        with np.load(path / shard_info["path"]) as shard:
            labels.append(np.asarray(shard["y_true"], dtype=int))
            predictions.append(np.asarray(shard["y_pred"], dtype=float))
    return np.concatenate(labels), np.concatenate(predictions)


def score_distribution(y_true, probability) -> dict:
    """Summarize score/logit range and positive-negative separation."""

    y_true = np.asarray(y_true, dtype=int)
    probability = _clip(probability)
    logit = _logit(probability)
    quantiles = (0.01, 0.05, 0.5, 0.95, 0.99)
    entropy = -probability * np.log(probability) - (1.0 - probability) * np.log1p(
        -probability
    )
    positive_mean = float(probability[y_true == 1].mean())
    negative_mean = float(probability[y_true == 0].mean())
    return {
        "prediction_mean": float(probability.mean()),
        "prediction_std": float(probability.std(ddof=1)),
        "prediction_quantiles": {
            str(value): float(np.quantile(probability, value)) for value in quantiles
        },
        "prediction_dynamic_range_99_01": float(
            np.quantile(probability, 0.99) - np.quantile(probability, 0.01)
        ),
        "logit_mean": float(logit.mean()),
        "logit_std": float(logit.std(ddof=1)),
        "mean_binary_entropy": float(entropy.mean()),
        "positive_prediction_mean": positive_mean,
        "negative_prediction_mean": negative_mean,
        "class_score_gap": positive_mean - negative_mean,
    }


def build_diagnostics(matrix_path: str | Path, seed: int = 2019) -> dict:
    """Evaluate raw and validation-calibrated predictions for all 18 runs."""

    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    meta_path = Path(matrix["data_dir"]) / "test_meta.csv"
    meta = read_meta(meta_path)
    source_root = Path(matrix["checkpoint_workdir_root"]) / "training"
    export_root = Path(matrix["workdir_root"]) / "representation_exports"
    runs = []
    for job in matrix["jobs"]:
        prediction_path = (
            source_root / job["name"] / "predictions" / f"{job['expid']}.csv"
        )
        alignment = check_prediction_alignment(prediction_path, meta_path)
        prediction = pd.read_csv(
            prediction_path, usecols=["row_id", "y_true", "y_pred"]
        )
        y_true = prediction["y_true"].to_numpy(dtype=int)
        raw_probability = prediction["y_pred"].to_numpy(dtype=float)
        validation_y, validation_probability = load_validation_predictions(
            export_root / job["name"] / "representations" / "valid"
        )
        calibration = select_calibrator(
            validation_probability, validation_y, seed=seed
        )
        calibrated_probability = calibration.pop("model")(raw_probability)
        raw_frame = meta.copy()
        raw_frame["y_true"] = y_true
        raw_frame["y_pred"] = raw_probability
        calibrated_frame = meta.copy()
        calibrated_frame["y_true"] = y_true
        calibrated_frame["y_pred"] = calibrated_probability
        runs.append(
            {
                "job": job["name"],
                "method": job["name"].rsplit("_seed", 1)[0],
                "seed": job["seed"],
                "prediction_path": str(prediction_path),
                "alignment": alignment,
                "distribution": score_distribution(y_true, raw_probability),
                "raw_metrics": compute_extended_metrics(raw_frame),
                "calibration": calibration,
                "calibrated_metrics": compute_extended_metrics(calibrated_frame),
            }
        )

    baseline = {run["seed"]: run for run in runs if run["method"] == "baseline"}
    for run in runs:
        reference = baseline[run["seed"]]
        run["relative_to_baseline"] = {
            "AUC_delta": run["raw_metrics"]["overall_AUC"]
            - reference["raw_metrics"]["overall_AUC"],
            "NLLH_delta": run["raw_metrics"]["overall_NLLH"]
            - reference["raw_metrics"]["overall_NLLH"],
            "prediction_std_ratio": run["distribution"]["prediction_std"]
            / reference["distribution"]["prediction_std"],
            "logit_std_ratio": run["distribution"]["logit_std"]
            / reference["distribution"]["logit_std"],
            "class_score_gap_ratio": run["distribution"]["class_score_gap"]
            / reference["distribution"]["class_score_gap"],
        }
        relative = run["relative_to_baseline"]
        run["prediction_collapse"] = bool(
            relative["AUC_delta"] <= -0.1
            or (
                relative["prediction_std_ratio"] < 0.25
                and relative["class_score_gap_ratio"] < 0.25
            )
        )
    return {
        "version": 1,
        "stage": "S12-M3-tier1",
        "git_commit": git_commit(PROJECT_ROOT),
        "calibration_protocol": {
            "candidates": list(CALIBRATORS),
            "selection": "validation_random_half_fit_half_select_then_refit_all_validation",
            "seed": seed,
            "test_used_for_selection": False,
        },
        "position_corrected": "unavailable_missing_external_propensity",
        "runs": runs,
    }


def _aggregate(payload: dict) -> dict:
    methods = {}
    for method in sorted({run["method"] for run in payload["runs"]}):
        runs = [run for run in payload["runs"] if run["method"] == method]
        methods[method] = {
            "collapse_seeds": sum(run["prediction_collapse"] for run in runs),
            "AUC_mean": statistics.mean(
                run["raw_metrics"]["overall_AUC"] for run in runs
            ),
            "NLLH_mean": statistics.mean(
                run["raw_metrics"]["overall_NLLH"] for run in runs
            ),
            "ECE_mean": statistics.mean(
                run["raw_metrics"]["overall_ECE"] for run in runs
            ),
            "prediction_std_mean": statistics.mean(
                run["distribution"]["prediction_std"] for run in runs
            ),
            "class_score_gap_mean": statistics.mean(
                run["distribution"]["class_score_gap"] for run in runs
            ),
            "calibrated_NLLH_mean": statistics.mean(
                run["calibrated_metrics"]["overall_NLLH"] for run in runs
            ),
            "calibrated_ECE_mean": statistics.mean(
                run["calibrated_metrics"]["overall_ECE"] for run in runs
            ),
            "selected_calibrators": [run["calibration"]["selected"] for run in runs],
        }
    return methods


def render_report(payload: dict) -> str:
    methods = _aggregate(payload)
    lines = [
        "# S12-M3 Tier 1 regularization, calibration, and collapse report",
        "",
        "- No CTR model was retrained.",
        "- Calibrators were fit and selected only within frozen validation rows.",
        "- Test predictions were used only for final diagnostics.",
        "- Position-corrected remains `unavailable_missing_external_propensity`.",
        "",
        "| Method | Collapse seeds | AUC | Raw NLLH | Calibrated NLLH | Raw ECE | Calibrated ECE | Score std | Class gap | Selected calibrators |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method in (
        "baseline",
        "global_suppression",
        "matched_random_suppression",
        "selective_suppression",
        "dp_regularization",
        "adversarial",
    ):
        item = methods[method]
        lines.append(
            f"| {method} | {item['collapse_seeds']}/3 | {item['AUC_mean']:.6f} | "
            f"{item['NLLH_mean']:.6f} | {item['calibrated_NLLH_mean']:.6f} | "
            f"{item['ECE_mean']:.6f} | {item['calibrated_ECE_mean']:.6f} | "
            f"{item['prediction_std_mean']:.6f} | {item['class_score_gap_mean']:.6f} | "
            f"{', '.join(item['selected_calibrators'])} |"
        )
    lines.extend(
        [
            "",
            "Adversarial results are classified as fairness-by-collapse when all three seeds satisfy the frozen collapse rule. "
            "Calibration changes probability quality but cannot restore lost ranking AUC because every candidate is group-blind and fitted after the frozen CTR model.",
            "",
            "Suppression NLLH/ECE gains that are matched by validation-only calibration are attributed to generic probability calibration rather than an independent fairness mechanism. "
            "Tier 2 regularization training remains conditional on this report and the M2 leakage classifications.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = build_diagnostics(args.matrix, seed=args.seed)
    write_run_manifest(args.out, payload)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "collapse": {
                    method: item["collapse_seeds"]
                    for method, item in _aggregate(payload).items()
                },
                "out": args.out,
                "report": args.report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
