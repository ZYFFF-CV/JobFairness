"""Audit Stage1.2 outcome floors and stability without retraining CTR models."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from fuxictr_ext.fairjob.build_m5_report import METHODS, POSITION_STATUS, SEEDS
from fuxictr_ext.fairjob.metrics import add_prediction_rank, utility
from fuxictr_ext.fairjob.prediction_io import check_prediction_alignment, read_meta
from fuxictr_ext.fairjob.run_manifest import write_run_manifest


QUALITY_METRICS = ("AUC", "NLLH", "Brier", "ECE")
PROTOCOLS = ("all_logged", "random_display", "context_conditioned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--calibration_bins", type=int, default=10)
    return parser.parse_args()


def _mean_std(values) -> dict:
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def calibration_curve(frame: pd.DataFrame, bins: int = 10) -> list[dict]:
    """Return fixed-width reliability bins, including empty-bin omissions."""

    if bins < 2:
        raise ValueError("Calibration curves require at least two bins.")
    probability = np.clip(frame["y_pred"].to_numpy(dtype=float), 0.0, 1.0)
    labels = frame["y_true"].to_numpy(dtype=float)
    indices = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = indices == index
        if not selected.any():
            continue
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": int(selected.sum()),
                "prediction_mean": float(probability[selected].mean()),
                "positive_rate": float(labels[selected].mean()),
            }
        )
    return rows


def context_conditioned_dp(decomposition: dict) -> dict:
    """Recover standardized group means from the frozen context weights."""

    contexts = decomposition["contexts"]
    group_0_mean = sum(
        row["reference_weight"] * row["group_0_mean"] for row in contexts
    )
    group_1_mean = sum(
        row["reference_weight"] * row["group_1_mean"] for row in contexts
    )
    signed = group_1_mean - group_0_mean
    return {
        "DP_signed": float(signed),
        "DP_abs": float(abs(signed)),
        "DP_group_0_mean": float(group_0_mean),
        "DP_group_1_mean": float(group_1_mean),
        "DP_group_0_n": int(sum(row["group_0_n"] for row in contexts)),
        "DP_group_1_n": int(sum(row["group_1_n"] for row in contexts)),
        "DP_senior_n": int(
            sum(row["group_0_n"] + row["group_1_n"] for row in contexts)
        ),
        "common_contexts": int(decomposition["common_contexts"]),
        "composition_residual_signed": float(
            decomposition["composition_residual_signed"]
        ),
    }


def _scope_dp(scope: dict) -> dict:
    keys = (
        "DP_signed",
        "DP_abs",
        "DP_group_0_mean",
        "DP_group_1_mean",
        "DP_group_0_n",
        "DP_group_1_n",
        "DP_senior_n",
    )
    return {key: scope[key] for key in keys}


def _group_quality(scope: dict) -> dict:
    groups = {
        str(group): {
            metric: scope[f"group_{group}_{metric}"]
            for metric in QUALITY_METRICS
        }
        for group in (0, 1)
    }
    worst = {
        "AUC": min(groups["0"]["AUC"], groups["1"]["AUC"]),
        "NLLH": max(groups["0"]["NLLH"], groups["1"]["NLLH"]),
        "Brier": max(groups["0"]["Brier"], groups["1"]["Brier"]),
        "ECE": max(groups["0"]["ECE"], groups["1"]["ECE"]),
    }
    gaps = {
        metric: abs(scope[f"group_gap_{metric}"]) for metric in QUALITY_METRICS
    }
    return {"groups": groups, "worst_group": worst, "absolute_gaps": gaps}


def _score_distance(frame: pd.DataFrame) -> dict:
    senior = frame[frame["senior"].astype(int) == 1]
    group_0 = senior[senior["protected_attribute"].astype(int) == 0][
        "y_pred"
    ].to_numpy(dtype=float)
    group_1 = senior[senior["protected_attribute"].astype(int) == 1][
        "y_pred"
    ].to_numpy(dtype=float)
    pooled_std = float(np.std(np.concatenate((group_0, group_1)), ddof=1))
    distance = float(wasserstein_distance(group_0, group_1))
    return {
        "scope": "senior_jobs",
        "group_0_n": int(len(group_0)),
        "group_1_n": int(len(group_1)),
        "wasserstein": distance,
        "wasserstein_over_pooled_std": distance / pooled_std
        if pooled_std > 0
        else None,
        "ks_statistic": float(ks_2samp(group_0, group_1).statistic),
    }


def _ranking_diagnostics(frame: pd.DataFrame) -> dict:
    """Measure ranking gaps while exposing the available cross-group support."""

    random_display = frame[frame["displayrandom"].astype(int) == 1].copy()
    proxy_counts = random_display.groupby("impression_id")[
        "protected_attribute"
    ].nunique()
    mixed_ids = proxy_counts[proxy_counts > 1].index
    ranked = add_prediction_rank(random_display)
    impression_size = ranked.groupby("impression_id")["y_pred"].transform("size")
    ranked["normalized_pred_rank"] = ranked["pred_rank"] / impression_size
    mixed = ranked[ranked["impression_id"].isin(mixed_ids)]
    if mixed.empty:
        within_impression = {
            "status": "not_identifiable_no_mixed_impressions",
            "metric": "mean_normalized_pred_rank_group_1_minus_group_0",
            "signed_gap": None,
            "gap_std": None,
        }
    else:
        means = mixed.groupby(
            ["impression_id", "protected_attribute"]
        )["normalized_pred_rank"].mean().unstack()
        gaps = means[1] - means[0]
        within_impression = {
            "status": "available_limited_support",
            "metric": "mean_normalized_pred_rank_group_1_minus_group_0",
            "signed_gap": float(gaps.mean()),
            "gap_std": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
        }
    group_utility = {
        str(group): utility(
            frame[frame["protected_attribute"].astype(int) == group]
        )
        for group in (0, 1)
    }
    return {
        "random_display_impressions": int(
            random_display["impression_id"].nunique()
        ),
        "mixed_proxy_impressions": int(len(mixed_ids)),
        "within_impression_proxy_rank_gap": within_impression,
        "group_stratified_U": group_utility,
        "group_stratified_U_signed_gap": float(
            group_utility["1"] - group_utility["0"]
        ),
    }


def prediction_diagnostics(
    payload: dict, meta: pd.DataFrame, calibration_bins: int
) -> dict:
    """Compute distribution and ranking diagnostics absent from M5 JSON files."""

    prediction_path = Path(payload["prediction_path"])
    meta_path = Path(payload["meta_path"])
    alignment = check_prediction_alignment(prediction_path, meta_path)
    prediction = pd.read_csv(
        prediction_path, usecols=["row_id", "y_true", "y_pred"]
    )
    frame = meta.copy()
    frame["y_true"] = prediction["y_true"].to_numpy(dtype=float)
    frame["y_pred"] = prediction["y_pred"].to_numpy(dtype=float)
    frame["click"] = frame["click"].to_numpy(dtype=float)
    random_display = frame[frame["displayrandom"].astype(int) == 1].copy()

    return {
        "alignment": alignment,
        "score_distance": {
            "all_logged": _score_distance(frame),
            "random_display": _score_distance(random_display),
        },
        "calibration_curves": {
            scope_name: {
                str(group): calibration_curve(
                    scope[
                        scope["protected_attribute"].astype(int) == group
                    ],
                    bins=calibration_bins,
                )
                for group in (0, 1)
            }
            for scope_name, scope in (
                ("all_logged", frame),
                ("random_display", random_display),
            )
        },
        "ranking": _ranking_diagnostics(frame),
    }


def load_runs(
    input_dir: str | Path, calibration_bins: int = 10
) -> list[dict]:
    """Load all frozen M5 outcomes and add prediction-only diagnostics."""

    root = Path(input_dir)
    runs = []
    meta_cache: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        for seed in SEEDS:
            path = root / f"{method}_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing outcome diagnostic: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("position_corrected") != POSITION_STATUS:
                raise ValueError(f"Unexpected position status in {path}.")
            meta_path = payload["meta_path"]
            if meta_path not in meta_cache:
                meta_cache[meta_path] = read_meta(meta_path)
            all_logged = payload["selection_scopes"]["all_logged"]
            random_display = payload["selection_scopes"]["random_display"]
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "source": str(path),
                    "git_commit": payload.get("git_commit"),
                    "protocols": {
                        "all_logged": _scope_dp(all_logged),
                        "random_display": _scope_dp(random_display),
                        "context_conditioned": context_conditioned_dp(
                            payload["dp_context_decomposition"]["pooled"]
                        ),
                    },
                    "group_quality": {
                        "all_logged": _group_quality(all_logged),
                        "random_display": _group_quality(random_display),
                    },
                    "overall": {
                        scope_name: {
                            metric: scope[f"overall_{metric}"]
                            for metric in QUALITY_METRICS
                        }
                        for scope_name, scope in (
                            ("all_logged", all_logged),
                            ("random_display", random_display),
                        )
                    },
                    "utility": {
                        "all_logged": {
                            "U": all_logged["U"],
                            "U_TILDE": all_logged["U_TILDE"],
                        },
                        "random_display": {
                            "U": random_display["U"],
                            "U_TILDE": random_display["U_TILDE"],
                        },
                    },
                    "cluster_bootstrap": payload["cluster_bootstrap"],
                    "proxy_measurement_sensitivity": payload[
                        "proxy_measurement_sensitivity"
                    ],
                    "prediction_diagnostics": prediction_diagnostics(
                        payload, meta_cache[meta_path], calibration_bins
                    ),
                }
            )
    return runs


def _aggregate_method(runs: list[dict]) -> dict:
    methods = {}
    for method in METHODS:
        selected = [run for run in runs if run["method"] == method]
        protocols = {}
        for protocol in PROTOCOLS:
            protocols[protocol] = {
                key: _mean_std(
                    run["protocols"][protocol][key] for run in selected
                )
                for key in (
                    "DP_signed",
                    "DP_abs",
                    "DP_group_0_mean",
                    "DP_group_1_mean",
                    "DP_group_0_n",
                    "DP_group_1_n",
                )
            }
        group_quality = {}
        for protocol in ("all_logged", "random_display"):
            group_quality[protocol] = {
                "worst_group": {
                    metric: _mean_std(
                        run["group_quality"][protocol]["worst_group"][metric]
                        for run in selected
                    )
                    for metric in QUALITY_METRICS
                },
                "absolute_gaps": {
                    metric: _mean_std(
                        run["group_quality"][protocol]["absolute_gaps"][metric]
                        for run in selected
                    )
                    for metric in QUALITY_METRICS
                },
            }
        methods[method] = {
            "protocols": protocols,
            "group_quality": group_quality,
            "utility": {
                protocol: {
                    metric: _mean_std(
                        run["utility"][protocol][metric] for run in selected
                    )
                    for metric in ("U", "U_TILDE")
                }
                for protocol in ("all_logged", "random_display")
            },
            "score_distance": {
                protocol: {
                    metric: _mean_std(
                        run["prediction_diagnostics"]["score_distance"][protocol][
                            metric
                        ]
                        for run in selected
                    )
                    for metric in ("wasserstein", "wasserstein_over_pooled_std", "ks_statistic")
                }
                for protocol in ("all_logged", "random_display")
            },
            "group_stratified_U_gap": _mean_std(
                run["prediction_diagnostics"]["ranking"][
                    "group_stratified_U_signed_gap"
                ]
                for run in selected
            ),
            "within_impression_rank_gap": _mean_std(
                run["prediction_diagnostics"]["ranking"][
                    "within_impression_proxy_rank_gap"
                ]["signed_gap"]
                for run in selected
            ),
            "mixed_proxy_impressions": sorted(
                {
                    run["prediction_diagnostics"]["ranking"][
                        "mixed_proxy_impressions"
                    ]
                    for run in selected
                }
            ),
        }
    return methods


def classify_dp_floor(runs: list[dict]) -> dict:
    """Apply the frozen three-way DP role decision."""

    baseline = [run for run in runs if run["method"] == "baseline"]
    all_abs = [run["protocols"]["all_logged"]["DP_abs"] for run in baseline]
    mean_abs = statistics.mean(all_abs)
    seed_std = statistics.stdev(all_abs)
    crossing = sum(
        run["cluster_bootstrap"]["all_logged"]["metrics"]["DP_signed"]["ci_low"]
        <= 0.0
        <= run["cluster_bootstrap"]["all_logged"]["metrics"]["DP_signed"][
            "ci_high"
        ]
        for run in baseline
    )
    sign_consistent = sum(
        len(
            {
                np.sign(run["protocols"][protocol]["DP_signed"])
                for protocol in PROTOCOLS
            }
        )
        == 1
        for run in runs
    )
    near_floor = mean_abs <= seed_std and crossing > 0
    if near_floor:
        classification = "DP_near_floor_for_current_setting"
    elif seed_std >= 0.5 * mean_abs or sign_consistent < len(runs):
        classification = "DP_secondary_high_variance_metric"
    else:
        classification = "DP_informative_primary_outcome"
    return {
        "classification": classification,
        "baseline_all_DP_abs_mean": mean_abs,
        "baseline_all_DP_abs_seed_std": seed_std,
        "baseline_bootstrap_intervals_crossing_zero": crossing,
        "protocol_sign_consistent_runs": sign_consistent,
        "total_runs": len(runs),
        "rule": (
            "near_floor_if_baseline_mean_abs_not_above_seed_std_and_any_"
            "baseline_cluster_interval_crosses_zero"
        ),
    }


def _method_effects(runs: list[dict], methods: dict) -> dict:
    baseline = {
        run["seed"]: run for run in runs if run["method"] == "baseline"
    }
    baseline_noise = methods["baseline"]["protocols"]["all_logged"]["DP_abs"][
        "std"
    ]
    effects = {}
    for method in METHODS[1:]:
        selected = [run for run in runs if run["method"] == method]
        effects[method] = {}
        for protocol in PROTOCOLS:
            delta = [
                run["protocols"][protocol]["DP_abs"]
                - baseline[run["seed"]]["protocols"][protocol]["DP_abs"]
                for run in selected
            ]
            summary = _mean_std(delta)
            summary["mean_abs_effect_over_baseline_seed_std"] = (
                abs(summary["mean"]) / baseline_noise
                if baseline_noise > 0
                else None
            )
            effects[method][protocol] = summary
    return effects


def _proxy_summary(runs: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        selected = [run for run in runs if run["method"] == method]
        base = [
            run["protocols"]["all_logged"]["DP_abs"] for run in selected
        ]
        summary[method] = {}
        for mode in ("symmetric_flip", "random_missing"):
            values = [
                run["proxy_measurement_sensitivity"]["modes"][mode]["0.2"][
                    "DP_abs_mean"
                ]
                for run in selected
            ]
            ratios = [
                value / baseline if baseline > 0 else None
                for value, baseline in zip(values, base)
            ]
            summary[method][mode] = {
                "DP_abs": _mean_std(values),
                "ratio_to_unperturbed": _mean_std(ratios),
            }
    return summary


def build_audit(input_dir: str | Path, calibration_bins: int = 10) -> dict:
    runs = load_runs(input_dir, calibration_bins=calibration_bins)
    methods = _aggregate_method(runs)
    commits = sorted(
        {run["git_commit"] for run in runs if run.get("git_commit")}
    )
    return {
        "version": 1,
        "stage": "S12-M4",
        "status": "complete",
        "position_corrected": POSITION_STATUS,
        "claim_boundary": "descriptive_association_not_causal_effect",
        "source_git_commits": commits,
        "protocols": list(PROTOCOLS),
        "calibration_bins": calibration_bins,
        "dp_role": classify_dp_floor(runs),
        "methods": methods,
        "matched_seed_dp_effects": _method_effects(runs, methods),
        "proxy_sensitivity": _proxy_summary(runs),
        "runs": runs,
    }


def _format_mean_std(item: dict) -> str:
    return f"{item['mean']:.6f} +/- {item['std']:.6f}"


def render_report(audit: dict) -> str:
    lines = [
        "# S12-M4 outcome floor and stability report",
        "",
        f"- Status: `{audit['status']}`.",
        f"- DP role: `{audit['dp_role']['classification']}`.",
        f"- Position-corrected: `{audit['position_corrected']}`.",
        "- Scope: 6 methods x 3 matched seeds; no CTR model was retrained.",
        "- Intervals are the existing 20-repeat impression-cluster screening "
        "intervals, not publication-grade confidence intervals.",
        "",
        "## Protocol DP",
        "",
        "| Method | All signed | All abs | Random signed | Random abs | Context signed | Context abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = audit["methods"][method]["protocols"]
        lines.append(
            f"| {method} | {_format_mean_std(values['all_logged']['DP_signed'])} | "
            f"{_format_mean_std(values['all_logged']['DP_abs'])} | "
            f"{_format_mean_std(values['random_display']['DP_signed'])} | "
            f"{_format_mean_std(values['random_display']['DP_abs'])} | "
            f"{_format_mean_std(values['context_conditioned']['DP_signed'])} | "
            f"{_format_mean_std(values['context_conditioned']['DP_abs'])} |"
        )

    lines.extend(
        [
            "",
            "Context-conditioned group means are direct-standardized over "
            "`displayrandom x rank` common-support contexts. The machine JSON "
            "retains both standardized means and group counts.",
            "",
            "## Matched-seed DP effects",
            "",
            "Negative deltas favor the intervention. The final column compares "
            "the all-logged mean effect with baseline between-seed variation.",
            "",
            "| Method | All abs delta | Random abs delta | Context abs delta | |All effect| / baseline seed std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS[1:]:
        values = audit["matched_seed_dp_effects"][method]
        lines.append(
            f"| {method} | {_format_mean_std(values['all_logged'])} | "
            f"{_format_mean_std(values['random_display'])} | "
            f"{_format_mean_std(values['context_conditioned'])} | "
            f"{values['all_logged']['mean_abs_effect_over_baseline_seed_std']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Group quality and calibration",
            "",
            "Worst-group values and absolute group gaps are averaged over the "
            "three seeds. For AUC, lower is worse; for NLLH/Brier/ECE, higher is worse.",
            "",
            "| Method | Worst AUC | Worst NLLH | Worst Brier | Worst ECE | AUC gap | NLLH gap | Brier gap | ECE gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        values = audit["methods"][method]["group_quality"]["all_logged"]
        worst = values["worst_group"]
        gaps = values["absolute_gaps"]
        lines.append(
            f"| {method} | {worst['AUC']['mean']:.6f} | "
            f"{worst['NLLH']['mean']:.6f} | {worst['Brier']['mean']:.6f} | "
            f"{worst['ECE']['mean']:.6f} | {gaps['AUC']['mean']:.6f} | "
            f"{gaps['NLLH']['mean']:.6f} | {gaps['Brier']['mean']:.6f} | "
            f"{gaps['ECE']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The same group-wise table for `random_display`, together with "
            "fixed-bin group calibration curves for both scopes, is retained "
            "in the machine-readable audit.",
            "",
            "## Distribution and ranking",
            "",
            "| Method | All Wasserstein / std | All KS | Random Wasserstein / std | Random KS | U | U_TILDE | Within-impression rank gap | Group-stratified U gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        values = audit["methods"][method]
        distance = values["score_distance"]
        utility_values = values["utility"]["all_logged"]
        lines.append(
            f"| {method} | "
            f"{distance['all_logged']['wasserstein_over_pooled_std']['mean']:.6f} | "
            f"{distance['all_logged']['ks_statistic']['mean']:.6f} | "
            f"{distance['random_display']['wasserstein_over_pooled_std']['mean']:.6f} | "
            f"{distance['random_display']['ks_statistic']['mean']:.6f} | "
            f"{utility_values['U']['mean']:.6f} | "
            f"{utility_values['U_TILDE']['mean']:.6f} | "
            f"{values['within_impression_rank_gap']['mean']:.6f} | "
            f"{values['group_stratified_U_gap']['mean']:.6f} |"
        )
    mixed = sorted(
        {
            value
            for method in METHODS
            for value in audit["methods"][method]["mixed_proxy_impressions"]
        }
    )
    lines.extend(
        [
            "",
            f"Mixed-proxy random-display impressions observed: `{mixed}`. The "
            "within-impression value is the mean group-1 minus group-0 "
            "normalized predicted-rank gap on these candidate sets. With only "
            "78 mixed impressions, it is exploratory; group-stratified U is "
            "reported as a broader ranking-quality diagnostic.",
            "",
            "## Bootstrap and proxy sensitivity",
            "",
        ]
    )
    role = audit["dp_role"]
    lines.extend(
        [
            f"Baseline all-logged DP abs is `{role['baseline_all_DP_abs_mean']:.6f} "
            f"+/- {role['baseline_all_DP_abs_seed_std']:.6f}`. "
            f"`{role['baseline_bootstrap_intervals_crossing_zero']}/3` baseline "
            "screening intervals cross zero. "
            f"`{role['protocol_sign_consistent_runs']}/{role['total_runs']}` runs "
            "keep the same signed direction across all/random/context protocols.",
            "",
            "| Method | 20% symmetric-flip / base | 20% random-missing / base |",
            "|---|---:|---:|",
        ]
    )
    for method in METHODS:
        values = audit["proxy_sensitivity"][method]
        lines.append(
            f"| {method} | "
            f"{_format_mean_std(values['symmetric_flip']['ratio_to_unperturbed'])} | "
            f"{_format_mean_std(values['random_missing']['ratio_to_unperturbed'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{role['classification']}` is selected because baseline mean "
            "absolute DP does not exceed its between-seed standard deviation "
            "and at least one baseline cluster interval crosses zero. DP remains "
            "a required reported metric, but it is not sufficiently resolved to "
            "serve as the sole or primary Stage2 optimization target.",
            "",
            "All three available-data protocols are retained. Quantitative "
            "magnitudes vary by logging scope and seed; context conditioning does "
            "not create a stable intervention winner. Proxy flips attenuate the "
            "measured signal, while random missingness is comparatively stable. "
            "These are sensitivity diagnostics for a behavioral proxy, not claims "
            "about verified demographic attributes.",
            "",
            "Position correction remains unavailable because no external logging "
            "propensity is present. No propensity is inferred from clicks or model "
            "predictions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    audit = build_audit(args.input_dir, calibration_bins=args.calibration_bins)
    write_run_manifest(args.out, audit)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "dp_role": audit["dp_role"]["classification"],
                "out": args.out,
                "report": args.report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
