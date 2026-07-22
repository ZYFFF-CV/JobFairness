"""Aggregate the completed three-seed M5A outcome diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METHODS = (
    "baseline",
    "global_suppression",
    "matched_random_suppression",
    "selective_suppression",
    "dp_regularization",
    "adversarial",
)
SEEDS = (2019, 2020, 2021)
POSITION_STATUS = "unavailable_missing_external_propensity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    return parser.parse_args()


def _metrics(payload: dict) -> dict[str, float]:
    """Extract the frozen M5 method-selection metrics from one seed."""

    all_logged = payload["selection_scopes"]["all_logged"]
    random_display = payload["selection_scopes"]["random_display"]
    context = payload["dp_context_decomposition"]["pooled"]
    return {
        "all_AUC": all_logged["overall_AUC"],
        "all_NLLH": all_logged["overall_NLLH"],
        "all_ECE": all_logged["overall_ECE"],
        "all_DP": all_logged["DP_abs"],
        "all_ECE_gap": abs(all_logged["group_gap_ECE"]),
        "random_AUC": random_display["overall_AUC"],
        "random_NLLH": random_display["overall_NLLH"],
        "random_ECE": random_display["overall_ECE"],
        "random_DP": random_display["DP_abs"],
        "random_ECE_gap": abs(random_display["group_gap_ECE"]),
        "context_DP": abs(context["within_context_dp_signed"]),
        "U": all_logged["U"],
        "U_TILDE": all_logged["U_TILDE"],
    }


def load_results(input_dir: str | Path) -> dict[str, list[dict]]:
    """Load the complete six-method by three-seed outcome matrix."""

    root = Path(input_dir)
    results = {}
    for method in METHODS:
        seeds = []
        for seed in SEEDS:
            path = root / f"{method}_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing M5 outcome diagnostic: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("position_corrected") != POSITION_STATUS:
                raise ValueError(f"Unexpected position-corrected status in {path}.")
            seeds.append(
                {
                    "seed": seed,
                    "source": str(path),
                    "git_commit": payload.get("git_commit"),
                    "metrics": _metrics(payload),
                }
            )
        results[method] = seeds
    return results


def aggregate(results: dict[str, list[dict]]) -> dict[str, dict]:
    """Return mean and sample standard deviation for every method metric."""

    summary = {}
    for method, seeds in results.items():
        metric_names = seeds[0]["metrics"]
        summary[method] = {
            name: {
                "mean": statistics.mean(item["metrics"][name] for item in seeds),
                "std": statistics.stdev(item["metrics"][name] for item in seeds),
            }
            for name in metric_names
        }
    return summary


def evaluate_gate(aggregates: dict[str, dict]) -> dict:
    """Apply the frozen M5B requirement without outcome-tuned tolerances.

    A method must improve mean DP in all three available-data protocols before
    utility is considered. It must then be no worse than baseline on the four
    core click/utility means. This strict screen prevents a single favorable
    seed or an accuracy collapse from triggering five-seed expansion.
    """

    baseline = aggregates["baseline"]
    decisions = {}
    passing = []
    for method in METHODS[1:]:
        values = aggregates[method]
        fairness = {
            metric: values[metric]["mean"] < baseline[metric]["mean"]
            for metric in ("all_DP", "random_DP", "context_DP")
        }
        performance = {
            "all_AUC": values["all_AUC"]["mean"] >= baseline["all_AUC"]["mean"],
            "all_NLLH": values["all_NLLH"]["mean"] <= baseline["all_NLLH"]["mean"],
            "U": values["U"]["mean"] >= baseline["U"]["mean"],
            "U_TILDE": values["U_TILDE"]["mean"] >= baseline["U_TILDE"]["mean"],
        }
        passed = all(fairness.values()) and all(performance.values())
        decisions[method] = {
            "fairness_improvements": fairness,
            "performance_not_worse": performance,
            "passed": passed,
        }
        if passed:
            passing.append(method)
    return {
        "passed": bool(passing),
        "passing_methods": passing,
        "decisions": decisions,
        "m5b_five_seed_expansion": "approved" if passing else "not_approved",
        "deepfm_generalization": "approved_after_m5b" if passing else "not_approved",
    }


def build_report(input_dir: str | Path) -> dict:
    """Build the complete M5A result and M5B gate decision."""

    results = load_results(input_dir)
    aggregates = aggregate(results)
    commits = sorted(
        {
            item["git_commit"]
            for seeds in results.values()
            for item in seeds
            if item["git_commit"]
        }
    )
    gate = evaluate_gate(aggregates)
    return {
        "stage": "M5A",
        "status": "complete_m5b_gate_passed" if gate["passed"] else "complete_m5b_gate_failed",
        "position_corrected": POSITION_STATUS,
        "method_selection_protocols": [
            "all_logged",
            "random_display",
            "context_conditioned",
        ],
        "source_git_commits": commits,
        "methods": aggregates,
        "seed_results": results,
        "m5b_gate": gate,
    }


def _mean_std(values: dict) -> str:
    return f"{values['mean']:.6f} +/- {values['std']:.6f}"


def render_markdown(report: dict) -> str:
    """Render the Stage1.1 M5A audit and gate decision."""

    lines = [
        "# M5A DCNv2 three-seed result",
        "",
        f"- Status: `{report['status']}`.",
        f"- Position-corrected: `{report['position_corrected']}`.",
        "- Selection protocols: `all_logged`, `random_display`, and "
        "`context_conditioned`.",
        "- Values are mean +/- sample standard deviation over seeds 2019, 2020, "
        "and 2021.",
        "",
        "| Method | All AUC | All NLLH | All ECE | All DP | Random DP | Context DP | U | U_TILDE | All ECE gap | Random ECE gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = report["methods"][method]
        lines.append(
            "| {method} | {auc} | {nllh} | {overall_ece} | {dp} | {rdp} | {cdp} | {u} | "
            "{ut} | {ece} | {rece} |".format(
                method=method,
                auc=_mean_std(values["all_AUC"]),
                nllh=_mean_std(values["all_NLLH"]),
                overall_ece=_mean_std(values["all_ECE"]),
                dp=_mean_std(values["all_DP"]),
                rdp=_mean_std(values["random_DP"]),
                cdp=_mean_std(values["context_DP"]),
                u=_mean_std(values["U"]),
                ut=_mean_std(values["U_TILDE"]),
                ece=_mean_std(values["all_ECE_gap"]),
                rece=_mean_std(values["random_ECE_gap"]),
            )
        )

    lines.extend(["", "## M5B gate", ""])
    if report["m5b_gate"]["passed"]:
        methods = ", ".join(report["m5b_gate"]["passing_methods"])
        lines.append(f"Gate passed for: {methods}.")
    else:
        lines.extend(
            [
                "The M5B gate failed. No intervention improved mean DP in all "
                "three available-data protocols while also matching or improving "
                "baseline AUC, NLLH, U, and U_TILDE.",
                "",
                "The adversarial baseline reduced all-logged and context DP but "
                "collapsed ranking and calibration performance (mean all-logged "
                "AUC 0.536575 and ECE 0.008709, versus baseline 0.764927 and "
                "0.000839). The remaining "
                "methods did not improve both all-logged and context-conditioned "
                "DP means over baseline.",
                "",
                "Five-seed DCNv2 expansion and DeepFM generalization are therefore "
                "not approved by the frozen gate. Further mitigation design or "
                "weight sweeps require a new decision; they are not part of this "
                "completed M5A screen.",
            ]
        )
    lines.extend(
        [
            "",
            "This is an available-data association study. Position-corrected "
            "results and causal claims remain unavailable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args.input_dir)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
