"""Aggregate converged Stage1.2 probes and classify leakage migration."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.run_manifest import git_commit, write_run_manifest


UPSTREAM_REPRESENTATIONS = (
    "embedding_flat",
    "cross_layer_0",
    "cross_layer_1",
    "cross_layer_2",
    "dcnv2_final_pre_mitigation",
)
FINAL_REPRESENTATION = "dcnv2_final"
MATERIAL_EFFECT = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_dir", required=True)
    parser.add_argument("--linear_retry_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _oriented_test_auc(probe: dict) -> float:
    """Orient test AUC using validation direction, never test labels."""

    test_auc = float(probe["test_auc"])
    return test_auc if float(probe["valid_auc"]) >= 0.5 else 1.0 - test_auc


def load_results(full_dir: str | Path, retry_dir: str | Path) -> list[dict]:
    """Load 180 complete results and replace only nonconverged linear probes."""

    full_dir = Path(full_dir)
    retry_dir = Path(retry_dir)
    retry = {}
    for path in retry_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        retry[path.stem] = next(
            probe for probe in payload["probes"] if probe["family"] == "linear"
        )
    rows = []
    for path in sorted(full_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        job, representation = path.stem.split("__", 1)
        method, seed_text = job.rsplit("_seed", 1)
        for original_probe in payload["probes"]:
            probe = original_probe
            source = "initial"
            if probe["family"] == "linear" and not probe["converged"]:
                if path.stem not in retry:
                    raise ValueError(f"Missing linear retry for {path.stem}.")
                probe = retry[path.stem]
                source = "linear_retry"
            if not probe["converged"]:
                raise ValueError(f"Probe remains unconverged: {path.stem}")
            rows.append(
                {
                    "job": job,
                    "method": method,
                    "seed": int(seed_text),
                    "representation": representation,
                    "family": probe["family"],
                    "valid_auc": float(probe["valid_auc"]),
                    "test_auc": float(probe["test_auc"]),
                    "oriented_test_auc": _oriented_test_auc(probe),
                    "n_iter": int(probe["n_iter"]),
                    "source": source,
                }
            )
    if len(rows) != 360:
        raise ValueError(f"Expected 360 family-level probe rows, found {len(rows)}.")
    if len(retry) != 68:
        raise ValueError(f"Expected 68 linear retry results, found {len(retry)}.")
    return rows


def _summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
        "negative_seeds": sum(value < 0 for value in values),
        "positive_seeds": sum(value > 0 for value in values),
    }


def _classify(method_deltas: dict[tuple[str, str], dict]) -> tuple[str, list[dict]]:
    """Apply the frozen exploratory 0.005 effect rule to one method."""

    final = {
        family: method_deltas[(FINAL_REPRESENTATION, family)]
        for family in ("linear", "nonlinear")
    }
    final_reduced = all(
        item["mean"] <= -MATERIAL_EFFECT and item["negative_seeds"] == 3
        for item in final.values()
    )
    migration = []
    for representation in UPSTREAM_REPRESENTATIONS:
        for family in ("linear", "nonlinear"):
            item = method_deltas[(representation, family)]
            if item["mean"] >= MATERIAL_EFFECT and item["positive_seeds"] == 3:
                migration.append(
                    {
                        "representation": representation,
                        "family": family,
                        "mean_delta": item["mean"],
                    }
                )
    if final_reduced and migration:
        return "redistributed", migration
    if final_reduced:
        return "leakage_reduced", migration
    final_means = [item["mean"] for item in final.values()]
    if min(final_means) < 0 < max(final_means):
        return "inconclusive_probe_family_disagreement", migration
    return "not_reduced", migration


def aggregate(rows: list[dict]) -> dict:
    """Compute same-seed baseline deltas and method classifications."""

    baseline = {
        (row["seed"], row["representation"], row["family"]): row[
            "oriented_test_auc"
        ]
        for row in rows
        if row["method"] == "baseline"
    }
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] == "baseline":
            continue
        key = (row["method"], row["representation"], row["family"])
        baseline_auc = baseline[(row["seed"], row["representation"], row["family"])]
        grouped[key].append((row["seed"], row["oriented_test_auc"] - baseline_auc))

    methods = {}
    for method in sorted({key[0] for key in grouped}):
        deltas = {}
        for (current_method, representation, family), seed_values in grouped.items():
            if current_method != method:
                continue
            seed_values = sorted(seed_values)
            deltas[(representation, family)] = _summary(
                [value for _, value in seed_values]
            )
        classification, migration = _classify(deltas)
        methods[method] = {
            "classification": classification,
            "migration_evidence": migration,
            "deltas": {
                f"{representation}:{family}": item
                for (representation, family), item in sorted(deltas.items())
            },
        }

    selective_vs_random = {}
    indexed = {
        (row["method"], row["seed"], row["representation"], row["family"]): row[
            "oriented_test_auc"
        ]
        for row in rows
    }
    for representation in ("dcnv2_final_pre_mitigation", "dcnv2_final"):
        for family in ("linear", "nonlinear"):
            values = [
                indexed[("selective_suppression", seed, representation, family)]
                - indexed[
                    ("matched_random_suppression", seed, representation, family)
                ]
                for seed in (2019, 2020, 2021)
            ]
            selective_vs_random[f"{representation}:{family}"] = _summary(values)
    return {"methods": methods, "selective_vs_matched_random": selective_vs_random}


def build_payload(full_dir: str | Path, retry_dir: str | Path) -> dict:
    rows = load_results(full_dir, retry_dir)
    summary = aggregate(rows)
    return {
        "version": 1,
        "stage": "S12-M2",
        "git_commit": git_commit(PROJECT_ROOT),
        "full_result_count": 180,
        "linear_retry_count": 68,
        "family_result_count": len(rows),
        "all_probes_converged": True,
        "material_effect_threshold": MATERIAL_EFFECT,
        "auc_orientation": "validation_direction_only",
        "summary": summary,
        "rows": rows,
    }


def render_report(payload: dict) -> str:
    methods = payload["summary"]["methods"]
    lines = [
        "# S12-M2 post-intervention leakage and migration report",
        "",
        "## Protocol receipt",
        "",
        "- Completed checkpoint-layer combinations: `180/180`.",
        "- Final family-level results: `360`; all converged.",
        "- Linear retries: `68`; only initial nonconverged results were replaced.",
        "- Frozen rows per train/valid/test split: `50000/50000/50000`.",
        "- Test AUC direction is oriented only by validation AUC direction.",
        "- Exploratory material-effect threshold: absolute AUC delta `0.005`.",
        "- Deltas compare each method with the same-seed baseline at the same layer and probe family.",
        "",
        "## Method classification",
        "",
        "| Method | Classification | Linear final delta | Nonlinear final delta | Migration evidence |",
        "|---|---|---:|---:|---|",
    ]
    for method in (
        "global_suppression",
        "matched_random_suppression",
        "selective_suppression",
        "dp_regularization",
        "adversarial",
    ):
        item = methods[method]
        linear = item["deltas"]["dcnv2_final:linear"]["mean"]
        nonlinear = item["deltas"]["dcnv2_final:nonlinear"]["mean"]
        migration = ", ".join(
            f"{entry['representation']}:{entry['family']} "
            f"({entry['mean_delta']:+.6f})"
            for entry in item["migration_evidence"]
        ) or "none above threshold"
        lines.append(
            f"| {method} | `{item['classification']}` | {linear:+.6f} | "
            f"{nonlinear:+.6f} | {migration} |"
        )

    pairwise = payload["summary"]["selective_vs_matched_random"]
    lines.extend(
        [
            "",
            "## Selective versus matched-random",
            "",
            "Negative values favor lower decodability for selective suppression.",
            "",
            "| Representation | Linear delta | Nonlinear delta |",
            "|---|---:|---:|",
            f"| pre-mitigation final | {pairwise['dcnv2_final_pre_mitigation:linear']['mean']:+.6f} | "
            f"{pairwise['dcnv2_final_pre_mitigation:nonlinear']['mean']:+.6f} |",
            f"| post-mitigation final | {pairwise['dcnv2_final:linear']['mean']:+.6f} | "
            f"{pairwise['dcnv2_final:nonlinear']['mean']:+.6f} |",
            "",
            "Selective suppression lowers post-mitigation final leakage in both probe families and all three seeds. "
            "However, its nonlinear pre-mitigation final and cross-layer-2 leakage increase in all three seeds. "
            "The evidence therefore supports local gate effectiveness plus leakage redistribution, not complete removal.",
            "",
            "Global suppression also lowers final leakage in both families but has an upstream linear increase at cross-layer 2. "
            "Matched-random suppression has probe-family disagreement. DP regularization does not materially reduce final leakage. "
            "Adversarial training lowers final leakage while increasing embedding leakage; prediction collapse is evaluated in S12-M3.",
            "",
            "## M2 gate",
            "",
            "S12-M2 passes as a diagnostic audit. Current local interventions can change final-layer decodability, but no method demonstrates clean, path-wide leakage removal. "
            "This result permits S12-M3 Tier 1 collapse and calibration attribution. It does not approve new long training, five seeds, DeepFM expansion, or Stage2 direction E.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = build_payload(args.full_dir, args.linear_retry_dir)
    write_run_manifest(args.out, payload)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "all_probes_converged": payload["all_probes_converged"],
                "classifications": {
                    method: item["classification"]
                    for method, item in payload["summary"]["methods"].items()
                },
                "out": args.out,
                "report": args.report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
