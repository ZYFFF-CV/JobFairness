"""Build FairJob smoke/integration result tables and a Markdown report.

Reports are generated from saved artifacts only. This keeps validation auditable:
every table row should trace back to a prediction CSV, metrics JSON, and hparams
JSON produced by a concrete run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRIC_COLUMNS = ["model", "regime", "NLLH", "AUC", "DP", "U", "U_TILDE", "n_rows"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/fairjob")
    parser.add_argument("--processed_dir", default="data/FairJob/processed")
    parser.add_argument("--out", default="docs/fairjob/integration_validation_report.md")
    parser.add_argument("--mode", choices=["smoke", "integration", "all"], default="all")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics(results_dir: Path, mode: str) -> pd.DataFrame:
    """Collect metrics JSON files matching the requested validation mode."""

    rows = []
    mode_key = "full" if mode == "integration" else mode
    for path in sorted((results_dir / "metrics").glob("*.json")):
        payload = read_json(path)
        if payload.get("mode") == mode_key:
            rows.append(payload)
    if not rows:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return pd.DataFrame(rows)


def write_mode_outputs(results_dir: Path, mode: str) -> pd.DataFrame:
    """Write the compact CSV/JSON summary for smoke or integration mode."""

    table = collect_metrics(results_dir, mode)
    table_path = results_dir / f"{mode}_table.csv"
    report_path = results_dir / f"{mode}_report.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if not table.empty:
        table[METRIC_COLUMNS].to_csv(table_path, index=False)
    else:
        table.to_csv(table_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "n_rows": int(len(table)),
                "table_path": str(table_path),
                "complete_four_runs": bool(len(table) == 4),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")
    return table


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small Markdown table without depending on optional tabulate."""

    if df.empty:
        return "No rows available.\n"
    rows = df[METRIC_COLUMNS].copy()
    lines = [
        "| " + " | ".join(METRIC_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(METRIC_COLUMNS)) + " |",
    ]
    for _, row in rows.iterrows():
        values = []
        for col in METRIC_COLUMNS:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_optional(path: Path) -> str:
    """Read optional manifests so report generation can describe missing inputs."""

    return path.read_text(encoding="utf-8") if path.exists() else "Not available."


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    processed_dir = Path(args.processed_dir)
    modes = ["smoke", "integration"] if args.mode == "all" else [args.mode]
    tables = {mode: write_mode_outputs(results_dir, mode) for mode in modes}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    split_manifest = read_optional(processed_dir / "split_manifest.json")
    feature_manifest = read_optional(processed_dir / "feature_manifest.json")

    # The report intentionally states the validation boundary. These runs prove
    # engineering integration, not paper-level reproduction with many trials.
    content = [
        "# FairJob Integration Validation Report",
        "",
        "This report is generated from single-run smoke/integration artifacts.",
        "It is not a strict reproduction of the NeurIPS FairJob paper results.",
        "",
        "## Data",
        "",
        f"Processed data directory: `{processed_dir}`",
        "",
        "## Split Manifest",
        "",
        "```json",
        split_manifest.strip(),
        "```",
        "",
        "## Feature Manifest",
        "",
        "```json",
        feature_manifest.strip(),
        "```",
    ]
    if "smoke" in tables:
        content += ["", "## Smoke Results", "", markdown_table(tables["smoke"])]
    if "integration" in tables:
        content += ["", "## Integration Results", "", markdown_table(tables["integration"])]

    content += [
        "",
        "## Quality Gates",
        "",
        "- QG1 row alignment is enforced by `check_alignment.py` and `evaluator.py`.",
        "- QG2 unaware configs exclude protected features.",
        "- QG3 unfair configs include `protected_attribute_feat`.",
        "- QG4 evaluator outputs NLLH/AUC/DP/U/U_TILDE.",
        "- QG5 smoke completion requires four metric rows.",
        "- QG6 integration completion requires four metric rows or recorded failure.",
        "- QG7 scripts default to zero XGBoost trials and single-seed runs.",
        "",
        "This validation confirms that FairJob has been integrated into the FuxiCTR engineering workflow and that LR/XGBoost baselines can be evaluated with FairJob-specific metrics. The reported results are single-run integration checks using available or reference hyperparameters. They are not claimed as strict reproduction of the NeurIPS FairJob paper results.",
        "",
        "Current results are not paper-level strict reproduction and do not include 100 trials, multi-seed runs, or a full fairness multiplier grid.",
        "",
    ]
    out.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
