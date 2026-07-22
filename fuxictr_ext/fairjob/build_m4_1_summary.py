"""Summarize M4.1 calibration and utility from existing outcome diagnostics.

This command is intentionally read-only with respect to model artifacts. It
uses the six completed prediction diagnostics and never retrains, recalibrates,
or estimates a missing logging propensity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


POSITION_STATUS = "unavailable_missing_external_propensity"
SCOPES = ("all_logged", "random_display")
GROUP_METRICS = ("NLLH", "Brier", "ECE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default="configs/fairjob/stage1_1_outcome_matrix.yaml"
    )
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    return parser.parse_args()


def load_jobs(matrix_path: str | Path, input_dir: str | Path | None) -> list[tuple[str, Path]]:
    """Resolve the expected six diagnostics without guessing model identities."""

    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    jobs = matrix.get("jobs", [])
    if len(jobs) != 6:
        raise ValueError(f"M4.1 requires exactly six prediction jobs, found {len(jobs)}.")
    root = (
        Path(input_dir)
        if input_dir is not None
        else Path(matrix["workdir_root"]) / "outcome_diagnostics"
    )
    return [(job["name"], root / f"{job['name']}.json") for job in jobs]


def _scope_summary(metrics: dict) -> dict:
    """Select group calibration and utility fields from one evaluated row scope."""

    groups = {}
    for group in (0, 1):
        groups[f"group_{group}"] = {
            metric: metrics[f"group_{group}_{metric}"] for metric in GROUP_METRICS
        }
        groups[f"group_{group}"].update(
            {
                "n": metrics[f"group_{group}_n"],
                "positive_rate": metrics[f"group_{group}_positive_rate"],
                "prediction_mean": metrics[f"group_{group}_prediction_mean"],
            }
        )
    signed_gap = float(metrics["group_gap_ECE"])
    return {
        "groups": groups,
        "calibration_gap": {
            "definition": "ECE_group_1_minus_ECE_group_0",
            "signed": signed_gap,
            "absolute": abs(signed_gap),
        },
        "group_gaps_signed": {
            metric: metrics[f"group_gap_{metric}"] for metric in GROUP_METRICS
        },
        "U": metrics["U"],
        "U_TILDE": metrics["U_TILDE"],
    }


def build_summary(job_paths: list[tuple[str, Path]]) -> dict:
    """Build the available-data M4.1 result and enforce its claim boundary."""

    models = []
    commits = set()
    for name, path in job_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing completed outcome diagnostic: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        scopes = payload.get("selection_scopes", {})
        missing = sorted(set(SCOPES).difference(scopes))
        if missing:
            raise ValueError(f"{path} is missing M4.1 scopes: {missing}")
        position_status = payload.get("position_corrected")
        # Older completed diagnostics used a structured unavailable marker. The
        # missing column is the same fact, so M4.1 normalizes it without reruns.
        if isinstance(position_status, dict):
            position_status = (
                POSITION_STATUS
                if position_status.get("status") == "unavailable"
                else position_status.get("status")
            )
        if position_status not in {POSITION_STATUS, "unavailable_protocol_not_frozen"}:
            raise ValueError(f"Unexpected position-corrected state in {path}: {position_status}")
        if payload.get("git_commit"):
            commits.add(payload["git_commit"])
        models.append(
            {
                "name": name,
                "labels": payload.get("labels", {}),
                "source": str(path),
                "selection_scopes": {
                    scope: _scope_summary(scopes[scope]) for scope in SCOPES
                },
            }
        )
    return {
        "stage": "M4.1",
        "m4_status": "conditional_pass",
        "available_data_status": "available_data_pass",
        "status_display": "conditional pass / available-data pass",
        "position_corrected": POSITION_STATUS,
        "position_corrected_used_for_method_selection": False,
        "current_method_selection_protocols": [
            "all_logged",
            "random_display",
            "context_conditioned",
        ],
        "source_git_commits": sorted(commits),
        "model_count": len(models),
        "models": models,
    }


def render_markdown(summary: dict) -> str:
    """Render the compact audit table committed with the Stage1.1 reports."""

    lines = [
        "# M4.1 Available-data calibration and utility audit",
        "",
        f"- M4 status: `{summary['m4_status']}` / `{summary['available_data_status']}`.",
        f"- Position-corrected status: `{summary['position_corrected']}`.",
        "- Position-corrected results are excluded from current method selection.",
        "- Current selection protocols: `all_logged`, `random_display`, and "
        "`context_conditioned`.",
        "- Calibration gap is `ECE(group 1) - ECE(group 0)`; both signed and "
        "absolute values are reported.",
        "",
        "| Model | Scope | G0 NLLH | G1 NLLH | G0 Brier | G1 Brier | G0 ECE | G1 ECE | ECE gap signed | ECE gap abs | U | U_TILDE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        for scope in SCOPES:
            values = model["selection_scopes"][scope]
            group_0 = values["groups"]["group_0"]
            group_1 = values["groups"]["group_1"]
            gap = values["calibration_gap"]
            lines.append(
                "| {name} | {scope} | {g0n:.6f} | {g1n:.6f} | {g0b:.6f} | "
                "{g1b:.6f} | {g0e:.6f} | {g1e:.6f} | {signed:.6f} | "
                "{absolute:.6f} | {utility:.6f} | {utility_tilde:.6f} |".format(
                    name=model["name"],
                    scope=scope,
                    g0n=group_0["NLLH"],
                    g1n=group_1["NLLH"],
                    g0b=group_0["Brier"],
                    g1b=group_1["Brier"],
                    g0e=group_0["ECE"],
                    g1e=group_1["ECE"],
                    signed=gap["signed"],
                    absolute=gap["absolute"],
                    utility=values["U"],
                    utility_tilde=values["U_TILDE"],
                )
            )
    lines.extend(
        [
            "",
            "`U` and `U_TILDE` internally evaluate only `displayrandom=1` rows. "
            "They are nevertheless listed under both declared outer scopes so the "
            "evaluation path remains explicit and auditable.",
            "",
            "This is an available-data diagnostic pass, not evidence that the "
            "position-corrected protocol has passed. No logging propensity was "
            "estimated from outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = build_summary(load_jobs(args.matrix, args.input_dir))
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
