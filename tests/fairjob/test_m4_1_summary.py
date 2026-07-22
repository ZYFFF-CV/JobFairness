import json

from fuxictr_ext.fairjob.build_m4_1_summary import (
    POSITION_STATUS,
    build_summary,
    render_markdown,
)


def _diagnostic(path, offset):
    metrics = {
        "group_0_NLLH": 0.10 + offset,
        "group_1_NLLH": 0.11 + offset,
        "group_0_Brier": 0.02 + offset,
        "group_1_Brier": 0.03 + offset,
        "group_0_ECE": 0.01 + offset,
        "group_1_ECE": 0.04 + offset,
        "group_0_n": 40,
        "group_1_n": 60,
        "group_0_positive_rate": 0.01,
        "group_1_positive_rate": 0.02,
        "group_0_prediction_mean": 0.03,
        "group_1_prediction_mean": 0.05,
        "group_gap_NLLH": 0.01,
        "group_gap_Brier": 0.01,
        "group_gap_ECE": 0.03,
        "U": 0.2,
        "U_TILDE": 0.25,
    }
    path.write_text(
        json.dumps(
            {
                "git_commit": "abc123",
                "labels": {"model": "DCNv2"},
                "selection_scopes": {
                    "all_logged": metrics,
                    "random_display": metrics,
                },
                "position_corrected": {
                    "status": "unavailable",
                    "reason": "legacy completed artifact",
                },
            }
        ),
        encoding="utf-8",
    )


def test_m4_1_summary_normalizes_existing_unavailable_artifacts(tmp_path):
    jobs = []
    for index in range(6):
        path = tmp_path / f"job_{index}.json"
        _diagnostic(path, index / 1000)
        jobs.append((f"job_{index}", path))

    summary = build_summary(jobs)
    assert summary["m4_status"] == "conditional_pass"
    assert summary["available_data_status"] == "available_data_pass"
    assert summary["position_corrected"] == POSITION_STATUS
    assert summary["position_corrected_used_for_method_selection"] is False
    assert summary["model_count"] == 6
    gap = summary["models"][0]["selection_scopes"]["all_logged"]["calibration_gap"]
    assert gap == {
        "definition": "ECE_group_1_minus_ECE_group_0",
        "signed": 0.03,
        "absolute": 0.03,
    }
    markdown = render_markdown(summary)
    assert "conditional_pass" in markdown
    assert POSITION_STATUS in markdown
