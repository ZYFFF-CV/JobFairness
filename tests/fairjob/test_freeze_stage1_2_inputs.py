import json
from pathlib import Path

import pandas as pd
import yaml

from fuxictr_ext.fairjob import freeze_stage1_2_inputs as freeze


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_freeze_inputs_records_complete_six_method_three_seed_matrix(
    tmp_path: Path, monkeypatch
):
    project_root = tmp_path / "project"
    workdir = tmp_path / "workdirs" / "stage1_1" / "m5"
    data_dir = tmp_path / "datasets" / "processed"
    probe_sample = tmp_path / "workdirs" / "stage1_1" / "probes" / "probe_rows.npz"
    for relative_path in freeze.REQUIRED_DOCS:
        path = project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path, encoding="utf-8")
    data_dir.mkdir(parents=True)
    for name in ("feature_manifest.json", "split_manifest.json", "test_meta.csv"):
        (data_dir / name).write_text(name, encoding="utf-8")
    probe_sample.parent.mkdir(parents=True)
    probe_sample.write_bytes(b"probe")
    probe_sample.with_suffix(".json").write_text("{}", encoding="utf-8")

    methods = (
        "baseline",
        "global_suppression",
        "matched_random_suppression",
        "selective_suppression",
        "dp_regularization",
        "adversarial",
    )
    jobs = []
    runtime = {"python": "test", "packages": {}}
    for method in methods:
        for seed in (2019, 2020, 2021):
            name = f"{method}_seed{seed}"
            expid = f"M5DCNv2_{method}_full"
            dataset_id = "fairjob_m5_full"
            jobs.append(
                {
                    "name": name,
                    "group": "post_intervention_export",
                    "expid": expid,
                    "dataset_id": dataset_id,
                    "seed": seed,
                }
            )
            run_dir = workdir / "training" / name
            _write_json(
                run_dir / "run_manifest.json",
                {
                    "status": "complete",
                    "seed": seed,
                    "git_commit": "training-commit",
                    "config_hash": f"hash-{name}",
                    "runtime": runtime,
                },
            )
            _write_json(run_dir / "job.success.json", {"returncode": 0})
            checkpoint = run_dir / "checkpoints" / dataset_id / f"{expid}.model"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(name.encode("ascii"))
            prediction = run_dir / "predictions" / f"{expid}.csv"
            prediction.parent.mkdir(parents=True)
            pd.DataFrame(
                {"row_id": [0, 1], "y_true": [0, 1], "y_pred": [0.1, 0.9]}
            ).to_csv(prediction, index=False)
            _write_json(run_dir / "hparams.json", {"seed": seed})
            _write_json(run_dir / "metrics.json", {"AUC": 1.0})

    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "checkpoint_workdir_root": str(workdir),
                "data_dir": str(data_dir),
                "probe_sample": str(probe_sample),
                "expected_test_rows": 2,
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(freeze, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(freeze, "git_commit", lambda _: "baseline-commit")
    monkeypatch.setattr(freeze, "runtime_versions", lambda: runtime)

    payload = freeze.freeze_inputs(matrix_path)
    assert payload["artifact_count"] == 18
    assert payload["m5a_training_commit"] == "training-commit"
    assert payload["expected_prediction_rows"] == 2
    assert len({item["job"] for item in payload["artifacts"]}) == 18
    assert payload["stage1_1_status"]["m5b_fairness_utility_gate"] == "failed"
