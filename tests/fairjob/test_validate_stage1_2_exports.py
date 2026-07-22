import json
from pathlib import Path

import numpy as np
import yaml

from fuxictr_ext.fairjob import validate_stage1_2_exports as validation
from fuxictr_ext.fairjob.representation_io import RepresentationShardWriter


def test_validate_complete_stage1_2_export_matrix(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "stage1_2"
    probe_sample = tmp_path / "probe_rows.npz"
    row_ids = np.array([2, 5], dtype=np.int64)
    np.savez(probe_sample, train=row_ids, valid=row_ids, test=row_ids)
    jobs = []
    for method in (
        "baseline",
        "global_suppression",
        "matched_random_suppression",
        "selective_suppression",
        "dp_regularization",
        "adversarial",
    ):
        for seed in (2019, 2020, 2021):
            name = f"{method}_seed{seed}"
            jobs.append({"name": name, "seed": seed})
            run_dir = workdir / "representation_exports" / name
            run_dir.mkdir(parents=True)
            (run_dir / "representation_export.success.json").write_text(
                json.dumps({"returncode": 0}), encoding="utf-8"
            )
            (run_dir / "representation_export_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "representation_export_complete",
                        "reference_prediction_check": {
                            "max_abs_prediction_diff": 0.0
                        },
                    }
                ),
                encoding="utf-8",
            )
            for split in ("train", "valid", "test"):
                split_dir = run_dir / "representations" / split
                writer = RepresentationShardWriter(split_dir)
                pre = np.ones((2, 3), dtype=np.float32)
                residual = pre.copy()
                suppressed = np.zeros_like(pre)
                writer.add(
                    {
                        "row_id": row_ids,
                        "y_true": np.array([0, 1], dtype=np.float32),
                        "protected_attribute": np.array([0, 1], dtype=np.int8),
                        "y_pred": np.array([0.1, 0.9], dtype=np.float32),
                        "dcnv2_final_pre_mitigation": pre,
                        "dcnv2_suppressed_component": suppressed,
                        "dcnv2_residual_component": residual,
                    }
                )
                shards = writer.close()
                (split_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "rows": 2,
                            "shards": shards,
                            "sampling": {
                                "method": "frozen_row_ids",
                                "row_id_source": str(probe_sample),
                            },
                            "representation_shapes": {
                                "dcnv2_final_pre_mitigation": [3],
                                "dcnv2_suppressed_component": [3],
                                "dcnv2_residual_component": [3],
                            },
                            "model_representation_metadata": {
                                "mitigation_method": method,
                                "training_seed": seed,
                                "suppressed_dimensions": 0,
                                "suppression_mask_sha256": "0" * 64,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "workdir_root": str(workdir),
                "probe_sample": str(probe_sample),
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(validation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(validation, "git_commit", lambda _: "test-commit")

    payload = validation.validate_exports(matrix)
    assert payload["status"] == "pass"
    assert payload["job_count"] == 18
    assert payload["split_manifest_count"] == 54
    assert payload["max_abs_prediction_diff"] == 0.0
    assert payload["max_component_identity_error"] == 0.0
