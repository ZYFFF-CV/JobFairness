import json

from fuxictr_ext.fairjob.run_stage2_pilot import DATASET_ID, expid
from fuxictr_ext.fairjob.stage2_1.freeze_protocol import audit_p2_runs


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_p2_freeze_requires_same_seed_teacher_and_compact_artifacts(tmp_path):
    pilot = tmp_path / "pilot"
    seed = 2019
    baseline_checkpoint = (
        pilot
        / f"seed{seed}"
        / "baseline"
        / "checkpoints"
        / DATASET_ID
        / f"{expid('baseline')}.model"
    )
    for method in ("baseline", "primary"):
        source = pilot / f"seed{seed}" / method
        checkpoint = (
            source
            / "checkpoints"
            / DATASET_ID
            / f"{expid(method)}.model"
        )
        prediction = source / "predictions" / f"{expid(method)}.csv"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text("row_id,y_pred\n1,0.5\n", encoding="utf-8")
        _write_json(
            source / "run_manifest.json",
            {"status": "complete", "git_commit": "training"},
        )
        _write_json(source / "metrics.json", {"AUC": 0.7})
        _write_json(
            source / "hparams.json",
            {
                "hparams": {
                    "stage2_backbone_checkpoint": (
                        None
                        if method == "baseline"
                        else str(baseline_checkpoint)
                    )
                }
            },
        )
    matrix = {
        "seeds": [seed],
        "methods": {"baseline": {}, "primary": {}},
    }
    result = audit_p2_runs(pilot, matrix, "training")
    assert set(result[str(seed)]) == {"baseline", "primary"}
    assert result[str(seed)]["primary"]["prediction_bytes"] > 0
