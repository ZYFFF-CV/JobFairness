from pathlib import Path

from fuxictr_ext.fairjob.run_protocol_matrix import command_for_job, read_matrix, select_jobs


ROOT = Path(__file__).resolve().parents[2]


def test_matrix_commands_keep_artifacts_outside_source():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_1_matrix.yaml")
    jobs = select_jobs(matrix, "primary_screening")
    assert len(jobs) == 4
    command, run_dir = command_for_job(jobs[0], matrix, dry_run=False)
    assert run_dir.as_posix().startswith("/root/autodl-tmp/workdirs/JobFairness/stage1_1")
    assert "--representation_out" in command
    assert "/root/autodl-tmp/workdirs/JobFairness/stage1_1/runtime_config" in command
    assert "--dry_run" not in command


def test_dry_run_uses_smoke_config_without_training_flag():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_1_matrix.yaml")
    job = select_jobs(matrix, "primary_screening")[0]
    command, _ = command_for_job(job, matrix, dry_run=True)
    assert "--dry_run" in command
    assert any(value.endswith("_smoke") for value in command)
    assert "--representation_out" not in command
