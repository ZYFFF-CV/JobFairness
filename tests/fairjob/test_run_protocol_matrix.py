from pathlib import Path

import sys

from fuxictr_ext.fairjob.run_protocol_matrix import (
    command_for_job,
    read_matrix,
    run_streaming,
    select_jobs,
)


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


def test_multiseed_jobs_use_isolated_run_dirs_and_explicit_seeds():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_1_matrix.yaml")
    jobs = select_jobs(matrix, "identity_multiseed")
    assert len(jobs) == 4
    commands_and_dirs = [command_for_job(job, matrix, dry_run=False) for job in jobs]
    assert len({str(run_dir) for _, run_dir in commands_and_dirs}) == 4
    for job, (command, run_dir) in zip(jobs, commands_and_dirs):
        seed_index = command.index("--seed")
        assert command[seed_index + 1] == str(job["seed"])
        assert str(job["seed"]) in run_dir.name


def test_multiseed_representation_exports_reuse_fixed_sampling_seed():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_1_matrix.yaml")
    jobs = select_jobs(matrix, "identity_multiseed")
    commands_and_dirs = [
        command_for_job(job, matrix, dry_run=False, representations_only=True)
        for job in jobs
    ]
    for command, run_dir in commands_and_dirs:
        assert "--export_representations_only" in command
        seed_index = command.index("--representation_seed")
        assert command[seed_index + 1] == "2019"
        out_index = command.index("--representation_out")
        assert command[out_index + 1] == str(run_dir / "representations_aligned")


def test_streaming_runner_mirrors_output_to_terminal_and_log(tmp_path, capsys):
    log_path = tmp_path / "runner.log"
    returncode = run_streaming(
        [sys.executable, "-c", "print('live-loss=0.125', flush=True)"],
        ROOT,
        log_path,
    )
    assert returncode == 0
    assert "live-loss=0.125" in capsys.readouterr().out
    assert "live-loss=0.125" in log_path.read_text(encoding="utf-8")


def test_m5a_matrix_is_six_methods_by_three_seeds():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_1_m5_matrix.yaml")
    jobs = select_jobs(matrix, "m5a_dcnv2_three_seed")
    assert len(jobs) == 18
    assert {job["seed"] for job in jobs} == {2019, 2020, 2021}
    assert len({job["name"] for job in jobs}) == 18
    for job in jobs:
        command, run_dir = command_for_job(job, matrix, dry_run=False)
        assert run_dir.as_posix().startswith(
            "/root/autodl-tmp/workdirs/JobFairness/stage1_1/m5/training/"
        )
        assert "--seed" in command
        assert "--representation_out" not in command


def test_stage1_2_matrix_reuses_all_m5a_checkpoints_and_predictions():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_2_matrix.yaml")
    assert matrix["stage1_1_frozen_groups"] == [
        "primary_screening",
        "identity_interventions",
        "identity_multiseed",
    ]
    jobs = select_jobs(matrix, "post_intervention_export")
    assert len(jobs) == 18
    assert {job["seed"] for job in jobs} == {2019, 2020, 2021}
    for job in jobs:
        command, run_dir = command_for_job(
            job, matrix, dry_run=False, representations_only=True
        )
        source = (
            Path(matrix["checkpoint_workdir_root"])
            / "training"
            / job["name"]
        )
        assert command[command.index("--model_root") + 1] == str(
            source / "checkpoints"
        )
        assert command[command.index("--reference_prediction") + 1] == str(
            source / "predictions" / f"{job['expid']}.csv"
        )
        assert command[command.index("--representation_out") + 1] == str(
            run_dir / "representations"
        )
