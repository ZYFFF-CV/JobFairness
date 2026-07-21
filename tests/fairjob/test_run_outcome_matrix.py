from pathlib import Path

from fuxictr_ext.fairjob.run_outcome_matrix import commands, read_matrix


ROOT = Path(__file__).resolve().parents[2]


def test_outcome_matrix_keeps_outputs_in_stage_workdir():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_1_outcome_matrix.yaml")
    jobs = commands(matrix)
    assert len(jobs) == 6
    for _, command, output in jobs:
        assert "--pred" in command
        assert command[command.index("--bootstrap_repeats") + 1] == "20"
        assert output.as_posix().startswith(
            "/root/autodl-tmp/workdirs/JobFairness/stage1_1/outcome_diagnostics"
        )
