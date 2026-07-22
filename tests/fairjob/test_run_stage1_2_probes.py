from pathlib import Path

from fuxictr_ext.fairjob.run_stage1_2_probes import probe_commands, read_matrix


ROOT = Path(__file__).resolve().parents[2]


def test_stage1_2_probe_matrix_covers_all_methods_seeds_and_layers():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_2_probe_matrix.yaml")
    commands = probe_commands(matrix, "full")
    assert len(commands) == 180
    assert len({name for name, _, _ in commands}) == 180
    for _, command, output in commands:
        assert command[command.index("--probe_type") + 1] == "both"
        assert command[command.index("--nonlinear_max_iter") + 1] == "500"
        assert "--nonlinear_early_stopping" in command
        assert "--skip_shard_validation" in command
        assert output.parent.name == "full"


def test_stage1_2_probe_smoke_is_one_frozen_probability_probe():
    matrix = read_matrix(ROOT / "configs/fairjob/stage1_2_probe_matrix.yaml")
    commands = probe_commands(matrix, "smoke")
    assert len(commands) == 1
    name, command, _ = commands[0]
    assert name == "baseline_seed2019__dcnv2_probability"
    assert command[command.index("--probe_sample") + 1] == matrix["probe_sample"]
