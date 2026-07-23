import json

from fuxictr_ext.fairjob.run_stage2_probe_matrix import (
    build_command,
    output_path,
    read_probe_completion,
)


def test_probe_command_uses_exact_p2_run_and_compact_output(tmp_path):
    command = build_command(
        "python",
        tmp_path / "config",
        tmp_path / "pilot",
        tmp_path / "probes",
        "graph.yaml",
        "protocol.yaml",
        0,
        2019,
        2020,
        "baseline",
    )
    assert str(tmp_path / "pilot" / "seed2020" / "baseline") in command
    assert str(
        tmp_path
        / "probes"
        / "seed2020"
        / "baseline"
        / "checkpoint_probe.json"
    ) in command
    assert "--gpu" in command
    assert "--probe_seed" in command


def test_probe_completion_requires_current_audit_commit(tmp_path):
    path = output_path(tmp_path, 2019, "baseline")
    assert read_probe_completion(path, "new") == "pending"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"status": "running", "audit_git_commit": "new"}),
        encoding="utf-8",
    )
    assert read_probe_completion(path, "new") == "incomplete"
    path.write_text(
        json.dumps({"status": "complete", "audit_git_commit": "old"}),
        encoding="utf-8",
    )
    assert read_probe_completion(path, "new") == "stale"
    path.write_text(
        json.dumps({"status": "complete", "audit_git_commit": "new"}),
        encoding="utf-8",
    )
    assert read_probe_completion(path, "new") == "complete"
