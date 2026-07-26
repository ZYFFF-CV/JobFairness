import json

import pytest

from fuxictr_ext.fairjob.representation_io import sha256_file
from fuxictr_ext.fairjob.stage2_1.run_tier1 import (
    build_command,
    output_path,
    read_completion,
    validate_closure,
)


def test_tier1_command_uses_existing_p2_checkpoint_run(tmp_path):
    command = build_command(
        "python",
        "protocol.yaml",
        tmp_path / "config",
        tmp_path / "pilot",
        tmp_path / "stage2_1",
        0,
        2019,
        2020,
        "baseline",
    )
    assert str(tmp_path / "pilot" / "seed2020" / "baseline") in command
    assert str(output_path(tmp_path / "stage2_1", 2020, "baseline")) in command
    assert "--protocol" in command


def test_closure_and_completion_require_exact_commit_and_protocol(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("version: 1\n", encoding="utf-8")
    closure = tmp_path / "closure.json"
    closure.write_text(
        json.dumps(
            {
                "stage": "S21-M0",
                "status": "frozen",
                "stage2_1_code_commit": "commit",
                "stage2_1_protocol_sha256": sha256_file(protocol),
                "p2_run_count": 27,
            }
        ),
        encoding="utf-8",
    )
    assert validate_closure(closure, protocol, "commit")["status"] == "frozen"
    with pytest.raises(ValueError, match="stale code"):
        validate_closure(closure, protocol, "other")

    result = tmp_path / "result.json"
    assert read_completion(result, "commit", "hash") == "pending"
    result.write_text(
        json.dumps(
            {
                "status": "complete",
                "audit_git_commit": "commit",
                "stage2_1_protocol_sha256": "hash",
            }
        ),
        encoding="utf-8",
    )
    assert read_completion(result, "commit", "hash") == "complete"
    assert read_completion(result, "other", "hash") == "stale"
