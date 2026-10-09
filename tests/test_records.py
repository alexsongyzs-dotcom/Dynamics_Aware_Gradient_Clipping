"""Artifact identity and overwrite-protection tests."""

import pytest

from src.records import RunRecorder, config_hash


def test_resume_path_does_not_change_scientific_config_hash() -> None:
    first = {"seed": 0, "checkpointing": {"resume_from": None, "every_epochs": 5}}
    second = {"seed": 0, "checkpointing": {"resume_from": "checkpoint.pt", "every_epochs": 5}}
    assert config_hash(first) == config_hash(second)


def test_recorder_refuses_to_mix_two_runs_with_same_id(tmp_path) -> None:
    config = {"seed": 0}
    RunRecorder(tmp_path, "same-id", config)
    with pytest.raises(FileExistsError):
        RunRecorder(tmp_path, "same-id", config)


def test_recorder_allows_matching_resume(tmp_path) -> None:
    first = {"seed": 0, "checkpointing": {"resume_from": None}}
    resumed = {"seed": 0, "checkpointing": {"resume_from": "checkpoint.pt"}}
    RunRecorder(tmp_path, "resume-id", first)
    RunRecorder(tmp_path, "resume-id", resumed, append_existing=True)


def test_resume_preserves_abandoned_suffix_without_duplicate_steps(tmp_path) -> None:
    recorder = RunRecorder(tmp_path, "run", {"seed": 1})
    for step in range(5):
        recorder.append_step({"step": step, "loss": 1.0})
    recorder.close()
    resumed = RunRecorder(tmp_path, "run", {"seed": 1}, append_existing=True)
    kept = resumed.reconcile_checkpoint(3)
    assert [row["step"] for row in kept] == [0, 1, 2]
    assert len(list(tmp_path.glob("*.bak"))) == 1
    resumed.append_step({"step": 3, "loss": 0.5})
    resumed.close()
    import json
    assert [json.loads(line)["step"] for line in resumed.steps_path.read_text().splitlines()] == [0, 1, 2, 3]
