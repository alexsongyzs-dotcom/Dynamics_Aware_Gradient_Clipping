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
