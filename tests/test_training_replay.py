"""G0 integration: replay real optimizer, augmentation, dropout and scheduler state."""

import json
import os

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

import src.data as data
import src.train as training


class AugmentedTinyDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(33)
        self.inputs = torch.randn(32, 1, 28, 28, generator=generator)
        self.targets = torch.randint(0, 3, (32,), generator=generator)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index] + 0.01 * torch.randn_like(self.inputs[index]), self.targets[index]


@pytest.fixture
def tiny_problem(monkeypatch):
    dataset = AugmentedTinyDataset()
    deterministic = TensorDataset(dataset.inputs, dataset.targets)
    monkeypatch.setattr(data, "_datasets", lambda *args: (dataset, deterministic, 3, 1, deterministic))
    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: nn.Sequential(
        nn.Flatten(), nn.Linear(784, 8), nn.Tanh(), nn.Dropout(0.2), nn.Linear(8, 3),
    ))


def config(tmp_path, name, policy=None):
    return dict(
        run_id=name, output_dir=str(tmp_path / name), seed=42, device=os.environ.get("DAGC_TEST_DEVICE", "cpu"),
        deterministic=True, cpu_threads=1, epochs=3, batch_size=8, num_workers=0,
        pin_memory=False, return_step_rows=True,
        dataset=dict(name="fashion_mnist", data_dir=str(tmp_path / "data")),
        model=dict(name="small_cnn"),
        optimizer=dict(name="sgd", lr=0.05, momentum=0.9, schedule="cosine"),
        clipping=policy or dict(name="autoclip", percentile=40, warmup=2),
        logging=dict(probe_size=8, gradient_sketch_size=8, full_gradient_alignment=True,
                     hessian_checkpoints=[0], hessian_power_iterations=2),
        checkpointing=dict(directory=str(tmp_path / name / "checkpoints"),
                           save_initial=True, save_final=True, save_latest=True, steps=[3]),
    )


def assert_models_equal(first, second):
    a = torch.load(first, weights_only=False, map_location="cpu")
    b = torch.load(second, weights_only=False, map_location="cpu")
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    return a, b


@pytest.mark.parametrize("pause_step", [3, 4, 7])
def test_training_resume_is_exact_mid_epoch_and_at_boundary(tmp_path, tiny_problem, pause_step):
    uninterrupted = training.train_run(config(tmp_path, "complete"))
    cfg = config(tmp_path, "resumed")
    paused = training.train_run(cfg, stop_after_steps=pause_step)
    cfg["checkpointing"]["resume_from"] = paused["final_checkpoint"]
    resumed = training.train_run(cfg)
    assert resumed["completed_steps"] == uninterrupted["completed_steps"] == 12
    assert_models_equal(uninterrupted["final_checkpoint"], resumed["final_checkpoint"])
    for key in ("loss", "coefficient", "gradient_alignment", "applied_update_norm", "learning_rate"):
        np.testing.assert_allclose(np.asarray([r[key] for r in uninterrupted["rows"]], dtype=float),
                                   np.asarray([r[key] for r in resumed["rows"]], dtype=float),
                                   rtol=0, atol=0, equal_nan=True)


def test_frozen_checkpoint_branch_matches_reference_with_common_continuation(tmp_path, tiny_problem):
    cfg = config(tmp_path, "reference", dict(name="fixed", threshold=0.5))
    original = training.train_run(cfg)
    gains = [row["coefficient"] for row in original["rows"][3:7]]
    sequence_path = tmp_path / "gains.json"
    sequence_path.write_text(json.dumps(gains))
    branch = config(tmp_path, "branch", dict(name="frozen_gain", sequence_path=str(sequence_path), start_step=3))
    branch["checkpointing"]["branch_from"] = str(tmp_path / "reference/checkpoints/reference.step-3.pt")
    branch["causal_branch"] = dict(intervention_end_step=7, continuation_clipping=cfg["clipping"])
    replay = training.train_run(branch)
    assert_models_equal(original["final_checkpoint"], replay["final_checkpoint"])
    assert replay["rows"][0]["step"] == 3
    np.testing.assert_allclose([r["loss"] for r in original["rows"][3:]],
                               [r["loss"] for r in replay["rows"]], rtol=0, atol=0)

    paused_branch = config(tmp_path, "paused-branch", branch["clipping"])
    paused_branch["checkpointing"]["branch_from"] = branch["checkpointing"]["branch_from"]
    paused_branch["causal_branch"] = branch["causal_branch"]
    paused = training.train_run(paused_branch, stop_after_steps=5)
    paused_branch["checkpointing"]["resume_from"] = paused["final_checkpoint"]
    resumed = training.train_run(paused_branch)
    assert_models_equal(original["final_checkpoint"], resumed["final_checkpoint"])


def test_validation_and_probe_exclude_test_and_validation_training_indices(tmp_path, tiny_problem):
    bundle = data.build_data_bundle(data_dir=str(tmp_path), validation_size=8,
                                   evaluation_split="validation", probe_size=24, pin_memory=False)
    training_indices = set(bundle.train_loader.dataset.indices)
    validation_indices = set(bundle.test_loader.dataset.indices)
    assert len(training_indices) == 24
    assert len(validation_indices) == 8
    assert not training_indices & validation_indices
    assert set(bundle.probe_loader.dataset.dataset.indices) == training_indices
