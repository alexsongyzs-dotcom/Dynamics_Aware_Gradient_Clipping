"""Round-trip tests for the state needed by deterministic branches."""

import random

import numpy as np
import torch

from src.checkpointing import build_checkpoint, load_checkpoint, save_checkpoint
from src.clipping import AutoClipPolicy, PolicyContext
from src.data import StatefulRandomSampler
from src.dynamics import ClippingEpisodeTracker, DynamicsHistory


def test_checkpoint_restores_rng_sampler_and_policy_state(tmp_path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sampler = StatefulRandomSampler(range(8), seed=11)
    sampler_iterator = iter(sampler)
    _ = [next(sampler_iterator) for _ in range(3)]
    policy = AutoClipPolicy(percentile=50.0, window=8, warmup=2)
    policy.decide(PolicyContext(step=0, grad_norm=1.0))
    episode_tracker = ClippingEpisodeTracker()
    observation = episode_tracker.update(1.2, step=0)
    history_tracker = DynamicsHistory(window=4)
    history_tracker.update(observation, intensity=0.2)

    payload = build_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=17,
        config={"seed": 7},
        policy=policy,
        sampler=sampler,
        episode_tracker=episode_tracker,
        history_tracker=history_tracker,
    )
    path = save_checkpoint(tmp_path / "roundtrip.pt", payload)
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    expected_indices = list(iter(sampler))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
    restored_sampler = StatefulRandomSampler(range(8), seed=99)
    restored_policy = AutoClipPolicy(percentile=50.0, window=8, warmup=2)
    restored_episode = ClippingEpisodeTracker()
    restored_history = DynamicsHistory(window=4)
    position, _ = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        policy=restored_policy,
        sampler=restored_sampler,
        episode_tracker=restored_episode,
        history_tracker=restored_history,
    )

    actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert position.epoch == 2 and position.global_step == 17
    assert actual_rng == expected_rng
    assert list(iter(restored_sampler)) == expected_indices
    assert restored_policy.state_dict() == policy.state_dict()
    assert restored_episode.state_dict() == episode_tracker.state_dict()
    assert restored_history.state_dict() == history_tracker.state_dict()
