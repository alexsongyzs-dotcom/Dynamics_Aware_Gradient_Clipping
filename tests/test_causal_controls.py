"""Invariants that make the causal branches interpretable."""

import numpy as np
import torch

from analysis.causal_branches import verify_gain_invariants
from src.clipping import (
    ClipDecision,
    apply_post_update_decision_,
    global_norm_clip,
    make_block_shuffled_gains,
    make_random_gate_gains,
    make_time_shuffled_gains,
    snapshot_parameters,
)


def test_norm_matched_vanilla_sgd_is_global_clipping() -> None:
    gradient = torch.tensor([3.0, 4.0])
    learning_rate = 0.2
    threshold = 2.5
    clipped_update = -learning_rate * global_norm_clip(gradient, threshold)
    coefficient = min(1.0, threshold / float(torch.linalg.vector_norm(gradient)))
    norm_matched_update = coefficient * (-learning_rate * gradient)
    torch.testing.assert_close(clipped_update, norm_matched_update)


def test_timing_controls_preserve_preregistered_marginals() -> None:
    gains = [1.0, 0.5, 1.0, 0.25, 0.75, 1.0]
    for intervention in (
        make_time_shuffled_gains(gains, seed=3),
        make_block_shuffled_gains(gains, seed=3, block_size=2),
        make_random_gate_gains(gains, seed=3),
    ):
        result = verify_gain_invariants(gains, intervention)
        assert result["same_multiset"]
        assert result["same_active_count"]


def test_pre_moment_and_post_update_match_step_but_not_momentum_state() -> None:
    pre = torch.nn.Parameter(torch.tensor([1.0]))
    post = torch.nn.Parameter(torch.tensor([1.0]))
    pre_optimizer = torch.optim.SGD([pre], lr=0.1, momentum=0.9)
    post_optimizer = torch.optim.SGD([post], lr=0.1, momentum=0.9)

    pre.grad = torch.tensor([1.0])
    pre_optimizer.step()

    post.grad = torch.tensor([2.0])
    before = snapshot_parameters([post])
    post_optimizer.step()
    apply_post_update_decision_([post], before, ClipDecision(coefficient=0.5, active=True))

    torch.testing.assert_close(pre, post)
    pre_buffer = pre_optimizer.state[pre]["momentum_buffer"]
    post_buffer = post_optimizer.state[post]["momentum_buffer"]
    assert not torch.allclose(pre_buffer, post_buffer)
    torch.testing.assert_close(pre_buffer, torch.tensor([1.0]))
    torch.testing.assert_close(post_buffer, torch.tensor([2.0]))


def test_post_update_target_norm_is_enforced_without_enlargement() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = snapshot_parameters([parameter])
    with torch.no_grad():
        parameter.add_(torch.tensor([3.0, 4.0]))
    result = apply_post_update_decision_(
        [parameter],
        before,
        ClipDecision(target_update_norm=2.5, active=True),
    )
    assert np.isclose(result.raw_update_norm, 5.0)
    assert np.isclose(result.applied_update_norm, 2.5)
    assert np.isclose(result.coefficient, 0.5)
