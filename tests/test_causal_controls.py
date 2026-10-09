"""Invariants that make the causal branches interpretable."""

from pathlib import Path

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


def test_exact_norm_matching_records_required_enlargement() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    before = snapshot_parameters([parameter])
    with torch.no_grad():
        parameter.add_(torch.tensor([3.0, 4.0]))
    result = apply_post_update_decision_([parameter], before, ClipDecision(target_update_norm=6.0), allow_enlarge=True)
    assert np.isclose(result.applied_update_norm, 6.0)
    assert np.isclose(result.coefficient, 1.2)


def test_frozen_gain_uses_checkpoint_relative_index() -> None:
    from src.clipping import FrozenGainPolicy, PolicyContext
    policy = FrozenGainPolicy([0.5, 0.25], start_step=100)
    assert policy.decide(PolicyContext(step=100, grad_norm=2.0)).coefficient == 0.5
    assert policy.decide(PolicyContext(step=101, grad_norm=2.0)).coefficient == 0.25


def test_branch_preparation_checks_shared_checkpoint_and_real_invariants(tmp_path) -> None:
    import json
    import yaml
    from src.branching import prepare_causal_branches
    base = tmp_path / "reference.yaml"
    base.write_text(yaml.safe_dump({"seed": 2, "clipping": {"name": "fixed", "threshold": 1.0}}))
    checkpoint = tmp_path / "reference.step-3.pt"
    torch.save({"position": {"global_step": 3}}, checkpoint)
    steps = tmp_path / "reference.steps.jsonl"
    gains = [1.0, 0.5, 0.25, 1.0]
    steps.write_text("\n".join(json.dumps({"step": step, "coefficient": gain, "applied_update_norm": 0.1 * gain})
                               for step, gain in enumerate(gains, start=3)))
    path = prepare_causal_branches(reference_steps=steps, base_config=base, output_dir=tmp_path / "branches",
                                   shuffle_seed=9, block_size=2, checkpoint_path=checkpoint, start_step=3, intervention_steps=4)
    manifest = json.loads(path.read_text())
    assert manifest["start_step"] == 3
    assert all(check["same_multiset"] and check["same_sum"] for check in manifest["invariants"].values())
    branch = yaml.safe_load(Path(manifest["branch_configs"]["reference_norm_post_update"]).read_text())
    assert branch["checkpointing"]["branch_from"] == str(checkpoint)
    assert branch["clipping"]["allow_enlarge"] is True
    assert branch["causal_branch"]["intervention_end_step"] == 7
