"""Reference optimizer baseline tests."""

import torch

from src.optimizers import StableAdamW


def test_stable_adamw_records_per_tensor_update_clipping_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = StableAdamW([parameter], lr=0.1, betas=(0.0, 0.9), eps=1e-8)
    parameter.grad = torch.tensor([10.0, 0.1])
    optimizer.step()
    state = optimizer.state[parameter]
    assert torch.isfinite(parameter).all()
    assert state["update_rms"] >= 0.0
    assert state["update_clip_scale"] >= 1.0
