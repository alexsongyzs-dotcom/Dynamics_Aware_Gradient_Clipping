"""Sanity tests for clipping operators (mirrors the companion math paper's
verified identities where applicable).

Run: python -m pytest tests/
"""

import pytest
import torch

from src.clipping import (
    AdaGCPolicy,
    clipping_coefficient,
    clipping_indicator,
    clipping_intensity,
    global_norm_clip,
)


def test_clip_reduces_norm() -> None:
    g = torch.randn(1000)
    out = global_norm_clip(g, threshold=0.5)
    assert torch.norm(out) <= 0.5 + 1e-6


def test_clip_inactive_below_threshold() -> None:
    g = torch.full((10,), 0.1)
    out = global_norm_clip(g, threshold=1.0)
    torch.testing.assert_close(out, g)


def test_coefficient_values() -> None:
    assert clipping_coefficient(torch.tensor(0.5), 1.0).item() == 1.0
    assert clipping_coefficient(torch.tensor(2.0), 1.0).item() == 0.5
    assert clipping_indicator(torch.tensor(2.0), 1.0).item() == 1.0
    assert clipping_intensity(torch.tensor(2.0), 1.0).item() == 0.5


def test_adagc_uses_clipped_norm_in_ema() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    policy = AdaGCPolicy(
        relative_threshold=1.0,
        beta=0.5,
        warmup_steps=1,
        warmup_global_threshold=1.0,
    )
    parameter.grad = torch.tensor([2.0])
    assert policy.apply_([parameter], global_grad_norm=2.0) == pytest.approx(0.5)
    assert policy.ema_norms == pytest.approx([1.0])
    parameter.grad = torch.tensor([4.0])
    assert policy.apply_([parameter], global_grad_norm=4.0) == pytest.approx(0.25)
    assert policy.ema_norms == pytest.approx([1.0])
