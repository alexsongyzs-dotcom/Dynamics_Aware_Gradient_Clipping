"""Sanity tests for clipping operators (mirrors the companion math paper's
verified identities where applicable).

Run: python -m pytest tests/
"""

import torch

from src.clipping import clipping_coefficient, clipping_indicator, clipping_intensity, global_norm_clip


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
