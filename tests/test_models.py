"""Real-model operations must support the strict CUDA replay contract."""

import os

import pytest
import torch
import torch.nn.functional as F

from src.models import DeterministicAdaptiveAvgPool2d, SmallCNN


@pytest.mark.parametrize("shape", [(14, 14), (16, 16), (7, 11)])
def test_deterministic_pool_matches_adaptive_windows_and_gradients(shape):
    actual_input = torch.randn(2, 3, *shape, dtype=torch.float64, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_()
    actual = DeterministicAdaptiveAvgPool2d((4, 4))(actual_input)
    reference = F.adaptive_avg_pool2d(reference_input, (4, 4))
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (reference * weights).sum().backward()
    torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-13, atol=1e-14)


def test_small_cnn_has_repeatable_backward_with_strict_determinism():
    device = os.environ.get("DAGC_TEST_DEVICE", "cpu")
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model = SmallCNN().to(device)
        inputs = torch.randn(4, 1, 28, 28, device=device)
        model(inputs).square().sum().backward()
        first = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        model(inputs).square().sum().backward()
        assert all(torch.equal(a, p.grad) for a, p in zip(first, model.parameters()))
    finally:
        torch.use_deterministic_algorithms(previous)
