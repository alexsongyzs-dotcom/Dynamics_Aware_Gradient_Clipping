"""Gradient clipping operators and the switching geometry.

Implements the state-dependent clipping rule
    g_tilde_t = min(1, c_t / ||g_t||) * g_t
and the associated empirical switching statistics (clipping frequency,
intensity, switching indicator).
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import Tensor


def clipping_coefficient(grad_norm: Tensor, threshold: float) -> Tensor:
    """Return alpha_t = min(1, c_t / ||g_t||)."""
    return torch.clamp(threshold / (grad_norm + 1e-12), max=1.0)


def clipping_indicator(grad_norm: Tensor, threshold: float) -> Tensor:
    """Return e_t = 1{||g_t|| > c_t}."""
    return (grad_norm > threshold).float()


def clipping_intensity(grad_norm: Tensor, threshold: float) -> Tensor:
    """Return I_t = (1 - c_t / ||g_t||)_+."""
    return torch.clamp(1.0 - threshold / (grad_norm + 1e-12), min=0.0)


def global_norm_clip(grad: Tensor, threshold: float) -> Tensor:
    """Global norm clipping of a single gradient tensor.

    Args:
        grad: gradient tensor (any shape).
        threshold: clipping threshold c_t.

    Returns:
        Clipped gradient with norm at most 'threshold'.
    """
    n = torch.norm(grad)
    if n > threshold:
        return grad * (threshold / n)
    return grad


class DynamicsAwareClipping:
    """DAGC bounded multiplicative controller.

    c_{t+1} = clip(c_t * exp(gamma * tanh(C_t)), c_min, c_max)

    Controller signal (scale-free, in (-inf, inf)):
        C_t = relax * (1 - E_t) - osc_weight * max(0, -a_ema)
    with a_ema the EMA of the gradient alignment and E_t the exposure EMA.
    The oscillation penalty acts only on direction reversal (a < 0), so
    ordinary noisy-but-aligned descent relaxes clipping, while genuine
    oscillatory regimes tighten it.

    Positive C_t relaxes clipping (increases c), negative C_t tightens it.
    The bounds (c_min, c_max) track the running gradient-norm scale, so the
    controller does not need per-problem threshold tuning.
    """

    def __init__(
        self,
        gamma: float = 0.05,
        beta: float = 0.9,
        relax: float = 0.3,
        osc_weight: float = 3.0,
        beta_a: float = 0.9,
        c_min_scale: float = 0.1,
        c_max_scale: float = 10.0,
        init_c: float = 1.0,
        norm_window: int = 200,
    ) -> None:
        self.gamma = gamma
        self.beta = beta
        self.relax = relax
        self.osc_weight = osc_weight
        self.beta_a = beta_a
        self.c_min_scale = c_min_scale
        self.c_max_scale = c_max_scale
        self.c = float(init_c)
        self.E = 0.0
        self.a_ema = 1.0
        self._norm_buf: deque[float] = deque(maxlen=norm_window)
        self.history: list[float] = []  # recorded thresholds (for diagnostics)

    def _bounds(self) -> tuple[float, float]:
        if not self._norm_buf:
            return self.c_min_scale, self.c_max_scale
        med = float(torch.median(torch.tensor(list(self._norm_buf))))
        return max(self.c_min_scale * med, 1e-12), max(self.c_max_scale * med, 1e-9)

    def update(self, grad_norm: float, alignment: float) -> float:
        """Advance one step; return the new threshold c_{t+1}."""
        self._norm_buf.append(grad_norm)
        e = 1.0 if grad_norm > self.c else 0.0
        self.E = self.beta * self.E + (1.0 - self.beta) * e

        self.a_ema = self.beta_a * self.a_ema + (1.0 - self.beta_a) * alignment
        c_t = self.relax * (1.0 - self.E) - self.osc_weight * max(0.0, -self.a_ema)
        self.c = self.c * math.exp(self.gamma * math.tanh(c_t))
        c_min, c_max = self._bounds()
        self.c = min(max(self.c, c_min), c_max)
        self.history.append(self.c)
        return self.c

    def clip_coefficient(self, grad_norm: float) -> float:
        """Current coefficient min(1, c / ||g||) for the ongoing step."""
        return min(1.0, self.c / (grad_norm + 1e-12))
