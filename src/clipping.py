"""Gradient clipping operators and the switching geometry.

Implements the state-dependent clipping rule
    g_tilde_t = min(1, c_t / ||g_t||) * g_t
and the associated empirical switching statistics (clipping frequency,
intensity, switching indicator).

Baselines:
    adaptive_gradient_clip_  - AGC (Brock et al., 2021), parameter-relative.
    ZClip                    - Kumar et al., 2025, z-score spike detection.
    AdaGC                    - Wang et al., 2025, per-unit EMA thresholds.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
from torch import Tensor


def finite_mask(values) -> np.ndarray:
    """Boolean mask of the entries that are finite.

    Diverged runs put NaN/inf into the loss and gradient-norm logs, so every
    downstream aggregate has to filter them or the summary itself becomes NaN.
    """
    return np.isfinite(np.asarray(values, dtype=float))


def nanmean_or_nan(values) -> float:
    """Mean over the finite entries, or NaN when none remain."""
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return float("nan")
    return float(arr[mask].mean())



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


def adaptive_gradient_clip_(parameters, clip_value: float = 0.01, eps: float = 1e-3) -> None:
    """Apply unit-wise adaptive gradient clipping in place.

    This is the AGC baseline: each parameter tensor's gradient is bounded by
    ``clip_value * max(||parameter||, eps)``. Scalars and vectors form one
    unit; convolutional and matrix weights use one unit per output channel.
    """
    for p in parameters:
        if p.grad is None:
            continue
        if p.ndim <= 1:
            param_norm = p.detach().norm().clamp_min(eps)
            grad_norm = p.grad.norm()
        else:
            dims = tuple(range(1, p.ndim))
            param_norm = torch.linalg.vector_norm(p.detach(), dim=dims, keepdim=True).clamp_min(eps)
            grad_norm = torch.linalg.vector_norm(p.grad, dim=dims, keepdim=True)
        max_norm = param_norm * clip_value
        scale = (max_norm / grad_norm.clamp_min(1e-12)).clamp(max=1.0)
        p.grad.mul_(scale)


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

    ``exposure_target`` activates the second controller version.  Instead of
    merely slowing threshold growth under high exposure, it regulates exposure
    around a target: E_t above the target tightens clipping and E_t below it
    relaxes clipping.  ``None`` retains the original controller for an honest
    V1 baseline.
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
        exposure_target: float | None = None,
    ) -> None:
        self.gamma = gamma
        self.beta = beta
        self.relax = relax
        self.osc_weight = osc_weight
        self.beta_a = beta_a
        self.c_min_scale = c_min_scale
        self.c_max_scale = c_max_scale
        self.c = float(init_c)
        self.exposure_target = exposure_target
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
        exposure_term = (
            self.relax * (1.0 - self.E)
            if self.exposure_target is None
            else self.relax * (self.exposure_target - self.E)
        )
        c_t = exposure_term - self.osc_weight * max(0.0, -self.a_ema)
        self.c = self.c * math.exp(self.gamma * math.tanh(c_t))
        c_min, c_max = self._bounds()
        self.c = min(max(self.c, c_min), c_max)
        self.history.append(self.c)
        return self.c

    def clip_coefficient(self, grad_norm: float) -> float:
        """Current coefficient min(1, c / ||g||) for the ongoing step."""
        return min(1.0, self.c / (grad_norm + 1e-12))


class ZClip:
    """ZClip: z-score spike mitigation for LLM pre-training.

    Kumar, Owen, Roy Chowdhury & Guera, arXiv:2504.02507 (2025).

    Maintains EMA estimates mean_t / var_t of the gradient-norm history and
    clips when the current norm deviates by more than ``z_thres`` standard
    deviations.  The first ``warmup`` steps are unmodified.

    Important for fair reporting: ZClip does NOT cap the norm at a threshold.
    Once z_t > z_thres it rescales the gradient to a *target* norm
    ``mean_t + (z_thres^2 / z_t) * sigma_t`` (reciprocal clipping, the
    variant its authors report as best).  When the current norm sits far
    below the running mean, that target lies ABOVE it and the update is
    scaled UP.  ``clip_coefficient`` returns the true multiplier, which may
    therefore exceed 1, and ``increased`` reports whether the step was scaled
    up.  Downstream exposure/intensity statistics must be derived from
    whether the raw norm exceeded the target, not from the multiplier, so
    that up-scaling does not masquerade as clipping.
    """

    def __init__(
        self,
        alpha: float = 0.97,
        z_thres: float = 2.5,
        warmup: int = 25,
        eps: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.z_thres = z_thres
        self.warmup = warmup
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self._warmup_norms: list[float] = []
        self.mean = 0.0
        self.var = 0.0
        self.target: float | None = None
        self.increased = False

    def clip_coefficient(self, grad_norm: float) -> float:
        """Multiplier applied to the current gradient (may exceed 1, see class doc)."""
        if self.t < self.warmup or not np.isfinite(grad_norm) or grad_norm <= 0.0:
            self.target = None
            self.increased = False
            return 1.0
        sigma = math.sqrt(max(self.var, 0.0)) + self.eps
        z = (grad_norm - self.mean) / sigma
        if z > self.z_thres:
            self.target = self.mean + (self.z_thres ** 2 / z) * math.sqrt(max(self.var, 0.0))
        else:
            self.target = grad_norm
        self.increased = self.target > grad_norm
        return self.target / (grad_norm + self.eps)

    def update(self, grad_norm: float) -> None:
        """Advance the running statistics. Call once per step, AFTER clipping."""
        if not np.isfinite(grad_norm):
            return
        self.t += 1
        if self.t <= self.warmup:
            self._warmup_norms.append(grad_norm)
            if self.t == self.warmup:
                arr = np.asarray(self._warmup_norms, dtype=float)
                self.mean = float(arr.mean())
                self.var = float(arr.var())
            return
        # Statistics are updated with the clipped norm when a spike fired,
        # otherwise with the raw norm (Kumar et al., Eq. 10).
        value = self.target if self.target is not None else grad_norm
        prev_mean = self.mean
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * value
        self.var = self.alpha * self.var + (1.0 - self.alpha) * (value - prev_mean) ** 2


class AdaGC:
    """AdaGC: adaptive gradient clipping on local gradient norms.

    Wang, Li, Chen, Zeng, Yang, Sun, Ma, Yu & Shen, arXiv:2502.11034 (2025).

    Clips each parameter/unit independently against an EMA of its own past
    gradient norm: h = min(1, lambda_rel * gamma_{t-1} / ||g_t||) with
    gamma_t = beta * gamma_{t-1} + (1 - beta) * ||g_t||.  The warm-up phase
    (``t_start`` steps) instead bounds each unit by lambda_abs, and tracks
    gamma as the running MINIMUM of the warm-up norms, so warm-up outliers
    cannot inflate the thresholds used later.
    """

    def __init__(
        self,
        lambda_rel: float = 0.5,
        beta: float = 0.999,
        t_start: int = 100,
        lambda_abs: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        self.lambda_rel = lambda_rel
        self.beta = beta
        self.t_start = t_start
        self.lambda_abs = lambda_abs
        self.eps = eps

    @staticmethod
    def _reduce(x: Tensor) -> Tensor:
        """Flat per-unit norm: whole tensor for ndim <= 1, else per output channel."""
        if x.ndim <= 1:
            return x.norm().reshape(())
        return x.flatten(start_dim=1).norm(dim=1)

    def step(self, parameters, t: int) -> None:
        """Clip gradients in place at global step ``t`` (1-based)."""
        delta = 1.0 - self.beta
        for p in parameters:
            if p.grad is None:
                continue
            g_norm = self._reduce(p.grad)
            if t <= self.t_start:
                p_norm = self._reduce(p.detach()).clamp_min(self.eps)
                scale = (self.lambda_abs * p_norm / g_norm.clamp_min(self.eps)).clamp(max=1.0)
                p.grad.mul_(scale.reshape(self._broadcast_shape(scale, p.grad)))
                clipped = self._reduce(p.grad)
                if hasattr(p, "_adagc_gamma"):
                    p._adagc_gamma = torch.minimum(p._adagc_gamma, clipped.detach())
                else:
                    p._adagc_gamma = clipped.detach().clone()
                continue
            if not hasattr(p, "_adagc_gamma"):
                p._adagc_gamma = g_norm.detach().clone()
            gamma = p._adagc_gamma
            scale = (self.lambda_rel * gamma / g_norm.clamp_min(self.eps)).clamp(max=1.0)
            p.grad.mul_(scale.reshape(self._broadcast_shape(scale, p.grad)))
            clipped = self._reduce(p.grad)
            p._adagc_gamma = self.beta * gamma + delta * clipped.detach()

    @staticmethod
    def _broadcast_shape(scale: Tensor, grad: Tensor) -> tuple:
        """Reshape a per-unit scale so it broadcasts against ``grad``."""
        if grad.ndim <= 1:
            return ()
        return (scale.shape[0],) + (1,) * (grad.ndim - 1)
