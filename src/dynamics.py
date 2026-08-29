"""Dynamical diagnostics for neural network training.

Implements the observables defined in the outline:
- stability pressure S_t = eta_t * lambda_max(H_t)
- clipping coordinate q_t = ||g_t|| / c_t, signed coordinate s_t = q_t - 1
- exposure EMA E_t
- gradient alignment a_t and curvature proxy kappa_hat_t
"""

from __future__ import annotations

import torch
from torch import Tensor


def stability_pressure(eta: float, top_eigenvalue: float) -> float:
    """S_t = eta_t * lambda_max(H_t)."""
    return eta * top_eigenvalue


def clipping_coordinate(grad_norm: Tensor, threshold: float) -> Tensor:
    """q_t = ||g_t|| / c_t."""
    return grad_norm / (threshold + 1e-12)


def gradient_alignment(g_t: Tensor, g_prev: Tensor) -> float:
    """a_t = <g_t, g_{t-1}> / (||g_t|| ||g_{t-1}||)."""
    num = torch.dot(g_t, g_prev)
    den = torch.norm(g_t) * torch.norm(g_prev) + 1e-12
    return float(num / den)


def curvature_proxy(g_t: Tensor, g_prev: Tensor, theta_t: Tensor, theta_prev: Tensor) -> float:
    """kappa_hat_t = ||g_t - g_{t-1}|| / (||theta_t - theta_{t-1}|| + eps)."""
    num = torch.norm(g_t - g_prev)
    den = torch.norm(theta_t - theta_prev) + 1e-12
    return float(num / den)


def exposure_ema(e_t: float, prev: float, beta: float) -> float:
    """E_t = beta * E_{t-1} + (1 - beta) * e_t."""
    return beta * prev + (1.0 - beta) * e_t
