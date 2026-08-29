"""Trajectory and basin-selection measurements.

Controlled paired runs differing only in clipping policy. Distances:
- parameter-space D_theta(t)
- function-space D_f(t) on a probe set
- final-solution comparison: linear mode connectivity barrier, CKA,
  calibration/subgroup differences, disagreement rates.
"""

from __future__ import annotations

import torch
from torch import Tensor


def parameter_distance(theta1: Tensor, theta2: Tensor) -> float:
    """D_theta(t) = ||theta1 - theta2|| / (1 + ||theta1||)."""
    raise NotImplementedError


def function_distance(model1, model2, probe_loader, metric: str = "l2") -> float:
    """D_f(t) averaged over a probe set."""
    raise NotImplementedError


def linear_mode_connectivity_barrier(model1, model2, data_loader, n_steps: int = 10) -> float:
    """Barrier along the linear interpolation between two final solutions."""
    raise NotImplementedError


def cka_similarity(model1, model2, data_loader, layer_ids) -> float:
    """Representation similarity (CKA) between paired networks."""
    raise NotImplementedError
