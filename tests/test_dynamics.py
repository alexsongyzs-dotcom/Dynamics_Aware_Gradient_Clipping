"""Sanity tests for dynamical diagnostics."""

import numpy as np

from src.dynamics import exposure_ema, stability_pressure
from src.oscillation import two_step_recurrence


def test_stability_pressure() -> None:
    assert stability_pressure(0.1, 20.0) == 2.0


def test_exposure_ema_converges() -> None:
    e = 0.0
    for _ in range(1000):
        e = exposure_ema(1.0, e, beta=0.9)
    assert abs(e - 1.0) < 1e-3


def test_recurrence_period_two() -> None:
    theta = np.array([[0.0], [1.0], [0.0], [1.0], [0.0]])
    r = two_step_recurrence(theta)
    assert r[0] < 1e-6  # theta_{t+2} == theta_t
