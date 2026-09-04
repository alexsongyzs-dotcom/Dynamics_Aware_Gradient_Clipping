"""Sanity tests for dynamical diagnostics."""

import numpy as np

from src.dynamics import (
    ClippingEpisodeTracker,
    EpisodeTransition,
    HysteresisConfig,
    exposure_ema,
    stability_pressure,
)
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


def test_hysteresis_prevents_boundary_chatter() -> None:
    tracker = ClippingEpisodeTracker(
        HysteresisConfig(on_margin=0.05, off_margin=0.05, min_dwell_steps=3)
    )
    observations = [tracker.update(q, step) for step, q in enumerate([1.06, 1.01, 0.94, 0.94])]
    assert observations[0].transition is EpisodeTransition.ENTER
    assert observations[2].clipped_state
    assert observations[3].transition is EpisodeTransition.EXIT
    assert tracker.switch_count == 2
