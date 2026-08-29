"""Oscillatory and period-2-like dynamics signatures.

- one-step directional correlation C_1(t)
- two-step directional correlation C_2(t)
- two-step recurrence score R_2(t)
- autocorrelation and dominant-frequency estimation (loss, gradient norm)
"""

from __future__ import annotations

import numpy as np


def one_step_alignment(g: np.ndarray) -> np.ndarray:
    """C_1(t) = <g_t, g_{t+1}> / (||g_t|| ||g_{t+1}||).

    Args:
        g: (T, d) array of consecutive gradients (or random projections).

    Returns:
        Array of length T-1.
    """
    num = np.sum(g[:-1] * g[1:], axis=1)
    den = np.linalg.norm(g[:-1], axis=1) * np.linalg.norm(g[1:], axis=1) + 1e-12
    return num / den


def two_step_alignment(g: np.ndarray) -> np.ndarray:
    """C_2(t) = <g_t, g_{t+2}> / (||g_t|| ||g_{t+2}||)."""
    num = np.sum(g[:-2] * g[2:], axis=1)
    den = np.linalg.norm(g[:-2], axis=1) * np.linalg.norm(g[2:], axis=1) + 1e-12
    return num / den


def two_step_recurrence(theta: np.ndarray) -> np.ndarray:
    """R_2(t) = ||theta_{t+2} - theta_t|| / (||theta_{t+1} - theta_t|| + eps).

    Small R_2 with nontrivial one-step motion indicates approximate two-step
    recurrence.
    """
    num = np.linalg.norm(theta[2:] - theta[:-2], axis=1)
    den = np.linalg.norm(theta[1:-1] - theta[:-2], axis=1) + 1e-12
    return num / den


def dominant_frequency(series: np.ndarray, sample_rate: float = 1.0) -> float:
    """Dominant frequency of a time series (e.g., loss, gradient norm)."""
    series = np.asarray(series, dtype=float)
    series = series - series.mean()
    if len(series) < 4:
        return 0.0
    spec = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(len(series), d=1.0 / sample_rate)
    power = np.abs(spec) ** 2
    if power.sum() < 1e-30:
        return 0.0
    return float(freqs[np.argmax(power)])
