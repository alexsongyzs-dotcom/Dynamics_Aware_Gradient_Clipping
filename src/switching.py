"""Switching-event analysis.

A switching crossing occurs when s_t * s_{t+1} < 0, where
s_t = q_t - 1 = ||g_t||/c_t - 1 is the signed switching coordinate.
"""

from __future__ import annotations

import numpy as np


def switching_crossings(s: np.ndarray) -> np.ndarray:
    """Indices t with s_t * s_{t+1} < 0."""
    s = np.asarray(s, dtype=float)
    return np.where(s[:-1] * s[1:] < 0)[0]


def switching_count(s: np.ndarray) -> int:
    """N_switch = sum_t 1{s_t s_{t+1} < 0}."""
    return int(len(switching_crossings(s)))


def event_aligned_statistics(series: np.ndarray, events: np.ndarray, half_window: int = 10) -> list[np.ndarray]:
    """Return event-aligned windows of a series around switching events."""
    series = np.asarray(series, dtype=float)
    out: list[np.ndarray] = []
    for e in events:
        lo = max(0, int(e) - half_window)
        hi = min(len(series), int(e) + half_window + 1)
        out.append(series[lo:hi])
    return out
