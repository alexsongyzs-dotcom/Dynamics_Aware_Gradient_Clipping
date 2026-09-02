"""Alternating and period-2-like diagnostics with explicit noise controls."""

from __future__ import annotations

from typing import Iterable

import numpy as np


EPS = 1e-12


def _rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("expected a (time, feature) array")
    return array


def one_step_alignment(gradients: np.ndarray) -> np.ndarray:
    """``C1(t)`` for full gradients or a fixed random projection."""

    values = _rows(gradients)
    if len(values) < 2:
        return np.empty(0, dtype=float)
    numerator = np.sum(values[:-1] * values[1:], axis=1)
    denominator = np.linalg.norm(values[:-1], axis=1) * np.linalg.norm(values[1:], axis=1)
    return numerator / (denominator + EPS)


def two_step_alignment(gradients: np.ndarray) -> np.ndarray:
    """``C2(t)`` for full gradients or a fixed random projection."""

    values = _rows(gradients)
    if len(values) < 3:
        return np.empty(0, dtype=float)
    numerator = np.sum(values[:-2] * values[2:], axis=1)
    denominator = np.linalg.norm(values[:-2], axis=1) * np.linalg.norm(values[2:], axis=1)
    return numerator / (denominator + EPS)


def two_step_recurrence(parameters: np.ndarray) -> np.ndarray:
    """Scale-normalized approximate two-step recurrence score."""

    values = _rows(parameters)
    if len(values) < 3:
        return np.empty(0, dtype=float)
    numerator = np.linalg.norm(values[2:] - values[:-2], axis=1)
    previous_step = np.linalg.norm(values[1:-1] - values[:-2], axis=1)
    next_step = np.linalg.norm(values[2:] - values[1:-1], axis=1)
    scale = 0.5 * (previous_step + next_step)
    return numerator / (scale + EPS)


def detrend_linear(series: Iterable[float]) -> np.ndarray:
    """Remove a least-squares linear trend before spectral analysis."""

    values = np.asarray(list(series), dtype=float)
    if len(values) < 2:
        return values - np.mean(values) if len(values) else values
    time = np.arange(len(values), dtype=float)
    design = np.column_stack([time, np.ones(len(time))])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def autocorrelation(series: Iterable[float], max_lag: int) -> np.ndarray:
    """Biased autocorrelation of a detrended series for lags ``0..max_lag``."""

    values = detrend_linear(series)
    if len(values) == 0:
        return np.empty(0, dtype=float)
    max_lag = min(int(max_lag), len(values) - 1)
    variance = float(np.dot(values, values))
    if variance <= EPS:
        result = np.zeros(max_lag + 1)
        result[0] = 1.0
        return result
    return np.asarray(
        [float(np.dot(values[: len(values) - lag], values[lag:]) / variance) for lag in range(max_lag + 1)]
    )


def power_spectrum(series: Iterable[float], sample_rate: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One-sided periodogram of a detrended real-valued sequence."""

    values = detrend_linear(series)
    if len(values) < 2:
        return np.empty(0), np.empty(0)
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
    return frequencies, np.abs(spectrum) ** 2


def dominant_frequency(series: Iterable[float], sample_rate: float = 1.0) -> float:
    """Dominant non-zero frequency after detrending."""

    frequencies, power = power_spectrum(series, sample_rate)
    if len(power) <= 1 or float(np.sum(power[1:])) <= EPS:
        return 0.0
    index = 1 + int(np.argmax(power[1:]))
    return float(frequencies[index])


def alternating_spectral_concentration(series: Iterable[float], bandwidth_bins: int = 1) -> float:
    """Fraction of spectral power near the Nyquist (period-2) frequency."""

    _, power = power_spectrum(series)
    if len(power) <= 1:
        return 0.0
    total = float(np.sum(power[1:]))
    if total <= EPS:
        return 0.0
    start = max(1, len(power) - max(1, int(bandwidth_bins)))
    return float(np.sum(power[start:]) / total)


def period_two_like_score(c1: Iterable[float], c2: Iterable[float]) -> float:
    """Share of aligned timestamps satisfying ``C1 < 0`` and ``C2 > 0``."""

    one = np.asarray(list(c1), dtype=float)
    two = np.asarray(list(c2), dtype=float)
    length = min(len(one), len(two))
    if length == 0:
        return 0.0
    return float(np.mean((one[:length] < 0.0) & (two[:length] > 0.0)))


def conditional_oscillation_score(
    score: Iterable[float], clipped_state: Iterable[int]
) -> dict[str, float]:
    """Compare an oscillation score inside and outside clipping episodes."""

    values = np.asarray(list(score), dtype=float)
    states = np.asarray(list(clipped_state), dtype=bool)
    length = min(len(values), len(states))
    values = values[:length]
    states = states[:length]
    clipped = float(np.mean(values[states])) if np.any(states) else float("nan")
    unclipped = float(np.mean(values[~states])) if np.any(~states) else float("nan")
    return {"clipped": clipped, "unclipped": unclipped, "difference": clipped - unclipped}
