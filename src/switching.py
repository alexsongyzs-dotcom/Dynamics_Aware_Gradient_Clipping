"""Robust clipping-episode extraction and event-aligned summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.dynamics import ClippingEpisodeTracker, EpisodeTransition, HysteresisConfig


@dataclass(frozen=True)
class Episode:
    episode_id: int
    start: int
    stop: int
    length: int
    maximum_q: float
    mean_intensity: float


def switching_crossings(signed_coordinate: np.ndarray) -> np.ndarray:
    """Legacy zero-crossings; use :func:`detect_episodes` for main results."""

    values = np.asarray(signed_coordinate, dtype=float)
    if values.size < 2:
        return np.empty(0, dtype=int)
    return np.where(values[:-1] * values[1:] < 0)[0]


def switching_count(signed_coordinate: np.ndarray) -> int:
    return int(len(switching_crossings(signed_coordinate)))


def detect_episodes(
    q: Iterable[float],
    config: HysteresisConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Episode]]:
    """Detect persistent clipped episodes from a clipping-coordinate series.

    Returns state, transition-code (1 enter, -1 exit, 0 otherwise), and
    completed episodes. An episode still active at the final step is closed at
    the end of the observed sequence.
    """

    values = np.asarray(list(q), dtype=float)
    tracker = ClippingEpisodeTracker(config)
    states = np.zeros(len(values), dtype=np.int8)
    transitions = np.zeros(len(values), dtype=np.int8)
    episodes: list[Episode] = []
    start: int | None = None

    for step, value in enumerate(values):
        observation = tracker.update(float(value), step)
        states[step] = int(observation.clipped_state)
        if observation.transition is EpisodeTransition.ENTER:
            transitions[step] = 1
            start = step
        elif observation.transition is EpisodeTransition.EXIT:
            transitions[step] = -1
            if start is not None:
                segment = values[start:step]
                episodes.append(
                    Episode(
                        episode_id=observation.episode_id,
                        start=start,
                        stop=step - 1,
                        length=step - start,
                        maximum_q=float(np.max(segment)),
                        mean_intensity=float(np.mean(np.clip(1.0 - 1.0 / segment, 0.0, 1.0))),
                    )
                )
                start = None

    if start is not None:
        segment = values[start:]
        episodes.append(
            Episode(
                episode_id=tracker.episode_id,
                start=start,
                stop=len(values) - 1,
                length=len(values) - start,
                maximum_q=float(np.max(segment)),
                mean_intensity=float(np.mean(np.clip(1.0 - 1.0 / segment, 0.0, 1.0))),
            )
        )
    return states, transitions, episodes


def event_aligned_matrix(
    series: Iterable[float], events: Iterable[int], half_window: int = 10
) -> np.ndarray:
    """Return fixed-width, NaN-padded event-aligned windows."""

    values = np.asarray(list(series), dtype=float)
    event_indices = np.asarray(list(events), dtype=int)
    width = 2 * half_window + 1
    matrix = np.full((len(event_indices), width), np.nan, dtype=float)
    for row, event in enumerate(event_indices):
        source_start = max(0, event - half_window)
        source_stop = min(len(values), event + half_window + 1)
        target_start = source_start - (event - half_window)
        target_stop = target_start + source_stop - source_start
        matrix[row, target_start:target_stop] = values[source_start:source_stop]
    return matrix


def event_aligned_statistics(
    series: Iterable[float], events: Iterable[int], half_window: int = 10
) -> dict[str, np.ndarray]:
    """Mean, median, count and standard error for aligned event windows."""

    matrix = event_aligned_matrix(series, events, half_window)
    if matrix.shape[0] == 0:
        width = 2 * half_window + 1
        empty = np.full(width, np.nan)
        return {"mean": empty, "median": empty.copy(), "sem": empty.copy(), "count": np.zeros(width)}
    count = np.sum(np.isfinite(matrix), axis=0)
    standard_deviation = np.nanstd(matrix, axis=0, ddof=1)
    sem = standard_deviation / np.sqrt(np.maximum(count, 1))
    return {
        "mean": np.nanmean(matrix, axis=0),
        "median": np.nanmedian(matrix, axis=0),
        "sem": sem,
        "count": count,
    }


def circular_shift_surrogate(values: Iterable[float], seed: int) -> np.ndarray:
    """Preserve autocorrelation but break alignment to another time series."""

    array = np.asarray(list(values), dtype=float)
    if len(array) < 2:
        return array.copy()
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, len(array)))
    return np.roll(array, shift)


def block_shuffle_surrogate(values: Iterable[float], seed: int, block_size: int) -> np.ndarray:
    """Shuffle blocks while preserving within-block short-range structure."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    array = np.asarray(list(values), dtype=float)
    blocks = [array[start : start + block_size].copy() for start in range(0, len(array), block_size)]
    np.random.default_rng(seed).shuffle(blocks)
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=float)
