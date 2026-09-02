"""Offline event detection, sensitivity grids, and surrogate comparisons."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from src.dynamics import HysteresisConfig
from src.switching import block_shuffle_surrogate, detect_episodes


def annotate_events(
    frame: pd.DataFrame,
    *,
    q_column: str = "clipping_coordinate",
    config: HysteresisConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add event state/transition columns and return an episode table."""

    state, transition, episodes = detect_episodes(frame[q_column].to_numpy(), config)
    annotated = frame.copy()
    annotated["episode_state_offline"] = state
    annotated["episode_transition_offline"] = transition
    episode_frame = pd.DataFrame([asdict(episode) for episode in episodes])
    return annotated, episode_frame


def event_sensitivity_grid(
    frame: pd.DataFrame,
    on_margins: Iterable[float],
    off_margins: Iterable[float],
    dwell_steps: Iterable[int],
    q_column: str = "clipping_coordinate",
) -> pd.DataFrame:
    """Recompute episode counts over a preregistered detector grid."""

    rows = []
    q = frame[q_column].to_numpy()
    for on, off, dwell in product(on_margins, off_margins, dwell_steps):
        config = HysteresisConfig(float(on), float(off), int(dwell))
        state, transition, episodes = detect_episodes(q, config)
        rows.append(
            {
                "on_margin": on,
                "off_margin": off,
                "min_dwell_steps": dwell,
                "episode_count": len(episodes),
                "transition_count": int(np.sum(np.abs(transition))),
                "duty_cycle": float(np.mean(state)) if len(state) else 0.0,
                "mean_episode_length": (
                    float(np.mean([episode.length for episode in episodes])) if episodes else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def surrogate_episode_counts(
    q: Iterable[float],
    config: HysteresisConfig,
    seeds: Iterable[int],
    block_size: int = 20,
) -> pd.DataFrame:
    """Null distribution from block-shuffled clipping coordinates."""

    values = np.asarray(list(q), dtype=float)
    rows = []
    for seed in seeds:
        surrogate = block_shuffle_surrogate(values, int(seed), block_size)
        state, transition, episodes = detect_episodes(surrogate, config)
        rows.append(
            {
                "seed": int(seed),
                "episode_count": len(episodes),
                "transition_count": int(np.sum(np.abs(transition))),
                "duty_cycle": float(np.mean(state)) if len(state) else 0.0,
            }
        )
    return pd.DataFrame(rows)
