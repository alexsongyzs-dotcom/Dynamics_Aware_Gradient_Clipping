"""Run-level clipping exposure features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def exposure_summary(
    frame: pd.DataFrame,
    *,
    active_column: str = "active",
    coefficient_column: str = "coefficient",
    early_fraction: float = 0.10,
) -> dict[str, float | int]:
    active = frame[active_column].astype(bool).to_numpy()
    coefficients = frame[coefficient_column].to_numpy(dtype=float)
    starts = np.where(active & ~np.r_[False, active[:-1]])[0]
    stops = np.where(active & ~np.r_[active[1:], False])[0]
    lengths = stops - starts + 1
    early_stop = max(1, int(len(active) * early_fraction))
    return {
        "steps": len(active),
        "clipping_frequency": float(np.mean(active)) if len(active) else 0.0,
        "mean_intensity": float(np.mean(1.0 - coefficients)) if len(active) else 0.0,
        "first_exposure_step": int(starts[0]) if len(starts) else -1,
        "episode_count": len(starts),
        "mean_episode_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "max_episode_length": int(np.max(lengths)) if len(lengths) else 0,
        "early_exposure": float(np.mean(active[:early_stop])) if len(active) else 0.0,
    }


def summarize_runs(frame: pd.DataFrame, run_column: str = "run_id") -> pd.DataFrame:
    rows = []
    for run_id, group in frame.groupby(run_column, sort=False):
        rows.append({run_column: run_id, **exposure_summary(group)})
    return pd.DataFrame(rows)
