"""Optimizer-state contamination and pre/post placement comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.switching import event_aligned_matrix


def event_aligned_optimizer_state(
    frame: pd.DataFrame,
    *,
    event_column: str = "transition",
    enter_value: str = "enter",
    metrics: Sequence[str] = (
        "adam_second_moment_mismatch",
        "momentum_buffer_alignment",
        "proposed_update_norm",
        "applied_update_norm",
    ),
    half_window: int = 20,
) -> pd.DataFrame:
    """Long-form event-aligned optimizer diagnostics."""

    events = np.where(frame[event_column].astype(str).to_numpy() == enter_value)[0]
    rows = []
    offsets = np.arange(-half_window, half_window + 1)
    for metric in metrics:
        matrix = event_aligned_matrix(frame[metric].to_numpy(), events, half_window)
        for offset, values in zip(offsets, matrix.T if len(matrix) else np.empty((len(offsets), 0))):
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "metric": metric,
                    "offset": int(offset),
                    "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                    "median": float(np.median(finite)) if len(finite) else float("nan"),
                    "count": len(finite),
                }
            )
    return pd.DataFrame(rows)


def placement_divergence(
    pre_frame: pd.DataFrame,
    post_frame: pd.DataFrame,
    *,
    step_column: str = "step",
    metrics: Sequence[str] = (
        "adam_second_moment_mismatch",
        "proposed_update_norm",
        "applied_update_norm",
        "loss",
    ),
) -> pd.DataFrame:
    """Align pre-state and post-update branches and compute their differences."""

    pre = pre_frame[[step_column, *metrics]].copy().set_index(step_column)
    post = post_frame[[step_column, *metrics]].copy().set_index(step_column)
    joined = pre.join(post, how="inner", lsuffix="_pre", rsuffix="_post")
    for metric in metrics:
        joined[f"{metric}_difference"] = joined[f"{metric}_pre"] - joined[f"{metric}_post"]
    return joined.reset_index()
