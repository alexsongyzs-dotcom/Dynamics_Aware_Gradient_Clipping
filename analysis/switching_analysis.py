"""Event-aligned clipping-entry and exit analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.switching import event_aligned_statistics


def summarize_event_geometry(
    frame: pd.DataFrame,
    *,
    transition_column: str = "transition",
    metrics: Sequence[str] = (
        "loss",
        "raw_grad_norm",
        "applied_update_norm",
        "gradient_alignment",
    ),
    half_window: int = 20,
) -> pd.DataFrame:
    rows = []
    for transition in ("enter", "exit"):
        events = np.where(frame[transition_column].astype(str).to_numpy() == transition)[0]
        for metric in metrics:
            statistics = event_aligned_statistics(frame[metric].to_numpy(), events, half_window)
            for index, offset in enumerate(range(-half_window, half_window + 1)):
                rows.append(
                    {
                        "transition": transition,
                        "metric": metric,
                        "offset": offset,
                        "mean": statistics["mean"][index],
                        "median": statistics["median"][index],
                        "sem": statistics["sem"][index],
                        "count": statistics["count"][index],
                    }
                )
    return pd.DataFrame(rows)
