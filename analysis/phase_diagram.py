"""Coarse-to-fine learning-rate/clipping phase-diagram aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def classify_regime(
    frame: pd.DataFrame,
    *,
    failure_column: str = "failed",
    loss_column: str = "final_train_loss",
    exposure_column: str = "clipping_frequency",
) -> pd.Series:
    """Transparent descriptive labels; not a learned scientific conclusion."""

    labels = []
    finite_losses = frame[loss_column].replace([np.inf, -np.inf], np.nan)
    high_loss = finite_losses.quantile(0.90)
    for failed, loss, exposure in zip(
        frame[failure_column].astype(bool), finite_losses, frame[exposure_column]
    ):
        if failed or not np.isfinite(loss):
            labels.append("failed")
        elif loss >= high_loss:
            labels.append("unstable_or_slow")
        elif exposure >= 0.80:
            labels.append("high_exposure")
        elif exposure > 0.0:
            labels.append("transition")
        else:
            labels.append("unclipped_or_inactive")
    return pd.Series(labels, index=frame.index, name="descriptive_regime")


def aggregate_phase_cells(
    summaries: pd.DataFrame,
    *,
    cell_columns: Sequence[str] = ("learning_rate", "clipping_threshold"),
    metric_columns: Sequence[str] = (
        "test_accuracy",
        "final_train_loss",
        "clipping_frequency",
        "episode_count",
    ),
) -> pd.DataFrame:
    """Aggregate independent seeds within each phase-diagram cell."""

    required = set(cell_columns) | set(metric_columns) | {"failed"}
    missing = required - set(summaries.columns)
    if missing:
        raise KeyError(f"missing phase-diagram columns: {sorted(missing)}")
    grouped = summaries.groupby(list(cell_columns), dropna=False)
    result = grouped[list(metric_columns)].agg(["mean", "std", "count"])
    result.columns = [f"{metric}_{statistic}" for metric, statistic in result.columns]
    result["failure_rate"] = grouped["failed"].mean()
    return result.reset_index()
