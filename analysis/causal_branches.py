"""Paired checkpoint-branch estimands for timing and placement interventions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analysis.statistics import paired_difference


def validate_branch_table(
    frame: pd.DataFrame,
    *,
    intervention_column: str = "intervention",
    pair_columns: Sequence[str] = ("seed", "checkpoint_id"),
) -> None:
    """Require exactly one row per pair/intervention cell."""

    required = set(pair_columns) | {intervention_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing branch columns: {sorted(missing)}")
    counts = frame.groupby([*pair_columns, intervention_column]).size()
    if (counts != 1).any():
        raise ValueError("branch table contains duplicate pair/intervention cells")


def paired_branch_effects(
    frame: pd.DataFrame,
    *,
    outcomes: Sequence[str],
    reference: str,
    intervention_column: str = "intervention",
    pair_columns: Sequence[str] = ("seed", "checkpoint_id"),
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate every intervention minus a preregistered reference."""

    validate_branch_table(
        frame,
        intervention_column=intervention_column,
        pair_columns=pair_columns,
    )
    interventions = sorted(set(frame[intervention_column]) - {reference})
    rows = []
    for outcome in outcomes:
        if outcome not in frame:
            raise KeyError(outcome)
        wide = frame.pivot(index=list(pair_columns), columns=intervention_column, values=outcome)
        if reference not in wide:
            raise ValueError(f"reference intervention is absent: {reference}")
        for intervention in interventions:
            if intervention not in wide:
                continue
            complete = wide[[intervention, reference]].dropna()
            result = paired_difference(
                complete[intervention].to_numpy(),
                complete[reference].to_numpy(),
                seed=seed,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "intervention": intervention,
                    "reference": reference,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def verify_gain_invariants(
    reference: Sequence[float],
    intervention: Sequence[float],
    tolerance: float = 1e-10,
) -> dict[str, bool | float]:
    """Check marginal-gain invariants before a timing intervention is run."""

    first = np.asarray(reference, dtype=float)
    second = np.asarray(intervention, dtype=float)
    same_length = len(first) == len(second)
    if not same_length:
        return {
            "same_length": False,
            "same_multiset": False,
            "same_sum": False,
            "same_active_count": False,
            "maximum_sorted_difference": float("inf"),
        }
    maximum_difference = float(np.max(np.abs(np.sort(first) - np.sort(second)))) if len(first) else 0.0
    return {
        "same_length": True,
        "same_multiset": maximum_difference <= tolerance,
        "same_sum": abs(float(np.sum(first) - np.sum(second))) <= tolerance,
        "same_active_count": int(np.sum(first < 1.0)) == int(np.sum(second < 1.0)),
        "maximum_sorted_difference": maximum_difference,
    }
