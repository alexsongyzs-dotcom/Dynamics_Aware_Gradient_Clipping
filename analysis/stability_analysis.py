"""Scope-aware stability-proxy summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_stability_pressure(
    hessian_records: pd.DataFrame,
    learning_rates: pd.DataFrame,
    *,
    optimizer_name: str,
) -> pd.DataFrame:
    """Join checkpoint curvature to learning rate and label interpretation scope."""

    merged = hessian_records.merge(learning_rates[["step", "learning_rate"]], on="step", how="left")
    merged["stability_pressure"] = merged["learning_rate"] * merged["top_eigenvalue"]
    merged["classical_boundary_applicable"] = optimizer_name.lower() == "sgd_no_momentum"
    return merged


def excursion_summary(pressure: np.ndarray, boundary: float = 2.0) -> dict[str, float | int]:
    values = np.asarray(pressure, dtype=float)
    above = values > boundary
    return {
        "observations": len(values),
        "maximum_pressure": float(np.nanmax(values)) if len(values) else float("nan"),
        "fraction_above_boundary": float(np.nanmean(above)) if len(values) else float("nan"),
        "excursion_count": int(np.sum(above & ~np.r_[False, above[:-1]])) if len(values) else 0,
    }
