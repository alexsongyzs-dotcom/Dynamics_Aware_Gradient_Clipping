"""Functional endpoints for controlled trajectory-selection branches."""

from __future__ import annotations

import pandas as pd

from src.trajectory import (
    cka_similarity,
    expected_calibration_error,
    function_distance,
    linear_mode_connectivity_barrier,
    parameter_distance,
)


def compare_final_solutions(
    reference_model,
    intervention_model,
    probe_loader,
    evaluation_loader,
    cka_layers: list[str],
) -> pd.Series:
    """Compute parameter diagnostics and primary functional endpoints."""

    return pd.Series(
        {
            "parameter_distance": parameter_distance(reference_model, intervention_model),
            "probability_distance": function_distance(
                reference_model, intervention_model, probe_loader, "probability_l2"
            ),
            "heldout_disagreement": function_distance(
                reference_model, intervention_model, evaluation_loader, "disagreement"
            ),
            "cka_similarity": cka_similarity(
                reference_model, intervention_model, probe_loader, cka_layers
            ),
            "mode_connectivity_barrier": linear_mode_connectivity_barrier(
                reference_model, intervention_model, probe_loader
            ),
            "reference_ece": expected_calibration_error(reference_model, evaluation_loader),
            "intervention_ece": expected_calibration_error(intervention_model, evaluation_loader),
        }
    )
