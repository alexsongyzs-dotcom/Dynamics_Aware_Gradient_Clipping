"""Offline alternating-dynamics summaries."""

from __future__ import annotations

import numpy as np

from src.oscillation import (
    alternating_spectral_concentration,
    autocorrelation,
    one_step_alignment,
    period_two_like_score,
    two_step_alignment,
)


def oscillation_summary(gradient_sketches: np.ndarray, loss: np.ndarray) -> dict[str, float]:
    c1 = one_step_alignment(gradient_sketches)
    c2 = two_step_alignment(gradient_sketches)
    loss_ac = autocorrelation(loss, max_lag=2)
    return {
        "mean_c1": float(np.mean(c1)) if len(c1) else float("nan"),
        "mean_c2": float(np.mean(c2)) if len(c2) else float("nan"),
        "period_two_like_fraction": period_two_like_score(c1, c2),
        "loss_lag1_autocorrelation": float(loss_ac[1]) if len(loss_ac) > 1 else float("nan"),
        "loss_lag2_autocorrelation": float(loss_ac[2]) if len(loss_ac) > 2 else float("nan"),
        "loss_nyquist_concentration": alternating_spectral_concentration(loss),
    }
