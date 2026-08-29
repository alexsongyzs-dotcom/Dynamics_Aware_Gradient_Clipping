"""Statistical protocol.

- mean +/- 95% CI across seeds
- bootstrap CIs, paired comparisons, effect sizes
- multiple-comparison correction
- no pseudoreplication: training iterations are not independent samples
"""

from __future__ import annotations

import numpy as np


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) of a bootstrap 95% CI."""
    raise NotImplementedError


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison statistics between two matched runs."""
    raise NotImplementedError
