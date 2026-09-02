"""Run-level and paired-branch statistical utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.estimate, self.lower, self.upper


def bootstrap_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
    statistic=np.mean,
) -> BootstrapInterval:
    """Percentile bootstrap CI over independent run-level observations."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    estimates = np.asarray([statistic(array[index]) for index in indices], dtype=float)
    alpha = 1.0 - confidence
    return BootstrapInterval(
        estimate=float(statistic(array)),
        lower=float(np.quantile(estimates, alpha / 2.0)),
        upper=float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        n=len(array),
    )


def mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    return bootstrap_interval(values, confidence, samples, seed).as_tuple()


def paired_permutation_pvalue(differences: np.ndarray, permutations: int = 50_000, seed: int = 0) -> float:
    """Two-sided sign-flip randomization test for paired differences."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(int(permutations)):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        extreme += abs(float(np.mean(values * signs))) >= observed
    return (extreme + 1.0) / (permutations + 1.0)


def paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Paired effect, CI, standardized effect, and randomization p-value."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("paired arrays must have equal shape")
    valid = np.isfinite(first) & np.isfinite(second)
    differences = first[valid] - second[valid]
    interval = bootstrap_interval(differences, confidence, samples, seed)
    standard_deviation = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
    standardized = interval.estimate / standard_deviation if standard_deviation > 0.0 else float("nan")
    return {
        "n_pairs": len(differences),
        "mean_difference": interval.estimate,
        "ci_lower": interval.lower,
        "ci_upper": interval.upper,
        "standardized_effect": standardized,
        "permutation_pvalue": paired_permutation_pvalue(differences, seed=seed),
    }


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""

    values = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full_like(values, np.nan)
    valid_indices = np.where(np.isfinite(values))[0]
    if len(valid_indices) == 0:
        return adjusted
    order = valid_indices[np.argsort(values[valid_indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def failure_rate(failed: Iterable[bool]) -> BootstrapInterval:
    """Bootstrap interval for independent run-level failure indicators."""

    return bootstrap_interval((float(value) for value in failed), statistic=np.mean)
