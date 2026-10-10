"""Seed-level inference. Callers must align paired arrays by condition and seed."""
from __future__ import annotations
import itertools
import numpy as np
from scipy import stats


def _sample(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) < 2 or not np.isfinite(arr).all():
        raise ValueError('At least two finite independent seed-level values are required')
    return arr


def mean_ci(values, confidence=.95):
    """Mean and two-sided Student-t interval over independent training seeds."""
    a = _sample(values)
    if not 0 < confidence < 1:
        raise ValueError('Confidence must be in (0,1)')
    mean = float(a.mean())
    half = float(stats.t.ppf((1+confidence)/2, len(a)-1) * stats.sem(a))
    return mean, mean-half, mean+half


def paired_difference(a, b, equivalence_margin=None):
    """Paired t interval and exact two-sided sign-flip p-value (up to 20 pairs).

    The sign-flip test assumes exchangeability under the null; it is not an
    assumption-free population test. Zero differences remain in the enumeration.
    """
    a, b = _sample(a), _sample(b)
    if a.shape != b.shape:
        raise ValueError('Pairs must have equal shape')
    d = a-b
    mean, lo, hi = mean_ci(d)
    if len(d) > 20:
        raise ValueError('Exact enumeration is limited to 20 pairs')
    observed = abs(d.mean())
    exceed = sum(abs(np.dot(signs,d)/len(d)) >= observed-1e-12
                 for signs in itertools.product((-1,1), repeat=len(d)))
    out = {'n':len(d), 'mean':mean, 'lower':lo, 'upper':hi,
           'sign_flip_p':exceed / 2**len(d), 'wins':int((d>0).sum()),
           'ties':int((d==0).sum()), 'losses':int((d<0).sum())}
    if equivalence_margin is not None:
        _, elo, ehi = mean_ci(d, .90)
        out.update(equivalence_margin=equivalence_margin, lower90=elo, upper90=ehi,
                   equivalent=bool(elo > -equivalence_margin and ehi < equivalence_margin))
    return out


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite p-values in [0,1] required')
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p),0,-1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result
