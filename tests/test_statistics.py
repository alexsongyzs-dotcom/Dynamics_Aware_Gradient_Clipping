import numpy as np
import pytest
from analysis.statistics import mean_ci, paired_difference, holm_adjust


def test_paired_exact_test_and_equivalence():
    r = paired_difference(np.ones(8), np.zeros(8), .5)
    assert r['sign_flip_p'] == 2/256
    assert r['lower'] == r['upper'] == 1
    assert not r['equivalent']
    zero = paired_difference(np.zeros(8), np.zeros(8), .5)
    assert zero['sign_flip_p'] == 1 and zero['equivalent']


def test_intervals_and_multiplicity():
    mean, lo, hi = mean_ci(np.arange(8))
    assert mean == 3.5 and lo < mean < hi
    assert np.allclose(holm_adjust([.03,.001,.04]), [.06,.003,.06])
    with pytest.raises(ValueError):
        mean_ci([0,np.nan])
