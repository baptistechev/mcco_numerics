"""Bootstrap, n_90 and binomial intervals of mcco_sim/aggregates.py."""

import numpy as np
import pytest

from mcco_sim.aggregates import percentile_interval, two_level_bootstrap, wilson_interval


def test_bootstrap_constant_outcomes():
    ones = np.ones((5, 4, 3))
    reps = two_level_bootstrap(ones, 50, np.random.default_rng(0))
    assert reps.shape == (50, 3) and np.all(reps == 1)


def test_bootstrap_centered_and_reproducible():
    rng = np.random.default_rng(1)
    outcomes = rng.random((40, 10, 2)) < np.array([0.2, 0.7])
    reps = two_level_bootstrap(outcomes, 2000, np.random.default_rng(2))
    assert np.allclose(reps.mean(axis=0), outcomes.mean(axis=(0, 1)), atol=0.01)
    again = two_level_bootstrap(outcomes, 2000, np.random.default_rng(2))
    assert np.array_equal(reps, again)


def test_bootstrap_resamples_instances():
    # Instance-level heterogeneity must widen the interval: half the instances always succeed.
    outcomes = np.zeros((20, 10, 1))
    outcomes[:10] = 1
    reps = two_level_bootstrap(outcomes, 2000, np.random.default_rng(3))
    low, high = percentile_interval(reps[:, 0])
    assert low < 0.4 and high > 0.6


def test_wilson_interval():
    low, high = wilson_interval(0, 300)
    assert low == pytest.approx(0, abs=1e-12) and 0 < high < 0.02
    low, high = wilson_interval(150, 300)
    assert low == pytest.approx(0.4437, abs=1e-3) and high == pytest.approx(0.5563, abs=1e-3)
    assert np.isnan(wilson_interval(0, 0)[0])
