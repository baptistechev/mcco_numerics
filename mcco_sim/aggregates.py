"""Aggregates of the run records (section 4): bootstrap intervals, binomial intervals."""

from __future__ import annotations

import math

import numpy as np


def two_level_bootstrap(outcomes: np.ndarray, n_boot: int, rng: np.random.Generator,
                        chunk: int = 200) -> np.ndarray:
    """Replicates of the ensemble mean under a two-level bootstrap.

    ``outcomes`` has shape (instances, runs, budgets). Each replicate resamples the instances
    with replacement, then the runs within each resampled instance, and averages. The same
    resampling is used for every budget, so a replicate is a whole curve.
    Returns an array of shape (n_boot, budgets).
    """
    outcomes = np.asarray(outcomes, dtype=float)
    n_inst, n_runs, n_budgets = outcomes.shape
    replicates = np.empty((n_boot, n_budgets))
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        inst = rng.integers(n_inst, size=(b, n_inst))
        runs = rng.integers(n_runs, size=(b, n_inst, n_runs))
        sample = outcomes[inst[:, :, None], runs]              # (b, instances, runs, budgets)
        replicates[start:start + b] = sample.mean(axis=(1, 2))
    return replicates


def percentile_interval(values: np.ndarray, confidence: float = 0.95, axis: int = 0) -> tuple:
    """Equal-tailed percentile interval (inf values allowed: they sort last)."""
    alpha = (1 - confidence) / 2
    return (np.quantile(values, alpha, axis=axis, method="lower"),
            np.quantile(values, 1 - alpha, axis=axis, method="higher"))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion."""
    if trials == 0:
        return (math.nan, math.nan)
    z = {0.95: 1.959963984540054}.get(confidence)
    if z is None:
        from scipy.stats import norm
        z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
