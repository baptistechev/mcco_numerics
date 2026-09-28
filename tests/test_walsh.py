"""Walsh coefficients, the monomial set K of the structured sketches, and the S2 quantities."""

import numpy as np

from mcco_sim.walsh import empirical_coefficients, k_mask, sketching_terms, truncation, walsh_coefficients


def _chi(w, N):
    x = np.arange(2 ** N)
    return (-1.0) ** np.array([bin(w & xx).count("1") for xx in x])


def test_k_mask_counts():
    N = 8
    for k in (1, 3, 4, 5):
        explicit = 0
        for w in range(2 ** N):
            positions = [N - 1 - b for b in range(N) if w >> b & 1]
            explicit += not positions or max(positions) - min(positions) + 1 <= k
        assert k_mask(N, k).sum() == explicit


def test_coefficients_are_means():
    N, rng = 8, np.random.default_rng(0)
    f = rng.random(2 ** N)
    a = walsh_coefficients(f)
    sample = rng.integers(0, 2 ** N, 400)
    a_hat = empirical_coefficients(sample, f[sample], 400, N)
    for w in (0, 3, 37, 200):
        assert np.isclose(a[w], (f * _chi(w, N)).mean())
        assert np.isclose(a_hat[w], (f[sample] * _chi(w, N)[sample]).mean())
    assert np.allclose(truncation(a, np.ones(2 ** N, bool)), f)


def test_function_in_span_of_K_has_no_sketching_error():
    """A sum of functions of 4 consecutive bits lies in the span of K (k = 4): L = 0 and F_K = f."""
    N, rng = 8, np.random.default_rng(1)
    x = np.arange(2 ** N)
    f = np.zeros(2 ** N)
    for i in range(N - 3):
        table = rng.random(16)
        f += table[(x >> (N - i - 4)) & 15]
    terms = sketching_terms(f, k_mask(N, 4))
    assert terms["L"] < 1e-9 and terms["sup_error"] < 1e-9
    assert sketching_terms(f, k_mask(N, 3))["L"] > 1e-3            # not in the span of window-3 functions
