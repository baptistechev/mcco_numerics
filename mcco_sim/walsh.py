"""Walsh (±1 Fourier) coefficients for S2 (supplementary_numerics_plan.md).

Basis: chi_S(x) = prod_{s in S} (-1)^{x_s}, indexed by the bit mask w of S in the same MSB-first
convention as the strings, so chi_w(x) = (-1)^{popcount(w & x)}. Then f = sum_S a_S chi_S with
a_S = 2^-N sum_x f(x) chi_S(x) = fwht(f)[S] / 2^N, and the empirical coefficient of a sample
s_1..s_n of values g is a_hat_S = (1/n) sum_k g(s_k) chi_S(s_k) = fwht(histogram)[S] / n.

K (the monomials carried by a nearest-neighbour sketch of window k): the supports S that fit in one
window of k consecutive positions, i.e. whose set bits span at most k positions (empty set included).
"""

from __future__ import annotations

import numpy as np

from .instances import fwht


def walsh_coefficients(f: np.ndarray) -> np.ndarray:
    """a_S for every S (index = bit mask of S)."""
    return fwht(f) / f.size


def inverse_walsh(coefficients: np.ndarray) -> np.ndarray:
    """f from its Walsh coefficients (the transform is its own inverse up to 2^N)."""
    return fwht(coefficients)


def k_mask(N: int, k: int) -> np.ndarray:
    """Boolean mask over the 2^N supports: True if the support spans at most k consecutive positions."""
    w = np.arange(2 ** N, dtype=np.int64)
    mask = w == 0
    nonzero = w[1:]
    highest = np.floor(np.log2(nonzero)).astype(np.int64)            # largest bit index
    lowest = np.log2(nonzero & -nonzero).astype(np.int64)            # smallest bit index
    mask[1:] = highest - lowest + 1 <= k
    return mask


def empirical_coefficients(sample: np.ndarray, values: np.ndarray, n: int, N: int) -> np.ndarray:
    """a_hat_S for every S from n draws (``sample``: indexes with nonzero value, ``values``: values)."""
    histogram = np.bincount(np.asarray(sample, dtype=np.int64), weights=np.asarray(values, dtype=float),
                            minlength=2 ** N)
    return fwht(histogram) / n


def truncation(coefficients: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """F_K = sum_{S in K} a_S chi_S, on every x."""
    return inverse_walsh(np.where(mask, coefficients, 0.0))


def sketching_terms(f: np.ndarray, mask: np.ndarray) -> dict:
    """L = sum_{S not in K} |a_S| and ||f - F_K||_inf (L is its bound, Lemma 1)."""
    a = walsh_coefficients(f)
    return {"L": float(np.abs(a[~mask]).sum()), "sup_error": float(np.abs(f - truncation(a, mask)).max()),
            "k": int(mask.sum())}
