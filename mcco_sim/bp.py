"""Basis pursuit decoder for S1 (supplementary_numerics_plan.md): nonnegative basis pursuit denoising

    min  sum(z)   s.t.  z >= 0,  ||Phi z - y||_2 <= eta,

on the 2^N entries of z, with only Phi and Phi^T products (StructuredPhi / DensePhi of sketches.py,
whose row order is TrOMA's). Solved by Chambolle-Pock with a fixed number of iterations.
"""

from __future__ import annotations

import math

import numpy as np


def noise_level(phi, sample: np.ndarray, values: np.ndarray, n: int, size: int) -> float:
    """eta = expected l2 norm of the Monte-Carlo noise of y = sum_s g(s) phi_s over the n draws.

    y_j is a sum of n i.i.d. terms g(s) phi_j(s), so its variance is n Var_s[g phi_j], estimated from
    the same draws: sum_s (g(s) phi_j(s))^2 - y_j^2 / n. ``sample`` holds the indexes of the draws
    with nonzero thresholded value g (zeros contribute nothing), ``values`` those values.
    """
    sample = np.asarray(sample, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if sample.size == 0:
        return 0.0
    if hasattr(phi, "M"):                             # dense: (Phi o Phi) on the sampled columns
        unique, inverse = np.unique(sample, return_inverse=True)
        g1 = np.bincount(inverse, weights=values, minlength=unique.size)
        g2 = np.bincount(inverse, weights=values ** 2, minlength=unique.size)
        columns = phi.M[:, unique]
        y = columns @ g1
        second = (columns ** 2) @ g2
    else:                                             # 0/1 structured: phi_j^2 = phi_j
        y = phi.apply(np.bincount(sample, weights=values, minlength=size))
        second = phi.apply(np.bincount(sample, weights=values ** 2, minlength=size))
    return float(math.sqrt(max(0.0, float(np.sum(second - y ** 2 / n)))))


def spectral_norm(phi, size: int, iterations: int = 50, seed: int = 0) -> float:
    """||Phi||_2 by power iteration on Phi^T Phi (slightly overestimated for a safe step size)."""
    v = np.random.default_rng(seed).standard_normal(size)
    v /= np.linalg.norm(v)
    value = 0.0
    for _ in range(iterations):
        w = phi.adjoint(phi.apply(v))
        value = float(np.linalg.norm(w))
        v = w / value
    return math.sqrt(value) * 1.01


def bp_nonneg(phi, y: np.ndarray, eta: float, size: int, iterations: int, norm: float) -> tuple[np.ndarray, dict]:
    """Chambolle-Pock (theta = 1, tau = sigma = 0.99 / ||Phi||) for min sum(z), z >= 0,
    ||Phi z - y|| <= eta. Returns z and diagnostics of the last iterate."""
    y = np.asarray(y, dtype=float)
    tau = sigma = 0.99 / norm
    z = np.zeros(size)
    z_bar = z.copy()
    u = np.zeros_like(y)
    for _ in range(iterations):
        # Dual step: prox of sigma F*, F = indicator of the ball B(y, eta), via Moreau.
        v = u + sigma * phi.apply(z_bar)
        w = v / sigma - y
        norm_w = np.linalg.norm(w)
        projected = y + (w if norm_w <= eta else w * (eta / norm_w))
        u = v - sigma * projected
        # Primal step: prox of tau (sum + indicator of z >= 0).
        z_new = np.maximum(z - tau * phi.adjoint(u) - tau, 0.0)
        z_bar = 2 * z_new - z
        z = z_new
    residual = float(np.linalg.norm(phi.apply(z) - y))
    return z, {"bp_residual": residual, "bp_residual_over_eta": residual / eta if eta > 0 else math.inf,
               "bp_l1": float(z.sum()), "bp_positive": int(np.count_nonzero(z > 0)),
               "bp_iterations": int(iterations), "bp_norm": norm}


def top_entries(z: np.ndarray, count: int) -> list[int]:
    """Indexes of the ``count`` largest positive entries of z, largest first."""
    positive = np.flatnonzero(z > 0)
    if positive.size == 0:
        return []
    order = positive[np.argsort(-z[positive], kind="stable")]
    return [int(x) for x in order[:count]]
