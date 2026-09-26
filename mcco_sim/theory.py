"""Theory checks (section 4): surrogate F, Theorem 1 / Corollary 1 constants, Eq. (6)-(7)."""

from __future__ import annotations

import math

import numpy as np

from .instances import Instance
from .params import Params
from .sketches import StructuredPhi


def theory_quantities(phi, g: np.ndarray, x_star: int | None, maximizers: np.ndarray, N: int,
                      delta: float) -> dict:
    """Surrogate F = 2^-N Phi^T Phi g and the constants of Eq. (6)-(7) for the sampled function g."""
    size = 2 ** N
    y = phi.apply(g)
    F = phi.adjoint(y) / size
    top = F.max()
    tol = 1e-12 * max(abs(top), 1e-300)
    argmax_set = np.flatnonzero(F >= top - tol)
    out = {
        "var_sampled_function": float(g.var()),
        "sup_sampled_function": float(g.max()),
        "F_argmax": int(argmax_set[0]),
        "F_argmax_unique": bool(argmax_set.size == 1),
        "F_argmax_in_maximizers": bool(np.isin(argmax_set[0], maximizers)),
        "sketch_rows": int(phi.m),
    }
    if x_star is None:
        return out
    theta = F[x_star] - F
    others = np.ones(size, dtype=bool)
    others[x_star] = False
    theta_min = float(theta[others].min())
    out["max_preserved"] = bool(theta_min > 0)
    out["theta_min"] = theta_min

    A = phi.weighted_gram(g ** 2)
    q_diag = phi.quad_diag(A) / size                    # E_s[g(s)^2 g^x(s)^2]
    q_star = phi.adjoint(A @ phi.column(x_star)) / size  # E_s[g(s)^2 g^x*(s) g^x(s)]
    variance = q_star[x_star] - 2 * q_star + q_diag - theta ** 2
    sigma2 = float(variance[others].max())
    out["sigma2"] = sigma2

    sup = float(np.abs(g).max())
    # |Delta_x(s) - Theta_x| <= 2 max_s |Delta_x(s)| <= 2 ||g||_inf max_{s,x} |G_{s,x*} - G_{s,x}|
    bounds = {"plan_2m_sup": 2 * phi.m * sup, "valid": 2 * sup * phi.g_range()}
    if isinstance(phi, StructuredPhi):
        out["M_valid_rule"] = "2 (N-k+1) ||g||_inf"
    else:
        out["M_valid_rule"] = "4 ||g||_inf max_x ||phi_x||^2"
        out["max_col_norm_sq"] = phi.max_col_norm_sq()
    out["M"] = bounds
    out["eq6_exponent"] = {}
    out["eq7_n"] = {}
    for label, M in bounds.items():
        if theta_min > 0:
            exponent = theta_min ** 2 / (2 * sigma2 + (2.0 / 3.0) * M * theta_min)
            n7 = (2 * sigma2 / theta_min ** 2 + 2 * M / (3 * theta_min)) * (N * math.log(2) + math.log(1 / delta))
        else:
            exponent, n7 = None, None
        out["eq6_exponent"][label] = exponent        # bound(n) = 2^N exp(-n * exponent)
        out["eq7_n"][label] = n7
    out["eq7_delta"] = delta
    return out


def thresholded(f: np.ndarray, t: float) -> np.ndarray:
    """T_t f: values below t set to 0."""
    return np.where(f >= t, f, 0.0)


def e2_thresholds(params: Params, inst: Instance) -> list[tuple[str, float]]:
    """E2 grid: t = 0, exact percentiles of f, f(x2) and the midpoint of f(x2) and f*."""
    out = [("zero", 0.0)]
    out += [(f"p{p:g}", float(np.percentile(inst.f, p))) for p in params.E2_PERCENTILES]
    out += [("f_x2", inst.f_x2), ("mid_x2_max", 0.5 * (inst.f_x2 + inst.f_star))]
    return out
