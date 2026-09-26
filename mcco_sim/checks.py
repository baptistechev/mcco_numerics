"""Correctness checks, run by pytest (tests/test_checks.py) and by the pilot on the target machine.

Each check function returns a list of {"check", "passed", "detail"} results.
"""

from __future__ import annotations

import numpy as np
from troma import ConstraintSketchMap, DitString

from .annealing import da_runs
from .instances import Instance, direct_cost, instance_descriptor
from .mcco import mcco_sample
from .params import Params
from .sketches import SketchSet, StructuredPhi
from .theory import theory_quantities, thresholded


def _result(name: str, passed, detail: str = "") -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _small(params: Params) -> Params:
    """N = 8 version of the params for the exhaustive checks."""
    return params.replace(N=8, N_MIN=50, N_MAX=800, J=2)


def _strip_timing(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "time"} for r in records]


def check_spectrum(params: Params) -> list[dict]:
    """Vectorized spectrum vs per-x direct evaluation (all x at N = 8, random x at N)."""
    results = []
    for N, n_check in ((8, None), (params.N, 300)):
        p = params.replace(N=N)
        for family in ("L", "W"):
            inst = Instance(p, instance_descriptor(p, "e1", family, 5, 0))
            xs = np.arange(2 ** N) if n_check is None else np.random.default_rng(0).integers(0, 2 ** N, n_check)
            ref = np.array([direct_cost(np.array(DitString.from_integer(int(x), N).tolist()), inst.rules)
                            for x in xs])
            results.append(_result(f"spectrum N={N} {family}", np.allclose(inst.f[xs], ref, rtol=0, atol=1e-12)))
    return results


def check_structured_phi(params: Params) -> list[dict]:
    """Structured Phi: row order, adjoint and columns identical to troma's ConstraintSketchMap."""
    results = []
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 2 ** 8, 500)
    vals = rng.random(500)
    for k in (4, 5):
        sm = ConstraintSketchMap(sketch_length=8, interaction_size=k, constraints="nearest_neighbors")
        phi = StructuredPhi(8, k)
        troma_y = np.array(sm.compute_marginal(DitString.from_integers(idx, 8), list(vals)))
        dense = np.bincount(idx, weights=vals, minlength=2 ** 8)
        results.append(_result(f"structured Phi apply k={k}", np.allclose(phi.apply(dense), troma_y)))
        cols = np.array([sm.reconstruct_structured_matrix_column(x) for x in range(2 ** 8)]).T
        y = rng.random(phi.m)
        results.append(_result(f"structured Phi adjoint k={k}", np.allclose(phi.adjoint(y), cols.T @ y)))
        results.append(_result(f"structured Phi column k={k}", np.allclose(phi.column(37), cols[:, 37])))
    return results


def check_theory_brute_force(params: Params) -> list[dict]:
    """Theta_min, sigma^2 and the M bounds vs explicit Delta_x(s) over all (x, s) at N = 8."""
    small = _small(params)
    results = []
    for fam in ("L", "W"):
        for draw in range(20):
            inst = Instance(small, instance_descriptor(small, "e1", fam, 3, draw))
            if inst.unique:
                break
        sketches = SketchSet(small, inst)
        for name in sketches.names:
            phi = sketches.phis[name]
            Phi = np.array([phi.column(x) for x in range(2 ** 8)]).T
            G = Phi.T @ Phi
            for t in (0.0, inst.q_exact):
                g = thresholded(inst.f, t)
                fast = theory_quantities(phi, g, inst.x_star, inst.maximizers, 8, 0.1)
                delta_x = g[:, None] * (G[:, [inst.x_star]] - G)          # Delta_x(s), rows s
                theta = delta_x.mean(axis=0)
                var = (delta_x ** 2).mean(axis=0) - theta ** 2
                mask = np.arange(2 ** 8) != inst.x_star
                M_true = np.abs(delta_x - theta)[:, mask].max()
                ok = (np.isclose(fast["theta_min"], theta[mask].min(), rtol=1e-9, atol=1e-12)
                      and np.isclose(fast["sigma2"], var[mask].max(), rtol=1e-8, atol=1e-12)
                      and M_true <= fast["M"]["valid"] * (1 + 1e-12)
                      and (phi.kind == "dense" or M_true <= fast["M"]["plan_2m_sup"]))
                results.append(_result(
                    f"theory brute force {fam} {name} t={t:.3g}", ok,
                    f"theta {fast['theta_min']:.4g} sigma2 {fast['sigma2']:.4g} "
                    f"M {M_true:.3g} <= {fast['M']['valid']:.3g}"))
    return results


def check_mcco(params: Params) -> list[dict]:
    """Problem II argmax = first MP candidate, query counts, reproducibility from the seeds."""
    small = _small(params)
    inst = Instance(small, instance_descriptor(small, "e1", "L", 4, 1))
    recs = mcco_sample(small, inst, SketchSet(small, inst), 0, small.budgets, [{"mode": "adaptive"}])
    # candidates keep MP's order, so candidates[0] is MP's first pick (the Problem II argmax).
    agree = [r["candidates"][0][0] == r["problem2_argmax"] for r in recs
             if not r["problem2_tie"] and r["candidates"]]
    again = mcco_sample(small, inst, SketchSet(small, inst), 0, small.budgets, [{"mode": "adaptive"}])
    return [
        _result("problem II argmax == first MP candidate", all(agree), f"{sum(agree)}/{len(agree)}"),
        _result("MCCO queries = n + new candidates",
                all(r["n"] <= r["queries"] <= r["n"] + small.MP_ITERATIONS for r in recs)),
        _result("MCCO reproducible from recorded seeds", _strip_timing(recs) == _strip_timing(again)),
    ]


def check_annealing(params: Params) -> list[dict]:
    """Annealing query budget and reproducibility from the seeds."""
    small = _small(params)
    inst = Instance(small, instance_descriptor(small, "e1", "L", 4, 1))
    setting = [(0, {"T0_mult": 1.0, "Tend_mult": 1.0, "offset_mult": 1.0})]
    budgets = [50, 51, 100, 1000]
    da = da_runs(small, inst, 0.1, setting, range(2), budgets, "selftest")
    again = da_runs(small, inst, 0.1, setting, range(2), budgets, "selftest")
    return [
        _result("DA queries <= n", all(r["queries"] <= r["n"] for r in da)),
        _result("DA reproducible from recorded seeds", _strip_timing(da) == _strip_timing(again)),
    ]


CHECKS = [check_spectrum, check_structured_phi, check_theory_brute_force, check_mcco, check_annealing]


def run_all(params: Params, verbose: bool = True) -> list[dict]:
    results = []
    for check in CHECKS:
        for r in check(params):
            results.append(r)
            if verbose:
                print(f"  [{'ok' if r['passed'] else 'FAIL'}] {r['check']} {r['detail']}", flush=True)
    return results
