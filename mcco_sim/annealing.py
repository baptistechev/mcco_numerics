"""Simulated-annealing baseline (sections 2 and 6): one single-bit-flip proposal per query to f,
Metropolis acceptance, geometric temperature schedule. One fixed setting (params.py) for every
instance. The budget matches MCCO's: at budget n, MCCO queries the n sampled strings and at most
MP_ITERATIONS decoded candidates, so annealing gets exactly n + MP_ITERATIONS queries."""

from __future__ import annotations

import math
import time

import numpy as np

from .instances import Instance
from .params import Params
from .seeds import derive_seed


def sa_setting(params: Params) -> dict:
    """The fixed annealing setting (same for every instance), in units of f."""
    return {"T0": params.SA_T0, "Tend": params.SA_TEND}


def annealing_budget(params: Params, n: int) -> int:
    """Queries given to annealing at budget n: the n samples plus the MCCO candidate evaluations."""
    return n + params.MP_ITERATIONS


def simulated_annealing(f: np.ndarray, N: int, budget: int, setting: dict, seed: int) -> dict:
    """Single-bit-flip simulated annealing maximizing f, using exactly ``budget`` queries:
    one for the random initial string, then one per proposed flip. Returns the best string queried."""
    rng = np.random.default_rng(seed)
    flips = (1 << (N - 1 - np.arange(N))).astype(np.int64)
    x = int(rng.integers(0, 2 ** N))
    fx = float(f[x])
    queries = 1
    best_x, best_f = x, fx
    steps = budget - 1
    T0, Tend = setting["T0"], setting["Tend"]
    temperatures = T0 * (Tend / T0) ** (np.arange(steps) / (steps - 1)) if steps > 1 else np.full(steps, T0)
    bits = rng.integers(0, N, size=steps)
    uniforms = rng.random(steps)
    accepted_moves = 0
    for s in range(steps):
        y = x ^ int(flips[bits[s]])
        fy = float(f[y])
        queries += 1
        if fy > best_f:
            best_x, best_f = y, fy
        delta = fy - fx                               # maximizing f: energy E = -f
        if delta >= 0 or uniforms[s] < math.exp(delta / temperatures[s]):
            x, fx = y, fy
            accepted_moves += 1
    return {"x_hat": best_x, "queries": queries, "steps": steps, "accepted_moves": accepted_moves,
            "T0": T0, "Tend": Tend}


def annealing_runs(params: Params, inst: Instance, run_ids: range, budgets: list[int], experiment: str) -> list[dict]:
    """One fresh annealing run per (run id, budget), seeded by (instance, run id, budget)."""
    setting = sa_setting(params)
    records = []
    for run_id in run_ids:
        for n in budgets:
            budget = annealing_budget(params, n)
            seed = derive_seed(params, "sa_run", inst.d["instance_seed"], run_id, n)
            start = time.perf_counter()
            out = simulated_annealing(inst.f, inst.N, budget, setting, seed)
            elapsed = time.perf_counter() - start
            assert out["queries"] == budget
            records.append({
                "record": "run", "method": "sa", "experiment": experiment, **inst.fields(),
                "run_id": run_id, "run_seed": seed, "n": n, "budget": budget, "setting": setting,
                "queries": out["queries"], "steps": out["steps"],
                "accepted_moves": out["accepted_moves"], "T0": out["T0"], "Tend": out["Tend"],
                **inst.evaluate_estimate(out["x_hat"]),
                "time": {"total": elapsed},
            })
    return records
