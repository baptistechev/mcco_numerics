"""Digital-annealing baseline (sections 2 and 6): parallel trial over single-bit flips with
dynamic offset, budget counted in queries to f. One fixed setting (params.py) for every instance."""

from __future__ import annotations

import time

import numpy as np

from .instances import Instance
from .params import Params
from .seeds import derive_seed


def da_setting(params: Params) -> dict:
    """The fixed annealing setting (same for every instance), in units of f."""
    return {"T0": params.DA_T0, "Tend": params.DA_TEND, "offset_increment": params.DA_OFFSET_INCREMENT}


def digital_annealing(f: np.ndarray, N: int, budget: int, setting: dict, seed: int) -> dict:
    """Parallel-trial digital annealing with dynamic offset, maximizing f; queries <= budget."""
    da_rng = np.random.default_rng(seed)
    flips = (1 << (N - 1 - np.arange(N))).astype(np.int64)
    x = int(da_rng.integers(0, 2 ** N))
    fx = float(f[x])
    queries = 1
    best_x, best_f = x, fx
    steps = (budget - 1) // N
    T0, Tend, increment = setting["T0"], setting["Tend"], setting["offset_increment"]
    offset = 0.0
    accepted_moves = 0
    for s in range(steps):
        T = T0 * (Tend / T0) ** (s / (steps - 1)) if steps > 1 else T0
        neighbours = x ^ flips
        f_neighbours = f[neighbours]
        queries += N
        j_best = int(np.argmax(f_neighbours))
        if f_neighbours[j_best] > best_f:
            best_x, best_f = int(neighbours[j_best]), float(f_neighbours[j_best])
        delta_energy = -(f_neighbours - fx)          # energy E = -f
        probability = np.exp(np.minimum(0.0, -(delta_energy - offset) / T))
        accepted = np.flatnonzero(da_rng.random(N) < probability)
        if accepted.size:
            j = int(accepted[da_rng.integers(accepted.size)])
            x, fx = int(neighbours[j]), float(f_neighbours[j])
            offset = 0.0
            accepted_moves += 1
        else:
            offset += increment
    return {"x_hat": best_x, "queries": queries, "steps": steps, "accepted_moves": accepted_moves,
            "T0": T0, "Tend": Tend}


def da_runs(params: Params, inst: Instance, run_ids: range, budgets: list[int], experiment: str) -> list[dict]:
    """One fresh annealing run per (run id, budget), seeded by (instance, run id, budget)."""
    setting = da_setting(params)
    records = []
    for run_id in run_ids:
        for n in budgets:
            seed = derive_seed(params, "da_run", inst.d["instance_seed"], run_id, n)
            start = time.perf_counter()
            out = digital_annealing(inst.f, inst.N, n, setting, seed)
            elapsed = time.perf_counter() - start
            assert out["queries"] <= n
            records.append({
                "record": "run", "method": "da", "experiment": experiment, **inst.fields(),
                "run_id": run_id, "run_seed": seed, "n": n, "setting": setting,
                "queries": out["queries"], "steps": out["steps"],
                "accepted_moves": out["accepted_moves"], "T0": out["T0"], "Tend": out["Tend"],
                **inst.evaluate_estimate(out["x_hat"]),
                "time": {"total": elapsed},
            })
    return records
