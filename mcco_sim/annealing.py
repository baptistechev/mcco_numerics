"""Digital-annealing baseline (sections 2 and 6): parallel trial over single-bit flips with
dynamic offset, budget counted in queries to f."""

from __future__ import annotations

import math
import time

import numpy as np

from .instances import Instance
from .params import Params
from .seeds import derive_seed


def da_grid(params: Params) -> list[dict]:
    """Tuning grid: multipliers on T0, T_end and the offset increment."""
    mult = params.DA_TUNING_MULTIPLIERS
    return [{"T0_mult": a, "Tend_mult": b, "offset_mult": c} for a in mult for b in mult for c in mult]


def da_setting(params: Params, multipliers: dict) -> dict:
    return {**multipliers, "T0_acceptance": params.DA_T0_ACCEPTANCE,
            "Tend_acceptance": params.DA_TEND_ACCEPTANCE, "offset_increment": params.DA_OFFSET_INCREMENT}


def digital_annealing(f: np.ndarray, N: int, budget: int, delta: float, setting: dict, seed: int) -> dict:
    """Parallel-trial digital annealing with dynamic offset, maximizing f; queries <= budget."""
    da_rng = np.random.default_rng(seed)
    flips = (1 << (N - 1 - np.arange(N))).astype(np.int64)
    x = int(da_rng.integers(0, 2 ** N))
    fx = float(f[x])
    queries = 1
    best_x, best_f = x, fx
    steps = (budget - 1) // N
    T0 = setting["T0_mult"] * delta / math.log(1 / setting["T0_acceptance"])
    Tend = setting["Tend_mult"] * delta / math.log(1 / setting["Tend_acceptance"])
    increment = setting["offset_mult"] * setting["offset_increment"] * delta
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
            "T0": T0, "Tend": Tend, "offset_increment_value": increment}


def da_runs(params: Params, inst: Instance, delta: float, settings: list[tuple[int, dict]], run_ids: range,
            budgets: list[int], experiment: str) -> list[dict]:
    """One fresh annealing run per (setting, run id, budget), seeded by (instance, run id, budget)."""
    records = []
    for setting_id, multipliers in settings:
        setting = da_setting(params, multipliers)
        for run_id in run_ids:
            for n in budgets:
                seed = derive_seed(params, "da_run", inst.d["instance_seed"], run_id, n)
                start = time.perf_counter()
                out = digital_annealing(inst.f, inst.N, n, delta, setting, seed)
                elapsed = time.perf_counter() - start
                assert out["queries"] <= n
                records.append({
                    "record": "run", "method": "da", "experiment": experiment, **inst.fields(),
                    "run_id": run_id, "run_seed": seed, "n": n, "setting_id": setting_id, "setting": setting,
                    "delta": delta, "queries": out["queries"], "steps": out["steps"],
                    "accepted_moves": out["accepted_moves"], "T0": out["T0"], "Tend": out["Tend"],
                    **inst.evaluate_estimate(out["x_hat"]),
                    "time": {"total": elapsed},
                })
    return records


def da_delta_values(params: Params, inst: Instance) -> tuple[np.ndarray, int]:
    """Nonzero |Delta f| over the N single flips of random x (tuning instances only)."""
    seed = derive_seed(params, "da_delta", inst.d["instance_seed"])
    points = np.random.default_rng(seed).integers(0, 2 ** inst.N, size=params.DA_DELTA_POINTS)
    flips = (1 << (inst.N - 1 - np.arange(inst.N))).astype(np.int64)
    diffs = np.abs(inst.f[points[:, None] ^ flips[None, :]] - inst.f[points][:, None]).ravel()
    return diffs[diffs > 0], seed
