"""Work units (run in worker processes) and their execution with commit-on-completion."""

from __future__ import annotations

import multiprocessing
import time
from datetime import datetime, timezone

import numpy as np
from troma import DitString

from .annealing import da_grid, da_runs
from .bp import bp_nonneg, noise_level, spectral_norm, top_entries
from .instances import Instance
from .mcco import mcco_sample, sampled_budgets, threshold_sample
from .params import Params
from .posthoc import best_sampled, combined_estimate
from .records import Output
from .seeds import derive_seed
from .sketches import SketchSet, sketch_names
from .theory import e2_thresholds, theory_quantities, thresholded


def unit_instance(params: Params, payload: dict) -> dict:
    inst = Instance(params, payload["descriptor"], with_wh=True)
    return {"instances": [inst.record()]}


def unit_theory(params: Params, payload: dict) -> dict:
    """Theory at t = 0 and t = exact Q-th percentile for the sketches of the instance's context or,
    with ``labels`` ({sketch: [E2-grid labels]}), only those thresholds of the E2 grid."""
    inst = Instance(params, payload["descriptor"])
    if "labels" in payload:
        grid = dict(e2_thresholds(params, inst))
        wanted = {name: [(label, grid[label]) for label in labels] for name, labels in payload["labels"].items()}
    else:
        base = [("zero", 0.0), ("q_exact", inst.q_exact)]
        wanted = {name: base for name in sketch_names(params, payload["descriptor"], payload["context"])}
    sketches = SketchSet(params, inst, names=[n for n in params.SKETCHES if wanted.get(n)])
    records = []
    for name in sketches.names:
        for label, t in wanted[name]:
            start = time.perf_counter()
            quantities = theory_quantities(sketches.phis[name], thresholded(inst.f, t), inst.x_star,
                                           inst.maximizers, inst.N, params.E5_DELTA)
            records.append({"record": "theory", **inst.fields(), "unique_maximizer": inst.unique,
                            "sketch": name, "sketch_seed": sketches.seeds[name], "t_label": label, "t": t,
                            "t_le_f_x2": bool(t <= inst.f_x2), **quantities,
                            "time": time.perf_counter() - start})
    return {"theory": records}


def unit_tuning(params: Params, payload: dict) -> dict:
    inst = Instance(params, payload["descriptor"])
    settings = list(enumerate(da_grid(params)))
    records = da_runs(params, inst, payload["delta"], settings, range(params.J), params.budgets, "tuning")
    return {"tuning_runs": records}


def unit_e1(params: Params, payload: dict) -> dict:
    """E1: MCCO (adaptive threshold, Problem II also at the exact threshold) and annealing."""
    inst = Instance(params, payload["descriptor"])
    sketches = SketchSet(params, inst, names=sketch_names(params, payload["descriptor"], "e1"))
    records = []
    fixed = [{"mode": "fixed", "label": "q_exact", "t": inst.q_exact}]
    for sample_id in range(params.J):
        for rec in mcco_sample(params, inst, sketches, sample_id, params.budgets,
                               [{"mode": "adaptive"}], problem2_only=fixed):
            rec["experiment"] = "e1"
            records.append(rec)
    choice = payload["da_setting"]
    records += da_runs(params, inst, payload["delta"], [(choice["setting_id"], choice["multipliers"])],
                       range(params.J), params.budgets, "e1")
    return {"runs": records}


def unit_sweep(params: Params, payload: dict) -> dict:
    """E2-budget and E3 sweep ("decode"): every budget, thresholds adaptive, zero and exact.
    E5a sweep ("problem2"): budgets up to E5_N_MAX, Problem II only at the exact threshold."""
    inst = Instance(params, payload["descriptor"])
    sketches = SketchSet(params, inst, names=sketch_names(params, payload["descriptor"], "single"))
    exact = {"mode": "fixed", "label": "q_exact", "t": inst.q_exact}
    if payload["kind"] == "problem2":
        budgets, modes, problem2_only = params.e5_budgets, [], [exact]
    else:
        budgets, problem2_only = params.budgets, None
        modes = [{"mode": "adaptive"}, {"mode": "fixed", "label": "zero", "t": 0.0}, exact]
    records = []
    for sample_id in payload["sample_ids"]:
        for rec in mcco_sample(params, inst, sketches, sample_id, budgets, modes, problem2_only=problem2_only):
            rec["experiment"] = "sweep"
            rec["role"] = payload["role"]
            records.append(rec)
    return {"runs": records}


def unit_e2(params: Params, payload: dict) -> dict:
    """E2: threshold grid at the chosen budget."""
    inst = Instance(params, payload["descriptor"])
    sketches = SketchSet(params, inst, names=sketch_names(params, payload["descriptor"], "single"))
    modes = [{"mode": "fixed", "label": label, "t": t} for label, t in e2_thresholds(params, inst)]
    records = []
    for sample_id in payload["sample_ids"]:
        for rec in mcco_sample(params, inst, sketches, sample_id, [payload["budget"]], modes):
            rec["experiment"] = "e2"
            records.append(rec)
    return {"runs": records}


def unit_s1(params: Params, payload: dict) -> dict:
    """S1: basis-pursuit decoding of E1 samples (same sample, adaptive threshold and troma sketch as the
    E1 runs, so the pair with the E1 matching-pursuit outcome is exact). One unit = one instance, the
    given sample ids and sketches (small units balance the load over many workers)."""
    inst = Instance(params, payload["descriptor"])
    sketches = SketchSet(params, inst, names=payload["sketches"])
    size = 2 ** inst.N
    norms = {name: spectral_norm(sketches.phis[name], size) for name in sketches.names}
    records = []
    for sample_id in payload["sample_ids"]:
        sample_seed = derive_seed(params, "sample", inst.d["instance_seed"], sample_id)
        for step in sampled_budgets(params, inst, sample_id, params.budgets):
            n, problem, oracle, prefix = step["n"], step["problem"], step["oracle"], step["prefix"]
            thresholded_sample = threshold_sample(step["full_sample"], step["t_adaptive"])
            in_sample = set(prefix.tolist())
            x_sample, f_sample = best_sampled(inst.f, prefix)
            for name in sketches.names:
                phi = sketches.phis[name]
                problem.sample = thresholded_sample
                start = time.perf_counter()
                y = np.asarray(problem.sketching(sketches.maps[name]).sketch_values, dtype=float)
                time_sketching = time.perf_counter() - start
                start = time.perf_counter()
                eta = noise_level(phi, np.asarray(thresholded_sample.indexes, dtype=np.int64),
                                  np.asarray(thresholded_sample.values, dtype=float), n, size)
                if np.any(y):
                    z, diagnostics = bp_nonneg(phi, y, eta, size, params.S1_BP_ITERATIONS, norms[name])
                    candidates = top_entries(z, params.MP_ITERATIONS)
                else:
                    diagnostics, candidates = {}, []
                time_decoding = time.perf_counter() - start
                start = time.perf_counter()
                values = [oracle(np.asarray(DitString.from_integer(x, inst.N))) for x in candidates]
                time_candidates = time.perf_counter() - start
                x_bp = candidates[int(np.argmax(values))] if candidates else None
                f_bp = max(values) if candidates else None
                x_hat = combined_estimate(x_bp, f_bp, x_sample, f_sample)
                records.append({
                    "record": "run", "method": "mcco", "experiment": "s1", "decoder": "bp", **inst.fields(),
                    "sample_id": sample_id, "sample_seed": sample_seed, "sketch": name,
                    "sketch_seed": sketches.seeds[name], "threshold_mode": "adaptive",
                    "t": step["t_adaptive"], "n": n, "n_kept": len(thresholded_sample.values), "eta": eta,
                    **diagnostics, "candidates": [[x, v] for x, v in zip(candidates, values)],
                    "queries": n + sum(1 for x in candidates if x not in in_sample),
                    **inst.evaluate_estimate(x_hat),
                    **{f"{k}_mp": v for k, v in inst.evaluate_estimate(x_bp).items()},   # decoder-only outcome
                    "sample_best_x": x_sample, "sample_best_f": f_sample, "sample_n_max": max(params.budgets),
                    "time": {"sampling": step["time_sampling"], "sketching": time_sketching,
                             "decoding": time_decoding, "candidates": time_candidates},
                })
    return {"runs": records}


UNIT_FUNCTIONS = {
    "instances": unit_instance,
    "theory": unit_theory,
    "tuning": unit_tuning,
    "e1": unit_e1,
    "sweep": unit_sweep,
    "e2": unit_e2,
    "s1": unit_s1,
}


def _run_unit(args: tuple) -> tuple[str, dict, float]:
    params, stage, unit = args
    start = time.perf_counter()
    outputs = UNIT_FUNCTIONS[stage](params, unit["payload"])
    for records in outputs.values():
        for rec in records:
            rec["unit"] = unit["key"]
    return unit["key"], outputs, time.perf_counter() - start


def execute(out: Output, stage: str, units: list[dict], workers: int) -> None:
    """Run the units not yet committed; each unit's records are written, then the unit is committed."""
    done = out.done_units()
    todo = [u for u in units if u["key"] not in done]
    print(f"[{stage}] {len(units)} units, {len(units) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tasks = [(out.params, stage, u) for u in todo]
    start = time.perf_counter()

    def commit(key, outputs, elapsed, i):
        for name, records in outputs.items():
            out.append(f"{name}.jsonl", records)
        out.append("progress.jsonl", [{"unit": key, "stage": stage, "elapsed": elapsed,
                                       "n_records": sum(len(r) for r in outputs.values()),
                                       "finished": datetime.now(timezone.utc).isoformat()}])
        print(f"[{stage}] {i}/{len(todo)} {key} ({elapsed:.1f}s, total {time.perf_counter() - start:.0f}s)",
              flush=True)

    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            commit(*_run_unit(task), i)
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_unit, tasks), 1):
                commit(*result, i)
