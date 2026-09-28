"""Pilot: correctness checks, a small run through every code path, and a compute estimate."""

from __future__ import annotations

import gc
import json
import math
import time

import numpy as np

from .annealing import da_delta_values, da_grid, da_runs
from .checks import run_all
from .instances import Instance, build_instance_descriptors
from .mcco import mcco_sample
from .params import Params
from .records import Output
from .runner import unit_s1, unit_theory
from .sketches import SketchSet, sketch_rows


def pilot(out: Output) -> None:
    """PILOT_INSTANCES instances (one per ensemble, in order) x PILOT_SAMPLES samples."""
    params = out.params
    results = run_all(params)
    out.write_json("selftest.json", results)
    if not all(r["passed"] for r in results):
        raise SystemExit("Self-checks failed; see selftest.json")

    descriptors = build_instance_descriptors(params)["e1"]
    chosen = [descriptors[i * params.I] for i in range(min(params.PILOT_INSTANCES, len(descriptors) // params.I))]
    timings = {"instance": [], "theory": {}, "sketch_setup": {}, "mcco": [], "da": []}
    for d in chosen:
        start = time.perf_counter()
        inst = Instance(params, d, with_wh=True)
        timings["instance"].append(time.perf_counter() - start)
        out.append("instances.jsonl", [inst.record() | {"unit": "pilot"}])
        th = unit_theory(params, {"descriptor": d, "context": "e1"})["theory"]  # own sketches, freed after
        for rec in th:
            timings["theory"].setdefault(rec["sketch"], []).append(rec["time"])
            rec["unit"] = "pilot"
        out.append("theory.jsonl", th)
        sketches = SketchSet(params, inst)       # every sketch, so each one is timed
        for name, seconds in sketches.setup_time.items():
            timings["sketch_setup"].setdefault(name, []).append(seconds)
        values, _ = da_delta_values(params, inst)
        delta = float(np.median(values))
        for sample_id in range(params.PILOT_SAMPLES):
            recs = mcco_sample(params, inst, sketches, sample_id, params.budgets, [{"mode": "adaptive"}],
                               problem2_only=[{"mode": "fixed", "label": "q_exact", "t": inst.q_exact}])
            for r in recs:
                r["unit"] = "pilot"
            out.append("runs.jsonl", recs)
            timings["mcco"] += recs
        default = {"T0_mult": 1.0, "Tend_mult": 1.0, "offset_mult": 1.0}
        da = da_runs(params, inst, delta, [(da_grid(params).index(default), default)],
                     range(params.PILOT_SAMPLES), params.budgets, "pilot")
        for r in da:
            r["unit"] = "pilot"
        out.append("runs.jsonl", da)
        timings["da"] += da
        print(f"[pilot] {d['key']} done", flush=True)
    # S1: one basis-pursuit decode per sketch on the first pilot instance (decode cost barely depends
    # on n: every iteration acts on the 2^N entries of z), for the sketches of S1_SKETCHES. Free the
    # loop's sketches first: with the random sketch, each SketchSet holds a 4.3 GB matrix.
    del sketches
    gc.collect()
    s1 = unit_s1(params.replace(J=1, N_MIN=params.N_MAX),
                 {"descriptor": chosen[0], "sample_ids": [0], "sketches": list(params.S1_SKETCHES)})["runs"]
    timings["s1_decode"] = {r["sketch"]: r["time"]["decoding"] for r in s1}
    print(f"[pilot] S1 decode seconds: {timings['s1_decode']}", flush=True)
    out.write_json("pilot_estimate.json", compute_estimate(params, timings))


def compute_estimate(params: Params, timings: dict) -> dict:
    """Extrapolate the CPU time of every stage from the pilot timings (single-thread seconds).

    Costs are counted per sketch: a sketch contributes to E1 only on its ``e1_instances`` subset,
    and to the sweep / E2 only if it is a single-instance sketch."""
    budgets = params.budgets
    sketches = list(params.SKETCHES)
    runs = [r for r in timings["mcco"] if r["record"] == "run"]
    p2 = [r for r in timings["mcco"] if r["record"] == "problem2"]

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    sampling = {n: mean([r["time"]["sampling"] for r in runs if r["n"] == n and r["sketch"] == sketches[0]])
                for n in budgets}
    decode = {(s, n): mean([r["time"]["sketching"] + r["time"]["decoding"] + r["time"]["candidates"]
                            + r["time"]["problem2"] for r in runs if r["n"] == n and r["sketch"] == s])
              for s in sketches for n in budgets}
    p2_only = {(s, n): mean([r["time"]["sketching"] + r["time"]["problem2"] for r in p2
                             if r["n"] == n and r["sketch"] == s]) for s in sketches for n in budgets}
    da = {n: mean([r["time"]["total"] for r in timings["da"] if r["n"] == n]) for n in budgets}
    instance_time = mean(timings["instance"])
    setup = {s: mean(timings["sketch_setup"].get(s, [])) for s in sketches}
    # Theory per instance and sketch = sum over its records (both thresholds).
    theory = {s: sum(v) / max(1, len(timings["instance"])) for s, v in timings["theory"].items()}

    n_ensembles = len(params.FAMILIES) * len(params.R_VALUES)
    n_e1 = n_ensembles * params.I
    n_tuning = n_ensembles * params.I_TUNING
    e1_instances = {s: n_ensembles * (params.I if spec["e1_instances"] is None else min(spec["e1_instances"], params.I))
                    for s, spec in params.SKETCHES.items()}
    single = [s for s, spec in params.SKETCHES.items() if spec["single_instance"]]
    J, J1 = params.J, params.J_SINGLE
    per_sample = {s: sum(decode[(s, n)] + p2_only[(s, n)] for n in budgets) for s in sketches}
    da_sample_cost = sum(da.values())
    n_sweep_units = math.ceil(J1 / params.BLOCK_SIZE)
    n_e2_units = 2 * math.ceil(J1 / params.BLOCK_SIZE)
    mid = budgets[len(budgets) // 2]
    single_setup = instance_time + sum(setup[s] for s in single)
    n_e5_units = 2 * math.ceil(J1 / params.BLOCK_SIZE)
    p2_single = {s: {n: p2_only[(s, n)] for n in budgets} for s in single}

    def scaled(per_budget: dict, n: int) -> float:
        """Timing at budget n; beyond the pilot grid (E5a), linear in n from the largest budget."""
        return per_budget[n] if n in per_budget else per_budget[budgets[-1]] * n / budgets[-1]

    estimate = {
        "instances": (n_e1 + n_tuning) * instance_time,
        "theory": sum(e1_instances[s] * theory.get(s, 0.0) for s in sketches),
        "tuning": n_tuning * J * len(da_grid(params)) * da_sample_cost,
        "e1": (n_e1 * (instance_time + J * (sum(sampling.values()) + da_sample_cost))
               + sum(e1_instances[s] * (setup[s] + J * per_sample[s]) for s in sketches)),
        "sweep": (J1 * sum(sampling[n] + 3 * sum(decode[(s, n)] for s in single) for n in budgets)   # E3
                  + 2 * J1 * sum(scaled(sampling, n) + sum(scaled(p2_single[s], n) for s in single)
                                 for n in params.e5_budgets)
                  + (n_sweep_units + n_e5_units) * single_setup),
        "e2 (budget at mid grid)": (2 * J1 * (sampling[mid] + (len(params.E2_PERCENTILES) + 3)
                                              * sum(decode[(s, mid)] for s in single))
                                    + n_e2_units * single_setup
                                    # theory of the threshold grid on the two E2 instances
                                    + 2 * (len(params.E2_PERCENTILES) + 3) / 2 * sum(theory.get(s, 0.0) for s in single)),
    }
    if "s1_decode" in timings:
        s1_instances = min(params.S1_INSTANCES, params.I)
        n_s1 = {s: len(params.FAMILIES) * len(params.S1_R_VALUES) * (s1_instances if spec["e1_instances"] is None
                                                                        else min(spec["e1_instances"], s1_instances))
                for s, spec in params.SKETCHES.items() if s in params.S1_SKETCHES}
        s1_J = min(params.S1_J, J)
        estimate["s1 (supplementary)"] = sum(
            n_s1[s] * s1_J * (len(params.s1_budgets) * timings["s1_decode"].get(s, 0.0)) for s in n_s1) \
            + sum(n_s1.values()) * s1_J * sum(sampling[n] for n in params.s1_budgets)
    total = sum(estimate.values())
    random_rows = [sketch_rows(params, s) for s in params.SKETCHES.values() if s["type"] == "gaussian"]
    report = {
        "cpu_seconds": estimate,
        "cpu_hours_total": total / 3600,
        "e1_instances_per_sketch": e1_instances,
        "single_instance_sketches": single,
        "per_budget_mean_seconds": {
            "sampling": sampling,
            "mcco_sketch_decode": {f"{s}/{n}": v for (s, n), v in decode.items()},
            "da": da,
        },
        "sketch_setup_seconds": setup,
        "theory_seconds_per_instance": theory,
        "note": "single-thread CPU seconds; divide by workers for wall-clock. A worker running a "
                f"Gaussian sketch holds rows {random_rows} x 2^{params.N} columns (float64).",
    }
    print(json.dumps(report["cpu_seconds"], indent=2), f"\ntotal CPU hours: {report['cpu_hours_total']:.1f}",
          flush=True)
    return report
