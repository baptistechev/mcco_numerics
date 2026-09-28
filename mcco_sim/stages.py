"""Stages of the simulation, in order: instances, theory, tuning, e1, sweep, e2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .annealing import da_delta_values
from .instances import (Instance, build_instance_descriptors, e3_descriptor, select_e2_typical,
                        select_e5_instances)
from .records import Output, RecordStore
from .sketches import sketch_names
from .theory import e2_thresholds
from .runner import execute


def stage_instances(out: Output, workers: int) -> None:
    """Every E1 and tuning instance, then the E3 instance (the E2 instances are chosen by e2select)."""
    params = out.params
    descriptors = build_instance_descriptors(params)
    units = [{"key": f"instances/{d['key']}", "payload": {"descriptor": d}}
             for group in descriptors.values() for d in group]
    execute(out, "instances", units, workers)

    # E3: draw 0, whatever its number of maximizers.
    if not out.path("selection.json").exists():
        e3 = e3_descriptor(params, 0)
        execute(out, "instances", [{"key": f"instances/{e3['key']}", "payload": {"descriptor": e3}}], 1)
        out.write_json("selection.json", {"e3": {"descriptor": e3}})


def stage_theory(out: Output, workers: int) -> None:
    """Theory checks for every E1 instance and the E3 instance (the E2 grid is done by stage_e2)."""
    selection = out.read_json("selection.json")
    units = [{"key": f"theory/{d['key']}", "payload": {"descriptor": d, "context": context}}
             for context, group in (("e1", build_instance_descriptors(out.params)["e1"]),
                                    ("single", [selection["e3"]["descriptor"]]))
             for d in group]
    execute(out, "theory", units, workers)


def stage_tuning(out: Output, workers: int) -> None:
    """Annealing energy scale per ensemble, tuning grid runs, and the global choice."""
    params = out.params
    tuning = build_instance_descriptors(params)["tuning"]
    if not out.path("da_delta.json").exists():
        deltas = {}
        for family in params.FAMILIES:
            for R in params.R_VALUES:
                group = [d for d in tuning if d["family"] == family and d["n_rules"] == R]
                values, seeds = [], {}
                for d in group:
                    v, seed = da_delta_values(params, Instance(params, d))
                    values.append(v)
                    seeds[d["key"]] = seed
                pooled = np.concatenate(values)
                deltas[f"{family}/R{R}"] = {"delta": float(np.median(pooled)), "n_values": int(pooled.size),
                                            "tuning_instances": [d["key"] for d in group], "seeds": seeds}
        out.write_json("da_delta.json", deltas)
    deltas = out.read_json("da_delta.json")
    units = [{"key": f"tuning/{d['key']}",
              "payload": {"descriptor": d, "delta": deltas[f"{d['family']}/R{d['n_rules']}"]["delta"]}}
             for d in tuning]
    execute(out, "tuning", units, workers)

    table = {}
    for rec in out.load("tuning_runs.jsonl"):
        entry = table.setdefault(rec["setting_id"], {"setting": rec["setting"], "success": [], "distance": []})
        entry["success"].append(rec["success"])
        entry["distance"].append(rec["functional_distance"])
    summary = [{"setting_id": sid,
                "multipliers": {k: e["setting"][k] for k in ("T0_mult", "Tend_mult", "offset_mult")},
                "mean_success": float(np.mean(e["success"])),
                "mean_functional_distance": float(np.mean(e["distance"])),
                "n_runs": len(e["success"])} for sid, e in sorted(table.items())]
    # Criterion: mean success over all budgets and tuning instances; ties -> smaller mean
    # functional distance, then grid order.
    best = min(summary, key=lambda s: (-s["mean_success"], s["mean_functional_distance"], s["setting_id"]))
    out.write_json("tuning_choice.json", {"criterion": "mean success over budget grid and tuning instances",
                                          "chosen": best, "table": summary})


def stage_e1(out: Output, workers: int) -> None:
    deltas = out.read_json("da_delta.json")
    chosen = out.read_json("tuning_choice.json")["chosen"]
    units = [{"key": f"e1/{d['key']}",
              "payload": {"descriptor": d, "delta": deltas[f"{d['family']}/R{d['n_rules']}"]["delta"],
                          "da_setting": chosen}}
             for d in build_instance_descriptors(out.params)["e1"]]
    execute(out, "e1", units, workers)


def _blocks(n: int, size: int) -> list[list[int]]:
    return [list(range(s, min(s + size, n))) for s in range(0, n, size)]


def stage_sweep(out: Output, workers: int) -> None:
    """J_SINGLE runs per instance on the E3 instance (all budgets, decoded) and on the E5a instances
    (budgets up to E5_N_MAX, Problem II only). The E5a instances are selected from the theory
    records of this directory (the theory stage must have run)."""
    params = out.params
    selection = out.read_json("selection.json")
    if "e5" not in selection:
        theory = out.load("theory.jsonl")
        if not theory:
            raise SystemExit("The sweep selects the E5a instances from theory.jsonl: run the theory stage first.")
        e1_units = {f"theory/{d['key']}" for d in build_instance_descriptors(params)["e1"]}
        if not e1_units <= out.done_units():
            raise SystemExit("The theory stage is not complete: finish it before the sweep.")
        # Instances: the single-instance sketches, i.e. those run in the sweep.
        selection["e5"] = select_e5_instances(params, theory, sketch_names(params, {}, "single"))
        out.write_json("selection.json", selection)
    targets = [("e3", selection["e3"]["descriptor"], "decode")]
    targets += [("e5_" + fam, s["descriptor"], "problem2") for fam, s in selection["e5"].items() if s]
    units = [{"key": f"sweep/{role}/{d['key']}/samples{block[0]}-{block[-1]}",
              "payload": {"descriptor": d, "sample_ids": block, "role": role, "kind": kind}}
             for role, d, kind in targets for block in _blocks(params.J_SINGLE, params.BLOCK_SIZE)]
    execute(out, "sweep", units, workers)


def stage_e2select(out: Output, workers: int, e1_records: str | Path | None = None) -> None:
    """E2 instances and budget from the E1 success curves (select_e2_typical). The E1 runs are read
    from this directory or, read-only, from ``e1_records`` (e.g. an earlier results directory)."""
    params = out.params
    selection = out.read_json("selection.json")
    if "e2" in selection and out.path("e2_budget_choice.json").exists():
        print("[e2select] already done", flush=True)
        return
    source = RecordStore(e1_records) if e1_records else out
    instances = out.load("instances.jsonl")
    if not instances:
        raise SystemExit("e2select needs the instance records of this directory: run the instances stage first.")
    if e1_records:
        # The pool must be the same instances in both directories.
        mine = {r["instance_key"]: r["instance_seed"] for r in instances
                if r["ensemble"] == "e1" and r["n_rules"] == params.E2_R}
        theirs = {r["instance_key"]: r["instance_seed"] for r in source.load("instances.jsonl")
                  if r["ensemble"] == "e1" and r["n_rules"] == params.E2_R}
        if mine != theirs:
            raise SystemExit(f"The E1 instances of {e1_records} differ from those of {out.dir} (seeds).")
    fields = ["experiment", "method", "record", "threshold_mode", "sketch", "instance_key", "n", "success", "success_mp"]
    runs = source.load("runs.jsonl", fields=fields, contains=['"experiment": "e1"', f'"n_rules": {params.E2_R},'])
    if not runs:
        raise SystemExit(f"No E1 runs found in {source.dir}.")
    chosen = select_e2_typical(params, instances, runs, sketch_names(params, {}, "single"))
    selection["e2"] = {fam: {k: v for k, v in c.items() if k in ("descriptor", "d_i", "pool_size", "pool_median_d_i")}
                       for fam, c in chosen.items()}
    out.write_json("selection.json", selection)
    out.write_json("e2_budget_choice.json", {
        fam: {"instance_key": c["descriptor"]["key"], "budget": c["n_star"], "d_i": c["d_i"],
              "pool_size": c["pool_size"], "pool_median_d_i": c["pool_median_d_i"],
              "median_success": c["median_success"], "instance_success": c["instance_success"],
              "rule": c["rule"], "e1_records": str(Path(source.dir).resolve())}
        for fam, c in chosen.items()})
    for fam, c in chosen.items():
        print(f"[e2select] {fam}: {c['descriptor']['key']}, n* = {c['n_star']}, d_i = {c['d_i']:.2f} "
              f"(pool {c['pool_size']}, median d_i {c['pool_median_d_i']:.2f})", flush=True)


def stage_e2(out: Output, workers: int) -> None:
    """Theory of the E2 threshold grid on the two E2 instances, then the E2 runs at n*."""
    params = out.params
    if not out.path("e2_budget_choice.json").exists():
        raise SystemExit("Run the e2select stage first (it writes e2_budget_choice.json).")
    choices = out.read_json("e2_budget_choice.json")
    selection = out.read_json("selection.json")
    theory = out.load("theory.jsonl")
    units = []
    for fam, s in selection["e2"].items():
        d = s["descriptor"]
        grid = [label for label, _ in e2_thresholds(params, Instance(params, d))]
        labels = {}
        for name in sketch_names(params, d, "single"):
            done = {r["t_label"] for r in theory if r["instance_key"] == d["key"] and r["sketch"] == name}
            labels[name] = [label for label in grid if label not in done]
        units.append({"key": f"theory_e2/{d['key']}", "payload": {"descriptor": d, "labels": labels}})
    execute(out, "theory", units, workers)
    units = [{"key": f"e2/{s['descriptor']['key']}/samples{block[0]}-{block[-1]}",
              "payload": {"descriptor": s["descriptor"], "sample_ids": block, "budget": choices[fam]["budget"]}}
             for fam, s in selection["e2"].items()
             for block in _blocks(params.J_SINGLE, params.BLOCK_SIZE)]
    execute(out, "e2", units, workers)


def stage_s1(out: Output, workers: int) -> None:
    """S1 (supplementary): basis-pursuit decoding of E1 samples: |R| in S1_R_VALUES, instance ids
    < S1_INSTANCES, sample ids < S1_J, sketches in S1_SKETCHES, budgets params.s1_budgets.
    The matching-pursuit outcome of the same samples is in the E1 records (results directory)."""
    # One unit per (instance, sample, sketch), so many workers stay busy until the end.
    params = out.params
    units = [{"key": f"s1/{d['key']}/sample{j}/{name}",
              "payload": {"descriptor": d, "sample_ids": [j], "sketches": [name]}}
             for d in build_instance_descriptors(params)["e1"]
             if d["n_rules"] in params.S1_R_VALUES and d["instance_id"] < params.S1_INSTANCES
             for name in sketch_names(params, d, "e1") if name in params.S1_SKETCHES
             for j in range(min(params.S1_J, params.J))]
    execute(out, "s1", units, workers)


STAGES = {
    "instances": stage_instances,
    "theory": stage_theory,
    "tuning": stage_tuning,
    "e1": stage_e1,
    "sweep": stage_sweep,
    "e2select": stage_e2select,
    "e2": stage_e2,
    "s1": stage_s1,
}

# Supplementary stages: not part of --stage all (run them explicitly).
SUPPLEMENTARY_STAGES = ("s1",)
