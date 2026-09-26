"""Stages of the simulation, in order: instances, theory, tuning, e1, sweep, e2."""

from __future__ import annotations

import numpy as np

from .annealing import da_delta_values
from .instances import Instance, build_instance_descriptors, e3_descriptor, select_e2_instances
from .records import Output
from .sketches import sketch_names
from .runner import execute


def stage_instances(out: Output, workers: int) -> None:
    """Every E1 and tuning instance, then the E3 instance and the E2 selection."""
    params = out.params
    descriptors = build_instance_descriptors(params)
    units = [{"key": f"instances/{d['key']}", "payload": {"descriptor": d}}
             for group in descriptors.values() for d in group]
    execute(out, "instances", units, workers)

    # E3: first draw with a unique maximizer (every draw tried is recorded).
    if not out.path("selection.json").exists():
        for draw in range(params.E3_MAX_DRAWS):
            d = e3_descriptor(params, draw)
            key = f"instances/{d['key']}"
            if key not in out.done_units():
                execute(out, "instances", [{"key": key, "payload": {"descriptor": d}}], 1)
            rec = next(r for r in out.load("instances.jsonl") if r["unit"] == key)
            if rec["unique_maximizer"]:
                e3 = d
                break
        else:
            raise RuntimeError("No E3 instance with a unique maximizer found.")
        selection = {"e2": select_e2_instances(params, out.load("instances.jsonl")), "e3": {"descriptor": e3}}
        out.write_json("selection.json", selection)


def stage_theory(out: Output, workers: int) -> None:
    """Theory checks for every E1 instance and the E3 instance (E2 grid on the E2 instances)."""
    selection = out.read_json("selection.json")
    e2_keys = {s["descriptor"]["key"] for s in selection["e2"].values()}
    units = [{"key": f"theory/{d['key']}", "payload": {"descriptor": d, "context": context,
                                                        "e2_grid": d["key"] in e2_keys}}
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
    params = out.params
    selection = out.read_json("selection.json")
    targets = [("e2_" + fam, s["descriptor"]) for fam, s in selection["e2"].items()]
    targets.append(("e3", selection["e3"]["descriptor"]))
    units = [{"key": f"sweep/{d['key']}/samples{block[0]}-{block[-1]}",
              "payload": {"descriptor": d, "sample_ids": block, "role": role}}
             for role, d in targets for block in _blocks(params.J_SINGLE, params.BLOCK_SIZE)]
    execute(out, "sweep", units, workers)


def choose_e2_budgets(out: Output) -> dict:
    """Budget where the t = 0 success of every single-instance sketch is strictly in (0, 1), preferring the mean
    success closest to 0.5 (fallback: closest to 0.5 without the (0, 1) condition, flagged)."""
    params = out.params
    selection = out.read_json("selection.json")
    records = [r for r in out.load("runs.jsonl")
               if r.get("experiment") == "sweep" and r["threshold_label"] == "zero" and r["method"] == "mcco"]
    choices = {}
    for family, s in selection["e2"].items():
        key = s["descriptor"]["key"]
        names = sketch_names(params, s["descriptor"], "single")
        rates = {}
        for n in params.budgets:
            rates[n] = {name: float(np.mean([r["success"] for r in records
                                             if r["instance_key"] == key and r["n"] == n and r["sketch"] == name]))
                        for name in names}
        valid = [n for n in params.budgets if all(0 < v < 1 for v in rates[n].values())]
        candidates = valid or params.budgets
        n_best = min(candidates, key=lambda n: (abs(np.mean(list(rates[n].values())) - 0.5), n))
        choices[family] = {"instance_key": key, "budget": n_best, "rule_satisfied": bool(valid),
                           "success_t0": {str(n): r for n, r in rates.items()}}
    return choices


def stage_e2(out: Output, workers: int) -> None:
    params = out.params
    if not out.path("e2_budget_choice.json").exists():
        out.write_json("e2_budget_choice.json", choose_e2_budgets(out))
    choices = out.read_json("e2_budget_choice.json")
    selection = out.read_json("selection.json")
    units = [{"key": f"e2/{s['descriptor']['key']}/samples{block[0]}-{block[-1]}",
              "payload": {"descriptor": s["descriptor"], "sample_ids": block, "budget": choices[fam]["budget"]}}
             for fam, s in selection["e2"].items()
             for block in _blocks(params.J_SINGLE, params.BLOCK_SIZE)]
    execute(out, "e2", units, workers)


STAGES = {
    "instances": stage_instances,
    "theory": stage_theory,
    "tuning": stage_tuning,
    "e1": stage_e1,
    "sweep": stage_sweep,
    "e2": stage_e2,
}
