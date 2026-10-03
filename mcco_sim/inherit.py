"""Complete results directories: every results/resultsN holds the records of every stage.

A run writes only the stages it runs. Every other stage is copied from the previous results
directory (which is itself complete), so the latest directory alone holds everything the figures
need, and plot.py reads a single directory.

What is copied, per stage not rerun:
- the records of its units (``<name>.inherited.NNN.jsonl``, read with the directory's own parts;
  a part whose lines are all kept is copied byte for byte, so git stores it once);
- its committed units (``progress.inherited.000.jsonl``);
- its JSON files (tuning: da_delta.json, tuning_choice.json; e2select: e2_budget_choice.json) and
  its key of selection.json (instances: e3, sweep: e5, e2select: e2).
``sources.json`` names the directory each stage comes from, the params that changed since the
previous directory, and the invocations of every source directory.

    python -m mcco_sim.inherit results      # rebuild results2, results3, ... from their predecessors
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .records import RecordStore
from .stages import STAGES, SUPPLEMENTARY_STAGES

MAIN_STAGES = [s for s in STAGES if s not in SUPPLEMENTARY_STAGES]
STAGE_OF_UNIT = {"instances": "instances", "theory": "theory", "tuning": "tuning", "e1": "e1", "sweep": "sweep",
                 "theory_e2": "e2", "e2": "e2", "s1": "s1"}
RECORD_FILES = ["instances.jsonl", "theory.jsonl", "tuning_runs.jsonl", "runs.jsonl"]
JSON_FILES = {"tuning": ["da_delta.json", "tuning_choice.json"], "e2select": ["e2_budget_choice.json"]}
SELECTION_KEYS = {"instances": "e3", "sweep": "e5", "e2select": "e2"}
SOURCES = "sources.json"


def stage_of(unit: str) -> str:
    return STAGE_OF_UNIT[unit.split("/", 1)[0]]


def _unit(line: bytes) -> str | None:
    i = line.find(b'"unit": "')
    if i < 0:
        return json.loads(line).get("unit")
    return line[i + 9:line.index(b'"', i + 9)].decode()


def expand(stages: list[str]) -> list[str]:
    """Pipeline stages of a --stage argument ("all" = the main stages)."""
    out = set()
    for s in stages:
        out |= set(MAIN_STAGES) if s == "all" else ({s} & set(STAGES))
    return [s for s in STAGES if s in out]


def own_invocations(store: RecordStore) -> list[dict]:
    return [json.loads(line) for path in store.own_parts("invocations.jsonl")
            for line in open(path) if line.strip()]


def run_stages(store: RecordStore) -> list[str]:
    """Stages run in this directory (its own invocations)."""
    return expand([s for inv in own_invocations(store) if inv.get("event") == "start"
                   for s in inv.get("stage", "").split()])


def stage_sources(store: RecordStore) -> dict[str, str]:
    """Stage -> name of the results directory its records come from."""
    if store.path(SOURCES).exists():
        return store.read_json(SOURCES)["stages"]
    return {s: store.dir.name for s in run_stages(store)}


def inherited_stages(store: RecordStore) -> set[str]:
    return {s for s, origin in stage_sources(store).items() if origin != store.dir.name}


def add_stages(store: RecordStore, stages: list[str]) -> None:
    """Record stages run in this directory after it was created (sources.json, if any)."""
    if not store.path(SOURCES).exists():
        return
    sources = store.read_json(SOURCES)
    merged = {**sources["stages"], **{s: store.dir.name for s in stages}}
    sources["stages"] = {s: merged[s] for s in STAGES if s in merged}
    store.path(SOURCES).write_text(json.dumps(sources, indent=2))


def _clear(store: RecordStore) -> None:
    """Remove what an earlier inherit() copied (inherit is idempotent)."""
    if store.path(SOURCES).exists():
        for stage in inherited_stages(store):
            for name in JSON_FILES.get(stage, []):
                store.path(name).unlink(missing_ok=True)
        store.path(SOURCES).unlink()
    for path in store.dir.glob("*.inherited.[0-9][0-9][0-9].jsonl"):
        path.unlink()


def inherit(directory: str | Path, previous: str | Path, rerun: list[str]) -> dict:
    """Copy into ``directory`` every stage of ``previous`` that is not in ``rerun`` (the stages
    ``directory`` runs itself). ``directory`` must hold its params.json. Returns sources.json."""
    out, prev = RecordStore(directory), RecordStore(previous)
    _clear(out)
    rerun = expand(rerun)
    kept = {s: origin for s, origin in stage_sources(prev).items() if s not in rerun}

    progress = [line for line in prev.lines("progress.jsonl")
                if line.strip() and stage_of(json.loads(line)["unit"]) in kept]
    units = {json.loads(line)["unit"] for line in progress}
    if progress:
        out.path("progress.inherited.000.jsonl").write_text("".join(progress))
    for name in RECORD_FILES:
        stem, ext = name.rsplit(".", 1)
        k = 0
        for part in prev.parts(name):
            with open(part, "rb") as fh:
                lines = fh.readlines()
            keep = [line for line in lines if line.strip() and _unit(line) in units]
            if not keep:
                continue
            target = out.path(f"{stem}.inherited.{k:03d}.{ext}")
            if len(keep) == len(lines):
                shutil.copyfile(part, target)          # unchanged part: identical bytes
            else:
                target.write_bytes(b"".join(keep))
            k += 1

    for stage in kept:
        for name in JSON_FILES.get(stage, []):
            if prev.path(name).exists():
                shutil.copyfile(prev.path(name), out.path(name))
    selection = out.read_json("selection.json") if out.path("selection.json").exists() else {}
    previous_selection = prev.read_json("selection.json") if prev.path("selection.json").exists() else {}
    for stage, key in SELECTION_KEYS.items():
        if stage in kept:
            selection.pop(key, None)
            if key in previous_selection:
                selection[key] = previous_selection[key]
    if selection:
        out.path("selection.json").write_text(json.dumps(selection, indent=2))

    params, previous_params = out.read_json("params.json"), prev.read_json("params.json")
    invocations = prev.read_json(SOURCES).get("invocations", {}) if prev.path(SOURCES).exists() else {}
    invocations[prev.dir.name] = own_invocations(prev)
    stages = {**kept, **{s: out.dir.name for s in rerun}}
    sources = {
        "inherited_from": prev.dir.name,
        "stages": {s: stages[s] for s in STAGES if s in stages},
        "params_changed": {k: {"before": previous_params.get(k), "here": params.get(k)}
                           for k in sorted(set(params) | set(previous_params))
                           if params.get(k) != previous_params.get(k)},
        "invocations": {origin: invocations[origin] for origin in sorted(set(kept.values())) if origin in invocations},
    }
    out.path(SOURCES).write_text(json.dumps(sources, indent=2))
    return sources


def previous_results(directory: str | Path) -> Path | None:
    """The results directory just before ``<root>/resultsN`` (highest number below N), if any."""
    directory = Path(directory)
    suffix = directory.name[len("results"):]
    if not directory.name.startswith("results") or not suffix.isdigit():
        return None
    below = [(int(p.name[len("results"):]), p) for p in directory.parent.glob("results*")
             if p.is_dir() and p.name[len("results"):].isdigit() and int(p.name[len("results"):]) < int(suffix)]
    return max(below)[1] if below else None


def rebuild(root: str | Path = "results") -> None:
    """Make every results directory under ``root`` complete, in order (results2 from results1, ...)."""
    dirs = sorted((int(p.name[len("results"):]), p) for p in Path(root).glob("results*")
                  if p.is_dir() and p.name[len("results"):].isdigit())
    for (_, prev), (_, cur) in zip(dirs, dirs[1:]):
        sources = inherit(cur, prev, run_stages(RecordStore(cur)))
        print(f"{cur.name}: " + ", ".join(f"{s} <- {o}" for s, o in sources["stages"].items()), flush=True)


if __name__ == "__main__":
    rebuild(sys.argv[1] if len(sys.argv) > 1 else "results")
