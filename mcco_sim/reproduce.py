"""Reproduction check: compare the records of a rerun with those of the results directory it
reproduces (``run.py --reproduce``), unit by unit.

Two runs of the same unit with the same params give the same records up to their timings
(``time``); floats are compared to 10 significant digits. Records written before the combined
estimate (MCCO runs holding the MP-only outcome under success, f_x_hat, ...) are compared on the
MP-only fields (``*_mp``) of the rerun.

    python -m mcco_sim.reproduce results/results5 results/results5_repro
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from .inherit import RECORD_FILES, SELECTION_KEYS, JSON_FILES, stage_of, stage_sources
from .records import RecordStore

VOLATILE = {"time"}
JSON_VOLATILE = {"e1_records"}                   # absolute path of the directory e2select read
# Outcome fields of MCCO runs: MP-only in records written before the combined estimate.
OUTCOME = ("x_hat", "x_hat_bits", "f_x_hat", "success", "functional_distance", "percentile_rank", "hamming_distance")
COMBINED_ONLY = ("sample_best_x", "sample_best_f", "sample_n_max")


def _round(value):
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def _mcco_run(rec: dict) -> bool:
    return rec.get("method") == "mcco" and rec.get("record") == "run"


def _normal(rec: dict, mp_only: bool) -> dict:
    """The fields compared: no timings; with ``mp_only``, MCCO runs keep their MP-only outcome only."""
    rec = {k: v for k, v in rec.items() if k not in VOLATILE}
    if mp_only and _mcco_run(rec):
        if "success_mp" not in rec:                  # old record: outcome fields hold the MP-only outcome
            rec = {(f"{k}_mp" if k in OUTCOME else k): v for k, v in rec.items()}
        else:
            rec = {k: v for k, v in rec.items() if k not in OUTCOME and k not in COMBINED_ONLY}
    return _round(rec)


def _old_format(store: RecordStore) -> bool:
    """Whether the MCCO runs of ``store`` hold the MP-only outcome only (before the combined estimate)."""
    for line in store.lines("runs.jsonl"):
        if '"method": "mcco"' in line and '"record": "run"' in line:
            return "success_mp" not in json.loads(line)
    return False


def _unit_hashes(store: RecordStore, name: str, stages: set[str], mp_only: bool) -> dict[str, str]:
    done = store.done_units()
    records = defaultdict(list)
    for line in store.lines(name):
        rec = json.loads(line)
        unit = rec.get("unit")
        if unit in done and stage_of(unit) in stages:
            records[unit].append(json.dumps(_normal(rec, mp_only), sort_keys=True))
    return {u: hashlib.sha256("\n".join(sorted(r)).encode()).hexdigest() for u, r in records.items()}


def _first_difference(a: RecordStore, b: RecordStore, name: str, unit: str, mp_only: bool) -> str:
    def recs(store):
        return sorted((_normal(json.loads(line), mp_only) for line in store.lines(name)
                       if f'"unit": "{unit}"' in line), key=lambda r: json.dumps(r, sort_keys=True))
    ra, rb = recs(a), recs(b)
    if len(ra) != len(rb):
        return f"{len(ra)} vs {len(rb)} records"
    for x, y in zip(ra, rb):
        keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
        if keys:
            return ", ".join(f"{k}: {str(x.get(k))[:40]} vs {str(y.get(k))[:40]}" for k in keys[:3])
    return "order of records"


def _json_part(store: RecordStore, name: str, key: str | None = None):
    if not store.path(name).exists():
        return None
    value = store.read_json(name)
    if key is not None:
        value = value.get(key)
    if isinstance(value, dict):
        value = {k: ({kk: vv for kk, vv in v.items() if kk not in JSON_VOLATILE} if isinstance(v, dict) else v)
                 for k, v in value.items()}
    return _round(value)


def compare(original: str | Path, rerun: str | Path, stages: list[str] | None = None, examples: int = 3) -> dict:
    """Unit-by-unit comparison of the records of ``stages`` (default: every stage of ``rerun`` that
    it ran itself) between ``original`` and ``rerun``."""
    a, b = RecordStore(original), RecordStore(rerun)
    if stages is None:
        stages = [s for s, origin in stage_sources(b).items() if origin == b.dir.name]
    mp_only = _old_format(a) or _old_format(b)
    report = {"original": str(a.dir), "rerun": str(b.dir), "mp_only_outcome": mp_only, "stages": {}, "files": {}}
    for stage in stages:
        report["stages"][stage] = {"units": 0, "identical": 0, "different": 0, "missing": 0, "extra": 0, "examples": []}
    for name in RECORD_FILES:
        ha, hb = _unit_hashes(a, name, set(stages), mp_only), _unit_hashes(b, name, set(stages), mp_only)
        for unit in sorted(set(ha) | set(hb)):
            entry = report["stages"][stage_of(unit)]
            entry["units"] += 1
            if unit not in hb:
                entry["missing"] += 1
            elif unit not in ha:
                entry["extra"] += 1
            elif ha[unit] == hb[unit]:
                entry["identical"] += 1
            else:
                entry["different"] += 1
                if len(entry["examples"]) < examples:
                    entry["examples"].append({"unit": unit, "file": name,
                                              "difference": _first_difference(a, b, name, unit, mp_only)})
    for stage in stages:
        for name in JSON_FILES.get(stage, []):
            report["files"][name] = _json_part(a, name) == _json_part(b, name)
        if stage in SELECTION_KEYS:
            key = SELECTION_KEYS[stage]
            report["files"][f"selection.json[{key}]"] = (_json_part(a, "selection.json", key)
                                                         == _json_part(b, "selection.json", key))
    report["identical"] = (all(e["units"] == e["identical"] for e in report["stages"].values())
                           and all(report["files"].values()))
    return report


def print_report(report: dict) -> None:
    print(f"[reproduce] {report['rerun']} vs {report['original']}"
          + (" (MCCO runs compared on the MP-only outcome)" if report["mp_only_outcome"] else ""))
    for stage, e in report["stages"].items():
        status = "identical" if e["units"] == e["identical"] else "DIFFERENT"
        print(f"  {stage:<10} {e['identical']}/{e['units']} units identical"
              + (f", {e['different']} different, {e['missing']} not rerun yet, {e['extra']} extra" if status != "identical" else ""))
        for ex in e["examples"]:
            print(f"      {ex['unit']}: {ex['difference']}")
    for name, same in report["files"].items():
        print(f"  {name:<28} {'identical' if same else 'DIFFERENT'}")
    print("[reproduce] " + ("records identical (timings aside)" if report["identical"] else "differences found"))


if __name__ == "__main__":
    print_report(compare(sys.argv[1], sys.argv[2]))
