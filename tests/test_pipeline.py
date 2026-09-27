"""Every stage on tiny params: record counts, resume, and refusal of mixed params."""

import json

import pytest

from mcco_sim import records
from mcco_sim.records import Output
from mcco_sim.stages import STAGES


RANDOM_K = 2                                                     # random sketch on instance ids 0, 1


@pytest.fixture(scope="module")
def tiny(params):
    sketches = {name: dict(spec) for name, spec in params.SKETCHES.items()}
    sketches["random"]["e1_instances"] = RANDOM_K
    return params.replace(N=10, I=4, I_TUNING=2, J=2, J_SINGLE=6, N_MAX=400, BLOCK_SIZE=3, SKETCHES=sketches)


@pytest.fixture(scope="module")
def completed(tiny, tmp_path_factory):
    out = Output(tmp_path_factory.mktemp("run"), tiny)
    for stage in STAGES.values():
        stage(out, workers=2)
    return out


def _count(out, name):
    return sum(1 for _ in out.lines(name))


def _runs(out):
    return [json.loads(line) for line in out.lines("runs.jsonl")]


def test_record_counts(tiny, completed):
    n_ensembles = 2 * 5
    n_budgets = len(tiny.budgets)
    single = [s for s, spec in tiny.SKETCHES.items() if spec["single_instance"]]
    # sketch-instance pairs in E1: structured sketches on all I, random on the first RANDOM_K
    e1_pairs = n_ensembles * sum(tiny.I if spec["e1_instances"] is None else spec["e1_instances"]
                                 for spec in tiny.SKETCHES.values())
    e1 = tiny.J * n_budgets * (2 * e1_pairs + n_ensembles * tiny.I)  # MCCO runs + Problem II rows + DA
    sweep = 3 * tiny.J_SINGLE * n_budgets * 3 * len(single)          # 3 instances x 3 threshold modes
    e2 = 2 * tiny.J_SINGLE * (len(tiny.E2_PERCENTILES) + 3) * len(single)
    assert _count(completed, "runs.jsonl") == e1 + sweep + e2
    # E2 instances add the grid without its t = 0 point (already among the E1 thresholds)
    theory = 2 * e1_pairs + 2 * len(single) + 2 * len(single) * (len(tiny.E2_PERCENTILES) + 2)
    assert _count(completed, "theory.jsonl") == theory
    tuning = n_ensembles * tiny.I_TUNING * 27 * tiny.J * n_budgets
    assert _count(completed, "tuning_runs.jsonl") == tuning == 4320


def test_random_sketch_subset(completed):
    random_runs = [r for r in _runs(completed) if r.get("sketch") == "random"]
    assert random_runs
    assert all(r["experiment"] == "e1" and r["instance_id"] < RANDOM_K for r in random_runs)
    ids = {(r["instance_key"]) for r in random_runs}
    assert len(ids) == 2 * 5 * RANDOM_K
    theory = [json.loads(line) for line in completed.lines("theory.jsonl")]
    random_theory = [r for r in theory if r["sketch"] == "random"]
    assert random_theory and all(r["ensemble"] == "e1" and r["instance_id"] < RANDOM_K
                                 and r["t_label"] in ("zero", "q_exact") for r in random_theory)


def test_theory_labels_unique(completed):
    keys = [(r["instance_key"], r["sketch"], r["t_label"]) for r in map(json.loads, completed.lines("theory.jsonl"))]
    assert len(keys) == len(set(keys))


def test_choice_files(completed):
    for name in ("selection.json", "da_delta.json", "tuning_choice.json", "e2_budget_choice.json"):
        assert completed.path(name).exists()
    choice = completed.read_json("e2_budget_choice.json")
    assert set(choice) == {"L", "W"}


def test_resume_skips_finished_units(tiny, completed, capsys):
    before = _count(completed, "runs.jsonl")
    STAGES["e1"](Output(completed.dir, tiny), workers=1)
    assert "0 to run" in capsys.readouterr().out
    assert _count(completed, "runs.jsonl") == before


def test_parts_split_and_reload(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(records, "PART_BYTES", 200)
    out = Output(tmp_path, tiny)
    recs = [{"unit": f"u{i}", "x": "a" * 40} for i in range(10)]
    for rec in recs:
        out.append("runs.jsonl", [rec])
    out.append("progress.jsonl", [{"unit": rec["unit"]} for rec in recs])
    names = [p.name for p in out.parts("runs.jsonl")]
    assert names[:2] == ["runs.jsonl", "runs.001.jsonl"] and len(names) > 2
    assert all(p.stat().st_size <= 200 for p in out.parts("runs.jsonl"))
    assert out.load("runs.jsonl") == recs


def test_other_params_refused(tiny, completed):
    with pytest.raises(SystemExit):
        Output(completed.dir, tiny.replace(J=3))


def test_seeds_recorded(completed):
    rec = json.loads(next(completed.lines("runs.jsonl")))
    assert {"instance_seed", "unit"} <= set(rec)
    assert "sample_seed" in rec or "run_seed" in rec
