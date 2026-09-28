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
    return params.replace(N=10, I=4, I_TUNING=2, J=2, J_SINGLE=6, N_MAX=400, E5_N_MAX=800, BLOCK_SIZE=3,
                          SKETCHES=sketches, S1_SKETCHES=("quintuplet", "random"), S1_INSTANCES=3, S1_J=1)


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
    sweep = tiny.J_SINGLE * n_budgets * 3 * len(single)              # E3, 3 threshold modes
    n_e5 = sum(1 for s in completed.read_json("selection.json")["e5"].values() if s)
    sweep += n_e5 * tiny.J_SINGLE * len(tiny.e5_budgets) * len(single)  # E5a: Problem II rows only
    e2 = 2 * tiny.J_SINGLE * (len(tiny.E2_PERCENTILES) + 3) * len(single)
    s1_pairs = 2 * sum(tiny.S1_INSTANCES if spec["e1_instances"] is None else min(spec["e1_instances"], tiny.S1_INSTANCES)
                       for name, spec in tiny.SKETCHES.items() if name in tiny.S1_SKETCHES) * len(tiny.S1_R_VALUES)
    s1 = s1_pairs * tiny.S1_J * len(tiny.s1_budgets)
    assert _count(completed, "runs.jsonl") == e1 + sweep + e2 + s1
    # stage_e2 adds the grid on the E2 instances, without its t = 0 point (already in theory.jsonl)
    theory = 2 * e1_pairs + 2 * len(single) + 2 * len(single) * (len(tiny.E2_PERCENTILES) + 2)
    assert _count(completed, "theory.jsonl") == theory
    tuning = n_ensembles * tiny.I_TUNING * 27 * tiny.J * n_budgets
    assert _count(completed, "tuning_runs.jsonl") == tuning == 4320


def test_random_sketch_subset(completed):
    random_runs = [r for r in _runs(completed) if r.get("sketch") == "random"]
    assert random_runs
    assert all(r["experiment"] in ("e1", "s1") and r["instance_id"] < RANDOM_K for r in random_runs)
    ids = {(r["instance_key"]) for r in random_runs}
    assert len(ids) == 2 * 5 * RANDOM_K
    theory = [json.loads(line) for line in completed.lines("theory.jsonl")]
    random_theory = [r for r in theory if r["sketch"] == "random"]
    assert random_theory and all(r["ensemble"] == "e1" and r["instance_id"] < RANDOM_K
                                 and r["t_label"] in ("zero", "q_exact") for r in random_theory)


def test_theory_labels_unique(completed):
    keys = [(r["instance_key"], r["sketch"], r["t_label"]) for r in map(json.loads, completed.lines("theory.jsonl"))]
    assert len(keys) == len(set(keys))


def test_e2select_rule(tiny, completed):
    import numpy as np

    choice = completed.read_json("e2_budget_choice.json")
    selection = completed.read_json("selection.json")["e2"]
    runs = [r for r in _runs(completed) if r["experiment"] == "e1" and r["method"] == "mcco"
            and r["record"] == "run" and r["n_rules"] == tiny.E2_R and r["sketch"] in ("quadruplet", "quintuplet")]
    unique = {r["instance_key"] for r in completed.load("instances.jsonl") if r["unique_maximizer"]}
    for family, c in choice.items():
        assert c["budget"] in tiny.budgets and c["instance_key"] == selection[family]["descriptor"]["key"]
        pool = sorted({r["instance_key"] for r in runs if r["family"] == family and r["instance_key"] in unique})
        s = np.array([[[np.mean([r["success_mp"] for r in runs if r["instance_key"] == key
                                 and r["sketch"] == k and r["n"] == n]) for n in tiny.budgets]
                       for k in ("quadruplet", "quintuplet")] for key in pool])
        d = np.abs(s - np.median(s, axis=0)).sum(axis=(1, 2))
        assert c["d_i"] == pytest.approx(d.min()) and c["pool_size"] == len(pool)
        assert c["instance_key"] == pool[int(np.argmin(d))]


def test_e2select_from_other_directory(tiny, completed, tmp_path):
    """instances + e2select + e2 in a new directory, reading the E1 runs of another one."""
    out = Output(tmp_path / "e2", tiny)
    STAGES["instances"](out, workers=1)
    STAGES["e2select"](out, workers=1, e1_records=completed.dir)
    STAGES["e2"](out, workers=1)
    assert out.read_json("selection.json")["e2"] == completed.read_json("selection.json")["e2"]
    assert out.read_json("e2_budget_choice.json")["L"]["budget"] == completed.read_json("e2_budget_choice.json")["L"]["budget"]
    e2 = [r for r in _runs(out) if r["experiment"] == "e2"]
    assert len(e2) == 2 * tiny.J_SINGLE * (len(tiny.E2_PERCENTILES) + 3) * 2
    theory = out.load("theory.jsonl")
    assert len(theory) == 2 * 2 * (len(tiny.E2_PERCENTILES) + 3)       # whole grid: no E1 theory here


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


def test_e3_instance_is_draw_zero(tiny, completed):
    selection = completed.read_json("selection.json")
    assert selection["e3"]["descriptor"]["key"] == f"e3/{tiny.E3_FAMILY}/R{tiny.E3_R}/0"


def test_e5_sweep(tiny, completed):
    selection = completed.read_json("selection.json")["e5"]
    assert set(selection) == set(tiny.FAMILIES)
    e5 = [r for r in _runs(completed) if str(r.get("role", "")).startswith("e5_")]
    assert all(r["record"] == "problem2" and r["threshold_label"] == "q_exact" for r in e5)
    assert {r["n"] for r in e5} <= set(tiny.e5_budgets)
    for family, s in selection.items():
        if s:
            assert s["descriptor"]["n_rules"] >= 2 and max(s["n_star"].values()) == s["max_n_star"]
            assert {r["n"] for r in e5 if r["role"] == f"e5_{family}"} == set(tiny.e5_budgets)


def test_combined_estimate_fields(completed):
    runs = [r for r in _runs(completed) if r["method"] == "mcco" and r["record"] == "run"]
    assert runs
    for r in runs:
        f_mp = r["f_x_hat_mp"]
        assert r["f_x_hat"] == (r["sample_best_f"] if f_mp is None else max(f_mp, r["sample_best_f"]))
        assert r["success"] >= r["success_mp"]


def test_posthoc_matches_recorded(tiny, completed):
    from mcco_sim.posthoc import descriptor_from_record, sample_best_rows

    instances = {r["instance_key"]: r for r in completed.load("instances.jsonl")}
    runs = [r for r in _runs(completed) if r["experiment"] == "e1" and r["method"] == "mcco" and r["record"] == "run"]
    key = runs[0]["instance_key"]
    rows = sample_best_rows(tiny, descriptor_from_record(tiny, instances[key]), list(range(tiny.J)), tiny.budgets)
    recomputed = {(r["sample_id"], r["n"]): (r["sample_best_x"], r["sample_best_f"]) for r in rows}
    recorded = {(r["sample_id"], r["n"]): (r["sample_best_x"], r["sample_best_f"]) for r in runs if r["instance_key"] == key}
    assert recorded == recomputed


def _old_format_copy(completed, target):
    """Copy of an output directory as written before the combined estimate: MCCO runs hold the
    MP-only outcome in success / functional_distance / ..., without *_mp and sample_best fields."""
    import shutil

    target.mkdir()
    for name in ("params.json", "instances.jsonl", "theory.jsonl", "selection.json", "e2_budget_choice.json",
                 "progress.jsonl"):
        shutil.copy(completed.path(name), target / name)
    with open(target / "runs.jsonl", "w") as fh:
        for r in _runs(completed):
            if r["method"] == "mcco" and r["record"] == "run":
                for field in ("x_hat", "x_hat_bits", "f_x_hat", "success", "functional_distance",
                              "percentile_rank", "hamming_distance"):
                    r[field] = r.pop(f"{field}_mp")
                for field in ("sample_best_x", "sample_best_f", "sample_n_max"):
                    r.pop(field)
            fh.write(json.dumps(r) + "\n")
    return target


def test_figures_from_records(completed, tmp_path):
    import plot

    plot.make_figures(completed.dir, tmp_path / "figs")
    for name in plot.FIGURES:
        assert (tmp_path / "figs" / f"{name}.pdf").stat().st_size > 0
    for name in ("e1_success.csv", "e3_mismatch.csv", "e4_cost.csv", "e4_cost.tex", "summary.json"):
        assert (tmp_path / "figs" / name).exists()
    with pytest.raises(SystemExit):                              # never replaces figures silently
        plot.make_figures(completed.dir, tmp_path / "figs")


def test_old_records_give_same_figures(completed, tmp_path):
    """Old-format records (MP-only outcome) + post-hoc best sampled string = the new records."""
    import pandas as pd

    import plot

    old = _old_format_copy(completed, tmp_path / "old")
    plot.make_figures(completed.dir, tmp_path / "new_figs")
    plot.make_figures(old, tmp_path / "old_figs", e3_e5_dir=completed.dir, workers=2)
    for name in ("e1_success.csv", "e2_threshold.csv"):
        new, recomputed = pd.read_csv(tmp_path / "new_figs" / name), pd.read_csv(tmp_path / "old_figs" / name)
        pd.testing.assert_frame_equal(new, recomputed)
    plot.make_figures(old, tmp_path / "old_only", workers=1)      # E3 from old records, post hoc too
    plot.make_figures(old, tmp_path / "split", e2_dir=completed.dir, e3_e5_dir=completed.dir)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "new_figs" / "e2_threshold.csv"),
                                  pd.read_csv(tmp_path / "split" / "e2_threshold.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "new_figs" / "e3_mismatch.csv"),
                                  pd.read_csv(tmp_path / "old_only" / "e3_mismatch.csv"))


def test_s1_basis_pursuit(tiny, completed):
    s1 = [r for r in _runs(completed) if r.get("experiment") == "s1"]
    assert s1 and all(r["decoder"] == "bp" and r["n_rules"] in tiny.S1_R_VALUES for r in s1)
    assert all(r["sketch"] in tiny.S1_SKETCHES and r["instance_id"] < tiny.S1_INSTANCES and r["sample_id"] < tiny.S1_J
               for r in s1)
    assert tiny.s1_budgets == [100, 400] and {r["n"] for r in s1} == set(tiny.s1_budgets)
    # the thresholded sample and budget are those of the E1 run of the same sample
    e1 = {(r["instance_key"], r["sample_id"], r["n"], r["sketch"]): r for r in _runs(completed)
          if r["experiment"] == "e1" and r["record"] == "run" and r["method"] == "mcco"}
    for r in s1:
        twin = e1[(r["instance_key"], r["sample_id"], r["n"], r["sketch"])]
        assert (r["t"], r["n_kept"], r["sample_best_f"]) == (twin["t"], twin["n_kept"], twin["sample_best_f"])
        assert len(r["candidates"]) <= tiny.MP_ITERATIONS
        if "bp_residual" in r:
            assert r["bp_residual_over_eta"] < 1.05


def test_supplementary_figures(completed, tmp_path):
    import supplementary

    supplementary.make_supplementary(completed.dir, tmp_path / "supp", e2_dir=completed.dir,
                                     e3_e5_dir=completed.dir, s1_dir=completed.dir)
    for name in supplementary.FIGURES_SUPP:
        assert (tmp_path / "supp" / f"{name}.pdf").stat().st_size > 0
    for name in ("s2_instances.csv", "s2_runs.csv", "s2c_condition.csv", "s2c_condition.tex", "s1_success.csv",
                 "s1_bp_diagnostics.csv", "s3_per_instance_success.csv", "summary_supplementary.json"):
        assert (tmp_path / "supp" / name).exists()
    with pytest.raises(SystemExit):
        supplementary.make_supplementary(completed.dir, tmp_path / "supp")
