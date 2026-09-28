# MCCO revision numerics

Numerics for the revision of *A Compressive Sensing Inspired Monte-Carlo Method for Combinatorial
Optimization*, following [`simulation_plan.md`](simulation_plan.md). The runs cover E1, E2, E3, the
E5 data, the E4 timings and the digital-annealing tuning of §6. They are built on the TrOMA library
and write JSON-lines records; figures are made from these records only.

## Layout

| Path | Content |
|---|---|
| [`params.py`](params.py) | **every hyperparameter** (N, I, J, N_MIN, N_MAX, Q, sketches, annealing grid, E2/E3/E5, pilot size); the only place values live |
| [`run.py`](run.py) | command-line entry point of the simulation |
| [`plot.py`](plot.py) | figures (PDF) and tables from the records of a results directory |
| `mcco_sim/params.py` | loads and validates `params.py` (all names required, no defaults in code) |
| `mcco_sim/seeds.py` | seeds derived from the master seed and the identifiers of each draw |
| `mcco_sim/instances.py` | rule families L and W, exact spectrum and ground truth, E2 selection |
| `mcco_sim/sketches.py` | TrOMA sketch maps and decoders; Φ algebra for Problem II and the theory |
| `mcco_sim/theory.py` | surrogate F, Θ_min, σ², M bounds, Eq. (6)–(7) |
| `mcco_sim/mcco.py` | MCCO runs through TrOMA: sample, threshold, sketch, matching pursuit |
| `mcco_sim/annealing.py` | digital-annealing baseline |
| `mcco_sim/stages.py`, `runner.py`, `records.py` | stages, work units and process pool, output directory |
| `mcco_sim/checks.py`, `pilot.py` | correctness checks; pilot run and compute estimate |
| `mcco_sim/aggregates.py` | two-level bootstrap, Wilson intervals (used by `plot.py`) |
| `mcco_sim/posthoc.py` | best sampled string of a run (new runs, and recomputation for old records) |
| `mcco_sim/bp.py`, `mcco_sim/walsh.py`, `mcco_sim/s2.py` | S1 basis pursuit; Walsh coefficients and the S2 error budget |
| [`supplementary.py`](supplementary.py) | supplementary figures S1–S3 |
| `tests/` | pytest: the checks, params loading, every stage on tiny params |
| [`pyproject.toml`](pyproject.toml) | dependencies (Python ≥ 3.11, pinned numpy/scipy, TrOMA), package and pytest settings |

## Environment

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[test]"   # or ".[plots]" for plot.py only
```

Dependencies are declared in [`pyproject.toml`](pyproject.toml), which also installs `mcco_sim` as
an editable package. TrOMA is installed from the branch `perf/vectorized-marginals` on GitHub
(baptistechev/TrOMA). That branch adds the vectorized sketch marginals, sparse sketching with
`ExplicitSketchMap`, and `DitString.from_integers`. The install works with uv and plain pip, and the
commit actually installed is recorded in every `invocations.jsonl` entry (`troma_source`).

## Running

Edit `params.py`, then:

```bash
python run.py --stage selftest                          # correctness checks (~5 s)
python run.py --stage pilot --out pilot                 # checks + timings + compute estimate
python run.py --stage all --out results --workers 16    # everything
python run.py --stage instances theory sweep --out results_v2 --workers 48   # several stages
python -m pytest                                        # test suite (~30 s)
```

- **Stages:** `instances`, `theory`, `tuning`, `e1`, `sweep`, `e2select` and `e2`. `all` runs them
  in this order, and each can also be run on its own (`--stage` accepts several).
- **E2 selection:** `e2select` picks the E2 instances and budget from the E1 success curves
  (`e2_selection_edits.md`). It reads the E1 runs of the output directory, or of another one with
  `--e1-records DIR` (read-only).
- **Other params file:** `--params other.py` uses another hyperparameter file.
- **Resuming:** an interrupted stage restarts with the same command. Finished work units are listed
  in `progress.jsonl` and skipped.
- **One params set per directory:** an output directory keeps its params (`params.json` and a copy of
  `params.py`). Different values need a different `--out`.
- **Random sketch on a subset:** it is the cost bottleneck, so `SKETCHES` in `params.py` limits it
  to the first 10 instance ids of each (family, |R|) ensemble in E1 (`"e1_instances"`). It is not
  used in the sweep, E2 or the E3 theory (`"single_instance": False`). Its theory checks cover the
  same E1 subset.
- **Memory:** a worker running the random sketch stores a 512 × 2^20 float64 matrix (4.3 GB), plus
  about 0.5 GB of work arrays. Other units don't build it. Choose `--workers` accordingly.
- **Threads:** BLAS uses one thread per worker unless `--blas-threads` is given.

## Figures

```bash
python plot.py results --e2 results_e2 --e3-e5 results_v2 --workers 8   # E1, E4, E5b from results; E2 from results_e2; E3, E5a from results_v2
python plot.py results_v2 --workers 8                   # everything from one directory
python plot.py results --figures figs                   # choose the output directory
```

`plot.py` only reads the results directories: all parts of the record files (`runs.jsonl`,
`runs.NNN.jsonl`, …) of committed units, and the resolved `params.json`. It never writes into
them except the default `figures/` subdirectory. It refuses to write into a directory that already
holds figures unless `--overwrite` is given, so earlier figures are never replaced by accident.
The default output is `<e3-e5 dir>/figures`, or `<results>/figures` without `--e3-e5`.
`--e2 DIR` and `--e3-e5 DIR` default to the results directory.

The plot-only settings are constants at the top of `plot.py`: bootstrap size and seed, the E5b
budgets, and the M of the Eq. (6) bound. They don't affect the simulation.

**MCCO estimate.** x̂ is the best of the sampled strings and the 5 MP candidates (no extra query),
the same rule as digital annealing.
- New runs record both this outcome and the MP-only one (`*_mp` fields).
- Records written before this change hold only the MP-only outcome. For them, `plot.py` recomputes
  the best sampled string from the recorded seeds (`mcco_sim/posthoc.py`, cached in
  `sample_best.csv`). This takes about 2 min with 8 workers for the E1 records.
- Which outcome each figure uses:
  - E1 and E3: the combined outcome;
  - E2: MP-only, to show the effect of the threshold on the decoding;
  - E5: Problem II.

| Output | Content |
|---|---|
| `e1_success.pdf`, `e1_success.csv` | success rate vs n, 2 × 5 panels (family × \|R\|), two-level bootstrap 95% intervals; MP-only rates in the CSV |
| `e1_distance.pdf` | median distance to the optimum (σ_f units) with IQR; percentile ranks in `e1_success.csv` |
| `e2_threshold.pdf`, `.csv` | success (Wilson intervals) and distance (median, IQR) vs threshold percentile on the two E2 instances at n*, Q marked; Var(T_t f) and Θ_min in the CSV |
| `e3_mismatch.pdf`, `.csv` | (a) success and (b) distance vs n on the E3 instance, quadruplet (matched) vs quintuplet (mismatched) |
| `e5_theory.pdf`, `e5a_problem2.csv`, `e5b_problem2.csv` | (a) Problem II failure vs n on the E5 instances with the Eq. (6) bound and Eq. (7) size; (b) Problem II success vs the predicted exponent on the E1 instances |
| `e4_cost.csv`, `e4_cost.tex`, `e4_cost_by_budget.csv` | median wall-clock per stage, every method × family at n = 102.4k (\|R\| pooled); every budget in `_by_budget` |
| `summary.json` | sources of each figure, aggregate settings and seeds, record counts, E3/E5 instances, Eq. (7) sizes, Spearman correlations of E5b |

## Revision edits (`figure_edits.md`, `e2_selection_edits.md`): what to rerun

The first run stays in `results/`. The new `params.py` (`E3_R = 3`, `E5_N_MAX = 409_600`) differs
from `results/params.json`, and `run.py` refuses to write into a directory with other params. The
new runs therefore go to a new directory:

```bash
python run.py --stage instances theory sweep --out results_v2 --workers 48                        # E3, E5a
python run.py --stage instances e2select e2 --out results_e2 --e1-records results --workers 48    # E2
python plot.py results --e2 results_e2 --e3-e5 results_v2 --workers 8
```

| Edit | New simulation? |
|---|---|
| 1 decade ticks, 3 no n₉₀, 7 E4 table | no: plotted from `results/` |
| 4 E2 layout, and E2 selection (`e2_selection_edits.md`) | **yes** for the new instances and n*: `instances`, `e2select` (reads E1 from `results/`), `e2` (grid theory + runs) in `results_e2`; about 7 min on 48 cores |
| 2 best of sample ∪ MP | no for `results/` (recomputed from the seeds); recorded directly by new runs |
| 5 E3: \|R\| = 3, draw 0 | **yes**: `instances` (E3 instance), `theory` (its checks), `sweep` (E3 runs) |
| 6 E5a: new instances, n up to 409.6k | **yes**: `theory` (the selection reads `theory.jsonl` of the same directory), `sweep` (E5 roles, Problem II only) |

The sweep no longer runs the E2 instances (they only served the old E2 budget choice). The
theory stage recomputes the E1 theory, which is deterministic and identical to `results/`. It
costs about 5 CPU-h, mostly the random sketch on its 100 instances, and is needed for the E5
selection. `tuning`, `e1` and `e2` are not rerun.

## Supplementary numerics (`supplementary_numerics_plan.md`)

```bash
python run.py --stage s1 --out results_s1 --workers 48          # S1: basis-pursuit decoding (workstation)
python supplementary.py results --e2 results_e2 --e3-e5 results_v2 --s1 results_s1 --figures figures_supp --workers 8
```

- **S1** (new runs): nonnegative basis pursuit on the same E1 samples, thresholds and sketches.
  Its records go in `results_s1/`. The matching-pursuit outcomes come from the E1 records.
  `s1` is not part of `--stage all`. Memory: a worker holding the random sketch uses about 5.5 GB.
- **S2** (post hoc, no new runs): error budget of the structured sketches in the ±1 Walsh basis
  (`mcco_sim/walsh.py`, `mcco_sim/s2.py`). The per-instance and per-run tables are cached in the
  figures directory.
- **S3** (plotting only): per-instance E1 success at n = 6400.
- Without `--s1`, S2 and S3 are made and S1 is skipped.

| Output | Content |
|---|---|
| `s1_decoders.pdf`, `s1_success.csv`, `s1_bp_diagnostics.csv` | success vs n, BP vs MP (decoder-only, bootstrap intervals), rows = families, columns = sketches; decode times, η and BP residuals |
| `s2a_coefficient_error.pdf` | ε_samp vs n (E5a instances) and vs threshold (E2 instances), with the Hoeffding + union bound, Δ/(4k) and the bias b_t |
| `s2b_sketching_error.pdf`, `s2_instances.csv` | L/Δ vs ‖f − F_K‖∞/Δ per unique-maximizer E1 instance, colored by E1 success |
| `s2c_condition.csv`, `.tex`, `s2_runs.csv` | fraction of runs with kε + L < Δ/2, split by argmax F̂_K = x* and by Problem II success |
| `s3_success_distribution*.pdf`, `s3_success_vs_properties.pdf`, `s3_per_instance_success.csv` | per-instance success histograms (\|R\| = 5, and all \|R\|), success vs γ/f* and WH sparsity |

## Outputs (`--out`)

| File | Content |
|---|---|
| `params.json`, `params.py` | the hyperparameters of this directory (resolved values and file copy) |
| `invocations.jsonl` | per invocation: argv, TrOMA install source (git URL + commit), git commit/dirty of `mcco_paper`, versions, hardware |
| `instances.jsonl` | one record per instance (rules, ground truth, gap, WH sparsity, maximizers, seeds) |
| `selection.json` | E3 instance, E5a instances (added by the sweep stage), E2 instances (added by `e2select`) |
| `theory.jsonl` | per instance × sketch × threshold: max preserved by G, Θ_min, σ², M bounds, Eq. (6)/(7) |
| `da_delta.json`, `tuning_runs.jsonl`, `tuning_choice.json` | digital-annealing δ per ensemble, tuning grid runs, chosen setting |
| `runs.jsonl` | one record per MCCO run (method `mcco`) or annealing run (`da`); `record: problem2` rows hold Problem II only (E1, fixed t = exact Q-th percentile) |
| `e2_budget_choice.json` | E2 instance and budget n* per family, d_i, pool medians m_k(n) and the instance's s_ik(n) |
| `progress.jsonl` | committed work units; readers ignore records of uncommitted units |

Seeds: every record holds the identifiers and the derived 64-bit seeds (`instance_seed`,
`sample_seed`, `sketch_seed`, `run_seed`). `numpy.random.default_rng(seed)` reproduces a single run
(see `derive_seed`, `generate_rules`, `mcco_sample`, `digital_annealing`).
