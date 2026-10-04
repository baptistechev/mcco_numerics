# MCCO revision numerics

Numerical experiments for the revision of *A Compressive Sensing Inspired Monte-Carlo Method for
Combinatorial Optimization* (MCCO). The simulation writes every result as records in
`results/`; the figures in `figures/` are made from those records only.

The experiments:

| Experiment | Question | Figure |
|---|---|---|
| E1 | success and distance to the optimum vs the number of queries n, MCCO vs digital annealing | main + supplementary |
| E2 | effect of the threshold t on two typical instances | main |
| E3 | sketch matched vs mismatched to the rule length | supplementary |
| E4 | wall-clock cost of each stage | main (LaTeX table) |
| E5 | Theorem 1: Problem II failure vs the bound of Eq. (6) | supplementary |
| S1 | basis pursuit vs matching pursuit as decoder | supplementary |

---

## 1. Reproducibility

### Everything is seeded

- **One master seed.** `MASTER_SEED` in [`params.py`](params.py) (20260926). Every random draw
  (instance, sample, random sketch, annealing run) has its own seed, derived from the master seed
  and the identifiers of the draw ([`mcco_sim/seeds.py`](mcco_sim/seeds.py)). A draw gets the same
  seed whatever the number of workers or the order in which the work is done.
- **Seeds are written in the records.** `instance_seed` in `instances.jsonl`; `sample_seed`,
  `sketch_seed` (MCCO) and `run_seed` (annealing) in `runs.jsonl`.
- **Params are saved with the results.** Each results directory holds `params.json` (the resolved
  values) and a copy of `params.py`. A run refuses to write into a directory made with other params.
- **The environment is saved too.** `invocations.jsonl` records, for every command: the command
  line, the git commit of this code, the TrOMA commit installed, the Python/numpy/scipy versions and
  the hardware.

### Reproduce a results directory

```bash
python run.py --reproduce --workers 48
```

This reruns every stage of the latest `results/resultsN`, with the `params.py` saved in it (so with
the same seeds), into `results/resultsN_repro`. At the end it compares the two directories record
by record (timings aside), prints the result and saves it as `reproduction.json`:

```
[reproduce] results/results5_repro vs results/results5
  instances  1101/1101 units identical
  theory     1001/1001 units identical
  ...
[reproduce] records identical (timings aside)
```

| Variant | Command |
|---|---|
| only some stages (the others are copied from the original) | `python run.py --reproduce --stage e2 --workers 48` |
| an older results directory | `python run.py --reproduce results/results3 --workers 48` |
| compare two existing directories, no rerun | `python -m mcco_sim.reproduce results/results5 results/results5_repro` |
| check the code against its recorded seeds (~5 s) | `python run.py --stage selftest` |

A full reproduction is the whole simulation (hours on 48 cores); an interrupted one resumes with the
same command. Records are identical up to floating-point details: another CPU or BLAS library can
change the last digits.

### Install

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[test]"     # or ".[plots]" to make the figures only
```

Dependencies are pinned in [`pyproject.toml`](pyproject.toml). TrOMA is installed from GitHub
(`baptistechev/TrOMA`, branch `perf/vectorized-marginals`); the commit used by a run is in its
`invocations.jsonl` (`troma_source`). A [`Dockerfile`](Dockerfile) gives the same environment:

```bash
docker build -t mcco-sim .
docker run --rm -v $(pwd):/app mcco-sim python run.py --stage selftest
```

---

## 2. Run a part of the simulation

```bash
python run.py --stage <stages> --workers 48
```

The run goes to a new directory, `results/results<N+1>` after the latest `results/resultsN`. Only
the stages given are computed; **every other stage is copied from the previous directory**, so the
new directory is complete and the figures can be made from it alone. `run.py` prints what it copied.

| Stage | Computes | For |
|---|---|---|
| `instances` | every instance (rules, exact spectrum, optimum), the E3 instance | all |
| `theory` | Theorem 1 / Corollary 1 quantities per instance and sketch | E5, E3 |
| `e1` | MCCO (every sketch) on the E1 instances | E1, E4, E5b |
| `da` | digital annealing (one fixed setting, `DA_*` in `params.py`) on the E1 instances | E1 |
| `sweep` | 300 runs on the E3 instance and on the E5a instances (chosen from `theory`) | E3, E5a |
| `e2select` | the two E2 instances and budget, from the E1 success curves | E2 |
| `e2` | threshold sweep on the E2 instances | E2 |
| `s1` | basis-pursuit decoding of E1 samples (not part of `all`) | S1 |

Examples:

```bash
python run.py --stage e1 --workers 48                  # rerun the E1 MCCO runs, copy the rest
python run.py --stage da --workers 48                  # rerun the E1 annealing runs only
python run.py --stage instances theory --workers 48    # recompute the instances and the theory
python run.py --stage all --workers 48                 # everything except s1
python run.py --stage s1 --workers 16                  # S1 (memory-bound: ~16 workers is best)
```

Good to know:

- **List every stage to redo in the first command.** A stage copied into a directory cannot be
  rerun there later (`run.py` refuses); use a new directory. Stages depending on a copied stage use
  the copy: rerunning `e1` alone keeps the E2 instances chosen from the old E1 runs.
- **Interrupted?** Run the same command again: it goes back to the unfinished directory and skips
  the work already done (`progress.jsonl`).
- **Changing params.** Edit `params.py` (the only place values live; the code has no defaults).
  If a copied stage depended on a changed value, `run.py` lists the change but still copies it.
- **Other output.** `--out DIR` writes elsewhere (e.g. `--out results/pilot`); such a directory is
  never picked by `plot.py`. `--from DIR` copies from another directory, `--from none` copies nothing.
- **Pilot.** `python run.py --stage pilot --out results/pilot` times a few units and estimates the
  cost of the full run.
- **Memory.** A worker running the random sketch holds a 512 × 2^20 matrix (4.3 GB); other work
  is light. BLAS uses one thread per worker (`--blas-threads` to change).

---

## 3. Make a plot

```bash
python plot.py --workers 8
```

This builds **every figure from the latest `results/resultsN`** into `figures/`, replacing the
previous version.

| Figure | `--only` | Output |
|---|---|---|
| E1 distance to the optimum vs n | `e1` | `figures/main/plot/e1_distance.pdf` |
| E1 success vs n | `e1` | `figures/supplementary/plot/e1_success.pdf` |
| E2 success vs threshold | `e2` | `figures/main/plot/e2_threshold.pdf` |
| E3 matched vs mismatched sketch | `e3` | `figures/supplementary/plot/e3_mismatch.pdf` |
| E4 cost per stage | `e4` | `figures/main/plot/e4_cost.tex` |
| E5 theory check | `e5` | `figures/supplementary/plot/e5_theory.pdf` |
| S1 decoders | `s1` | `figures/supplementary/plot/s1_decoders.pdf` |

Each figure's table is in the matching `data/` folder (same name, `.csv`).

| To | Command |
|---|---|
| make some figures only | `python plot.py --only e2 e5` |
| plot from an older results directory | `python plot.py results/results3 --only e2` |
| redraw after changing the drawing code only (seconds) | `python plot.py --only e5 --from-csv` |
| write somewhere else | `python plot.py --figures figs` |

- `--from-csv` reads the tables already in `figures/` instead of recomputing them from the records.
  Use a normal run when the data or the table computation changed.
- Each `data/summary.json` records which results directory every figure was made from
  (`made_from`), plus the settings and seeds of the statistics.
- Plot-only settings (bootstrap size and seed, E5 panels) are constants at the top of
  [`plot.py`](plot.py); they don't affect the simulation.
- **MCCO estimate.** E1 and E3 use the best of the sampled strings and the 5 matching-pursuit
  candidates; E2 the matching-pursuit candidates only; E5 Problem II; S1 the decoders alone. E1
  records written before this rule hold the matching-pursuit outcome only: `plot.py` recomputes the
  best sampled string from their seeds once (~2 min with 8 workers) and caches it in
  `figures/main/data/sample_best.csv`.

---

## 4. Repository architecture

```
params.py          every hyperparameter (the only place values live)
run.py             runs the simulation (stages, --reproduce)
plot.py            makes every figure and table from one results directory
mcco_sim/          the simulation package
tests/             pytest: checks, params, every stage on tiny params, figures
results/           results1/, results2/, ...: the records of each run
figures/           main/ and supplementary/, each with data/ (CSV) and plot/ (PDF)
Dockerfile, pyproject.toml
```

### `mcco_sim/`

| Module | Role |
|---|---|
| `params.py` | loads and checks `params.py` (every name required) |
| `seeds.py` | seeds derived from the master seed and the identifiers of each draw |
| `instances.py` | rule families L and W, exact spectrum and optimum, instance selection |
| `sketches.py` | TrOMA sketch maps and decoders; sketch algebra for Problem II and the theory |
| `theory.py` | Theorem 1 / Corollary 1 quantities: surrogate F, Θ_min, ν², M bounds, Eq. (6)–(7) |
| `mcco.py` | one MCCO run: sample, threshold, sketch, matching pursuit |
| `annealing.py` | digital-annealing baseline (fixed T0, T_end and offset increment for every instance) |
| `bp.py` | basis-pursuit decoder (S1) |
| `posthoc.py` | best sampled string of a run, recomputed from its seed |
| `stages.py`, `runner.py` | the stages, split into work units run by a process pool |
| `records.py` | results directories: records, committed units, params, provenance |
| `inherit.py` | copies the stages not rerun from the previous results directory |
| `reproduce.py` | compares two results directories record by record |
| `checks.py`, `pilot.py` | correctness checks (`selftest`); pilot run and cost estimate |
| `aggregates.py` | bootstrap and Wilson intervals |

### A results directory

| File | Content |
|---|---|
| `params.json`, `params.py` | the params of this run (resolved values and file copy) |
| `invocations.jsonl` | every command run here: command line, code and TrOMA commits, versions, hardware |
| `sources.json` | which results directory each stage comes from (when some were copied) |
| `instances.jsonl` | one record per instance: rules, optimum, gap, maximizers, seed |
| `theory.jsonl` | per instance, sketch and threshold: Θ_min, ν², M bounds, Eq. (6)/(7) |
| `runs.jsonl` | one record per MCCO or annealing run (E1, sweep, E2, S1), with its seeds |
| `selection.json`, `e2_budget_choice.json` | E3, E5a and E2 instances; E2 budget |
| `progress.jsonl` | finished work units (records of unfinished units are ignored) |
| `reproduction.json` | comparison with the original (`*_repro` directories only) |

Large files are split into parts of at most 50 MB (`runs.001.jsonl`, ...). Copies from the previous
directory are named `*.inherited.NNN.jsonl`.

### The results so far

| Directory | Ran | Why |
|---|---|---|
| `results1` | every stage (2026-09-26) | first run |
| `results2` | `instances theory sweep` (2026-09-27) | revised E3 (\|R\| = 3) and E5a (n up to 409.6k) |
| `results3` | `instances e2select e2` (2026-09-27) | new E2 instance selection |
| `results4` | `s1` (2026-09-28) | basis pursuit |
| `results5` | `instances theory` (2026-10-03) | Theorem 1 with ν² instead of σ² |

Each directory also holds everything it did not rerun, so `results5` is the complete, latest state.

results1–5 come from the older code: digital annealing was tuned per family and |R| (an energy
scale δ from extra tuning instances, a `tuning` stage, `da_delta.json`, `tuning_choice.json`). The
current code uses one fixed setting for every ensemble, like MCCO, in its own `da` stage. Copying
from those directories drops their `tuning` stage, and their annealing runs (inside the `e1` units)
count as stage `da`: `--stage da` replaces them and keeps their MCCO runs. Reproducing them (`--reproduce`) needs the commit before this change,
because their `params.py` has the tuning names.
Older notes and logs use the former names: `results` = results1, `results_v2` = results2,
`results_e2` = results3, `results_s1` = results4, `results_theory_nu2` = results5.
