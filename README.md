# MCCO revision numerics

Numerics for the revision of *A Compressive Sensing Inspired Monte-Carlo Method for Combinatorial
Optimization*, following [`simulation_plan.md`](simulation_plan.md). The runs cover E1, E2, E3, the
E5 data, the E4 timings and the digital-annealing tuning of §6. They are built on the TrOMA library
and write JSON-lines records; figures are made from these records only.

## Layout

| Path | Content |
|---|---|
| [`params.py`](params.py) | **every hyperparameter** (N, I, J, N_MIN, N_MAX, Q, sketches, annealing grid, E2/E3/E5, pilot size); the only place values live |
| [`run.py`](run.py) | command-line entry point |
| `mcco_sim/params.py` | loads and validates `params.py` (all names required, no defaults in code) |
| `mcco_sim/seeds.py` | seeds derived from the master seed and the identifiers of each draw |
| `mcco_sim/instances.py` | rule families L and W, exact spectrum and ground truth, E2 selection |
| `mcco_sim/sketches.py` | TrOMA sketch maps and decoders; Φ algebra for Problem II and the theory |
| `mcco_sim/theory.py` | surrogate F, Θ_min, σ², M bounds, Eq. (6)–(7) |
| `mcco_sim/mcco.py` | MCCO runs through TrOMA: sample, threshold, sketch, matching pursuit |
| `mcco_sim/annealing.py` | digital-annealing baseline |
| `mcco_sim/stages.py`, `runner.py`, `records.py` | stages, work units and process pool, output directory |
| `mcco_sim/checks.py`, `pilot.py` | correctness checks; pilot run and compute estimate |
| `tests/` | pytest: the checks, params loading, every stage on tiny params |
| [`pyproject.toml`](pyproject.toml) | dependencies (Python ≥ 3.11, pinned numpy/scipy, TrOMA), package and pytest settings |

## Environment

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[test]"
```

Dependencies are declared in [`pyproject.toml`](pyproject.toml), which also installs `mcco_sim` as
an editable package. TrOMA comes from `../troma_lib`, branch `perf/vectorized-marginals`, through
`[tool.uv.sources]` (uv). Once the branch is pushed, that line becomes a pinned git commit that pip
understands too. The branch adds the vectorized sketch marginals, sparse sketching with `ExplicitSketchMap`, and
`DitString.from_integers`.

## Running

Edit `params.py`, then:

```bash
python run.py --stage selftest                          # correctness checks (~5 s)
python run.py --stage pilot --out pilot                 # checks + timings + compute estimate
python run.py --stage all --out results --workers 16    # everything
python -m pytest                                        # test suite (~30 s)
```

- **Stages:** `instances`, `theory`, `tuning`, `e1`, `sweep` and `e2`. `all` runs them in this
  order, and each can also be run on its own.
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

## Outputs (`--out`)

| File | Content |
|---|---|
| `params.json`, `params.py` | the hyperparameters of this directory (resolved values and file copy) |
| `invocations.jsonl` | per invocation: argv, git commit/dirty of `troma_lib` and `mcco_paper`, versions, hardware |
| `instances.jsonl` | one record per instance (rules, ground truth, gap, WH sparsity, maximizers, seeds) |
| `selection.json` | E2 instances (closest to median gap and sparsity) and E3 instance |
| `theory.jsonl` | per instance × sketch × threshold: max preserved by G, Θ_min, σ², M bounds, Eq. (6)/(7) |
| `da_delta.json`, `tuning_runs.jsonl`, `tuning_choice.json` | digital-annealing δ per ensemble, tuning grid runs, chosen setting |
| `runs.jsonl` | one record per MCCO run (method `mcco`) or annealing run (`da`); `record: problem2` rows hold Problem II only (E1, fixed t = exact Q-th percentile) |
| `e2_budget_choice.json` | E2 budget per instance with the t = 0 success table |
| `progress.jsonl` | committed work units; readers ignore records of uncommitted units |

Seeds: every record holds the identifiers and the derived 64-bit seeds (`instance_seed`,
`sample_seed`, `sketch_seed`, `run_seed`). `numpy.random.default_rng(seed)` reproduces a single run
(see `derive_seed`, `generate_rules`, `mcco_sample`, `digital_annealing`).
