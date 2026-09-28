# MCCO revision — supplementary numerics plan

Scientific Reports manuscript, *A Compressive Sensing Inspired Monte-Carlo Method for
Combinatorial Optimization*. Supplementary experiments, complementing
`simulation_plan.md` (main text). Same generator, seeds, sampling protocol, threshold
rule and records as the main plan unless stated otherwise.

---

## S1 — Greedy decoder vs basis pursuit

**Claim tested.** Methods: greedy decoding is preferred over ℓ1 minimization because
basis pursuit overfits the sample (R1-5).

**Setup.**

- Instances: E1 ensembles, both families. If basis pursuit is too costly on all of them,
  restrict to `|R| = 5`.
- Same samples, thresholds and sketches as E1 (shared sample, paired comparison).
- Decoders: matching pursuit (TrOMA, 5 iterations) as in the main numerics, taken from the E1
  records (same samples, no rerun); nonnegative basis pursuit denoising
  `min Σz s.t. z ≥ 0, ‖Φz − y‖₂ ≤ η`, solved by Chambolle–Pock with a fixed number of iterations
  (`S1_BP_ITERATIONS = 1000` in `params.py`), using only `Φ` and `Φᵀ` products.
- Outcome compared: decoder-only (best of the 5 candidates by `f`), i.e. MP's `success_mp`.
- Candidates: the 5 largest entries of each reconstruction, ranked by `f`, so both
  decoders spend the same number of queries.
- `η`: the expected ℓ2 norm of the Monte-Carlo noise of `y = Σ_s T_t f(s) φ_s` over the `n` draws,
  estimated from the same draws: `η² = Σ_j [Σ_s (T_t f(s) φ_j(s))² − y_j²/n]`.
- Scope: `|R| ∈ S1_R_VALUES = (5,)`, both families, instance ids < `S1_INSTANCES = 20`, samples
  0..`S1_J − 1 = 4`, sketches `S1_SKETCHES` = quadruplet, quintuplet (random dropped: ~21 h per unit),
  budgets `100, 400, …, 102400` (`S1_BUDGET_STRIDE = 2` down from `N_MAX`): 400 units, ~1.5 h on
  16 workers. Run: `python run.py --stage s1 --out results_s1 --workers 16`.

**Metrics.** Success, functional distance, percentile rank vs `n`; decoding wall-clock.

**Figure.** Success rate vs `n`, OMP vs basis pursuit, one panel per sketch.

**Cost note.** Basis pursuit acts on `2^N ≈ 10^6` variables; use an operator-based
first-order solver (only `Φ` and `Φᵀ` products). Time it in the pilot before fixing the
instance set.

---

## S2 — Error budget of the structured sketches

**Claims tested.** Proposition `prop:wh_samp` (coefficient estimation), Lemma `lem:linf`
(ℓ∞ error budget) and the sufficient condition after Proposition `prop:gap`
(`kε + Σ_{S∉K} |a_S| < Δ/2`, ensured by `ε = Δ/(4k)` and `Σ_{S∉K} |a_S| < Δ/4`).

**Scope.** Quadruplet and quintuplet sketches (`K` = set of monomials carried by `Φ`,
`k = |K|`). **Basis: ±1 Walsh characters** `χ_S(x) = Π_{s∈S} (−1)^{x_s}`, so that `â_S` is a
sample mean (Prop. 2) and `|χ_S| = 1` (Lemma 1); Eq. (8) is to be rewritten with `χ_S`.
Instances with a unique maximizer.

### S2a — Coefficient error vs sample size (sampling term `kε`)

- Instances: E2 and E3 instances (`J = 300`), plus the E1 runs (`ε` is cheap to compute).
- Per run: `ε = max_{S∈K} |a_S − â_S|`.
- Thresholding changes the target: the empirical coefficients `â_S` are unbiased for the
  coefficients of `T_t f`, not of `f`. Record the two parts separately, with the run's `t`:
  - sampling error `ε_samp = max_{S∈K} |a_S(T_t f) − â_S|`;
  - thresholding bias `b_t = max_{S∈K} |a_S(f) − a_S(T_t f)|`.
- **Figure.** `ε_samp` vs `n`, log–log, median and 10–90% band over runs, one curve per
  sketch. Overlaid:
  - the Hoeffding/union-bound scaling of Proposition `prop:wh_samp`,
    `∝ ‖T_t f‖_∞ √(ln(2k/δ) / n)`;
  - the horizontal line `Δ/(4k)` and the resulting sample size `O(k² log k / Δ²)`.
  The bias `b_t` is shown as a horizontal line per sketch (it does not decrease with `n`).

### S2b — ℓ1 mass outside `K` (sketching term)

- Instances: all E1 instances with a unique maximizer, and the E3 instance.
- Per instance and per structured sketch, exact from the full WH transform:
  - `L = Σ_{S∉K} |a_S|`;
  - the actual sketching error `‖f − F‖_∞`, where `F` is the truncation of Eq. (8)
    (`L` is its upper bound);
  - the gap `Δ`.
- **Figure.** Scatter of `L / Δ` per instance (family by marker, sketch by panel), with the
  line `L/Δ = 1/4` (sufficient condition) and `‖f − F‖_∞ / Δ = 1/2` (Proposition
  `prop:gap`), colored by the E1 success rate at a fixed `n`. Shows how conservative the
  sufficient condition is compared with actual success.

### S2c — Combined budget (optional panel)

- On the E2/E3 runs: whether `kε + L < Δ/2` holds, split by Problem II success / failure.
  Expected: the condition holds on few runs, while the maximum is preserved on many
  (the condition is sufficient, not necessary).

---

## S3 — Per-instance success distribution

**Purpose.** The E1 success rate is an average over instances. It does not say whether
MCCO succeeds with similar probability on every instance, or always on some instances
and never on others. The per-instance distribution separates the two cases and, in the
second one, shows which instances fail.

**Data.** E1 runs only, no new runs.

- Per instance: success rate over its `J = 10` runs at a fixed `n`, chosen where the
  ensemble success rate is intermediate (same `n` for all methods). With `J = 10`, rates
  take values in steps of 0.1.

**Figure.**

- (a) Histogram of the per-instance success rate, one panel per family, `|R| = 5`
  (other `|R|` in the same layout if informative), one series per method (three
  sketches + digital annealing).
- (b) Per-instance success rate vs relative gap `γ / f*` and vs WH sparsity `s`, marker
  by family, to check whether failures concentrate on small-gap or low-sparsity
  instances.

---

## Reusing the existing data (S2, S3)

S2 and S3 need **no new MCCO or annealing runs**. Everything is computed post hoc from
the records of the main numerics (`results/`, `results_e2/`, `results_v2/` in
`mcco_paper/`). Only numpy is needed (no troma).

### What is stored where

| Data | Location | Fields |
|---|---|---|
| Instances (rules, ground truth) | `<dir>/instances.jsonl` | `rules`, `f_star`, `gap`, `n_maximizers`, `unique_maximizer`, `wh_sparsity`, `t_q_exact` |
| E1 runs (1000 instances × 10 samples × 12 budgets) | `results/runs*.jsonl` | `record = "run"` (adaptive `t`), `record = "problem2"` (fixed `t = t_q_exact`); `sample_id`, `n`, `t`, `success`, `problem2_success`, `sketch` / `method` |
| E1 combined outcome (sample ∪ MP) | `results_v2/figures/sample_best.csv` | `instance_key`, `sample_id`, `n`, `sample_best_x`, `sample_best_f` |
| E2 runs (2 instances, 300 samples, `n = 6400`, 16 thresholds) | `results_e2/runs.jsonl` | `threshold_label` (`zero`, `p50` … `p99.9`, `f_x2`, `mid_x2_max`), `t` |
| E5a runs (2 instances, 300 samples, 14 budgets up to 409.6k) | `results_v2/runs*.jsonl` | `record = "problem2"`, `role = "e5_L"` / `"e5_W"`, fixed `t = t_q_exact` |
| E3 runs | `results_v2/runs*.jsonl` | `role = "e3"`, thresholds `adaptive`, `zero`, `q_exact` |
| Params (seeds, budgets) | `<dir>/params.json` | `MASTER_SEED`, `N_MIN`, `N_MAX`, `E5_N_MAX` |

### Rebuilding `f`, the samples and `K`

- **`f`:** `mcco_sim.instances.compute_spectrum(rules, N)` from the `rules` of the instance
  record (exact, 2²⁰ values). WH coefficients: `mcco_sim.instances.fwht(f) / 2**N`.
- **Sample of a run:** `mcco_sim.posthoc.sample_indexes(params, instance_seed, sample_id,
  n_max, N)`, with `params` read from the run's own directory (`plot.stored_params`). A
  run of budget `n` uses the first `n` indexes. `n_max` is the largest budget of the stage
  that produced the run: `N_MAX` for E1 and E3, `E5_N_MAX` for E5a, the single budget
  (6400) for E2. Run records of type `run` also store it as `sample_n_max`.
- **Threshold:** the run's `t` field, with troma semantics (values `< t` set to 0).
- **`K`:** the quadruplet/quintuplet sketches are 0/1 pattern indicators on windows of `k`
  consecutive bits, so their row span is the set of WH monomials `S` whose support fits in
  one window of `k` consecutive bits. Enumerate these supports once per sketch.

### S2b — per instance, exact

- Instances: the 551 E1 instances with `unique_maximizer = true` (`results/instances.jsonl`).
  Gap `Δ` = `gap` field.
- `L = Σ_{S∉K} |a_S|` from the full WH transform; `F` = inverse WH of the coefficients
  restricted to `K`; `‖f − F‖_∞` directly.
- Color by E1 success at the chosen `n`: per-instance mean of `success` over the 10
  samples (combined outcome, from `sample_best.csv` with the MP candidates of the run
  records, as in `plot.py`).
- Cost: one FWHT of 2²⁰ values per instance and sketch; minutes in total.

### S2a / S2c — per run, recomputed

- Instances: the two **E5a** instances (`e1/L/R3/35`, `e1/W/R4/34`: unique maximizer,
  maximum preserved, fixed exact threshold, budgets up to 409.6k) and the two **E2**
  instances (`e1/L/R5/48`, `e1/W/R5/26`: unique maximizer, `n = 6400`, all thresholds).
  **The E3 instance is excluded:** it has 5 maximizers and `Δ = 0`, so `Δ/(4k)` is
  undefined.
- Per run: rebuild the sample prefix, apply `T_t` with the run's `t`, compute
  `â_S = (1/n) Σ_s T_t f(s) χ_S(s)` for `S ∈ K`; `ε_samp` against the exact coefficients of
  `T_t f`, `b_t` against those of `f` (`b_t` depends only on `t`: once per instance and
  threshold).
- E5a gives `ε_samp` vs `n` at fixed `t` (S2a figure). E2 gives `b_t` and `ε_samp` vs `t` at
  `n = 6400`.
- S2c: per run with `t ≤ f(x₂)` (so the gap of `T_t f` is `Δ`, Prop. 1), the condition applied to
  `T_t f`, `k·ε_samp + L(T_t f) < Δ/2`, and the version written with `f`, `k·ε_f + L(f) < Δ/2`
  (`ε_f = max_K |a_S(f) − â_S|`). Split both by whether `argmax F̂_K = x*` (the event the condition
  guarantees, `F̂_K = Σ_{S∈K} â_S χ_S`) and by the stored `problem2_success` (MCCO's surrogate, a
  different function).
- Extending S2a to all E1 runs (≈ 10⁴ samples × 12 budgets × 2 sketches) is possible with
  the same code but is the only costly part; not needed for the figure.

### S3 — plotting only

- Per-instance success at fixed `n`: E1 combined outcome for MCCO (as in `plot.py`,
  `e1_table`), `success` of the annealing runs (`method = "da"`), 10 runs per instance.
- Instance properties from `results/instances.jsonl`: `gap / f_star`, `wh_sparsity`.
- Choices:
  - `n = 6400` (intermediate ensemble success for `|R| = 5`, same budget as E2);
  - panel (b): the 551 instances with a unique maximizer (449 have `gap = 0`);
  - random sketch: only 10 instances per ensemble, so shown as points or left out of (a).

---

## Records

Added to the per-run records of the main plan: decoder (OMP / basis pursuit), `ε_samp`,
`b_t`. Added to the per-instance records: `L`, `‖f − F‖_∞` per structured sketch.
