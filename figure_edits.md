# Edits to the revision numerics

Edits to make after reviewing the revision figures. Nothing has been changed yet.

## 1. Budget axis: decade ticks (10², 10³, 10⁴, 10⁵)

**Problem.** The budget axis is already logarithmic (`set_xscale("log", base=2)`), but ticks sit on
grid points (`budgets[::step]`). In the E1 grids (`step=4`) that gives labels 50, 800, 12.8k — a factor
16 apart, with three unlabeled points after 12.8k (25.6k, 51.2k, 102.4k). It reads as irregular.

**Change.** Put ticks on powers of 10, independent of the grid:

- In `budget_axis` (l. 76–81): replace `set_xticks(budgets[::step])` / `set_xticklabels(...)` with a
  `matplotlib.ticker.LogLocator(base=10)` and `LogFormatterMathtext()` → labels 10², 10³, 10⁴, 10⁵
  (the range 50–102.4k contains no 10¹, and 0 cannot appear on a log axis).
- Optional: log minor ticks (2–9 × 10ᵏ) instead of `minorticks_off()`.
- Drop the `step` argument (now unused) and the `step=4` in `fig_e1_grid` (l. 176).
- Keep `set_xlim(budgets[0] / 1.3, budgets[-1] * 1.3)`.

**Affected figures** (all go through `budget_axis`): `e1_success`, `e1_distance` (l. 176),
`e3_mismatch` (l. 294), `e5_theory` panels (a) (l. 372).

**Same change on the y axis of `e1_n90`** (`fig_e1_n90`, l. 211–214) for consistency: decade ticks
instead of `budgets[::2]`, keeping the extra "not reached" tick at `top`.

**Not chosen: % of the sample space (n / 2^N).** N = 20 in every experiment, so it is the same axis
under a constant rescaling and adds no information; labels (0.005 % … 10 %) are harder to read.
State the fraction once in the captions instead: n_max = 102.4k ≈ 10 % of 2²⁰.

## 2. MCCO success: best of sample ∪ MP candidates

**Problem.** The success rules differ. Digital annealing returns the best string among *all* its
queries (`best_x` in `digital_annealing`); MCCO returns the best of the ≤ 5 MP candidates only
(`x_hat` in `mcco_sample`, l. 118), ignoring the n sampled strings it has already evaluated.

**Change.** x̂ = argmax of f over (sampled strings ∪ MP candidates), for MCCO (all sketches).

- `mcco_sim/mcco.py`, `mcco_sample`: the sampled values are already known (`full_sample.values`),
  so this costs no extra query. Compute the best sampled string of the prefix and take the max with
  the candidate values before `evaluate_estimate`.
- Keep the MP-only outcome as separate fields (e.g. `success_mp`, `functional_distance_mp`), so the
  contribution of the decoding stays visible.
- No new simulation needed for the existing results: the sample of each run is reproducible from
  `sample_seed` (`integers(0, 2**N, size=n_max)`, prefix of length n) and f from the instance rules.
  The sample-best outcome can be added post hoc to the records (or computed in `plot.py`).
- Update the success definition in `simulation_plan.md` (§3, "Success", l. 139) and in the paper.

**Expected size of the effect.** Sampling is uniform with replacement, so the sample contains a
maximizer with probability 1 − (1 − k/2²⁰)ⁿ for k maximizers: ≈ 1.2 % at n = 12.8k and ≈ 9.3 % at
n = 102.4k for k = 1. The gain is small except for instances with many maximizers.

## 3. Remove the E1 n₉₀ figure

**Reason.** n₉₀ is where the success curves cross the dotted 0.9 line of `e1_success`; the figure
repeats that, and 30 of its 40 entries are "not reached".

**Change (`plot.py`).**

- Delete `fig_e1_n90` (l. 192–222) and its call (l. 478).
- Delete the `e1_n90.csv` output (l. 475).
- With edit 7 (E4 table without n₉₀), nothing uses n₉₀ any more: also delete the n₉₀ computation
  in `e1_aggregates` (l. 144–150, `n90_rows`), `SUCCESS_LEVEL` (l. 38) and its mention in the
  docstring (l. 10) and README; `first_budget_reaching` in `mcco_sim/aggregates.py` becomes unused.
  Keep the dotted 0.9 line of `e1_success` only if wanted (it uses `SUCCESS_LEVEL`).
- Update the module docstring (l. 6–7) and `README.md` (Figures table, `e1_n90` row).
- `simulation_plan.md` l. 197 lists an n₉₀ summary for each method, family and |R|: update or drop.

## 4. E2 figure: percentile axis, success and distance rows

**Change (`fig_e2`, `plot.py` l. 253–277).**

- **x axis:** percentile of t among the 2²⁰ values of f, linear scale, label "threshold percentile
  (% of the 2²⁰ strings)". Take it from the threshold label: `p{p}` → p, `zero` → 0.
- **Drop the points `f_x2` and `mid_x2_max`** from the figure (both sit at ≈ 100 on this axis and are
  not of interest). Remove the `t = f(x₂)` line and annotation (l. 264–267).
- **Dashed line at 85** (= `Q` in `params.py`, read from `res.params.Q`, not hard-coded), labeled
  e.g. "Q = 85". Note for the caption: in E1 the threshold is the 85th percentile of the *n sampled
  values*, an estimate of this exact percentile.
- **Layout:** 2 rows × 2 columns (families), shared x axis.
  - Row 1: success probability (Wilson intervals, as now).
  - Row 2: distance to the optimum of x̂, (f* − f(x̂)) / σ_f, median with IQR band (same statistic as
    `e1_distance`). `functional_distance` is already in the E2 records; add its median and quartiles
    per (instance, sketch, threshold) in `e2_table` (l. 241).
- **`e2_threshold.csv`:** add the percentile column and the distance median / quartiles.

**Interaction with edit 2.** If x̂ includes the best sampled string, that part does not depend on t
and adds a constant to every point. E2 should use the MP-only outcome (`success_mp`,
`functional_distance_mp`), so the figure shows the effect of the threshold on the decoding.

## 5. E3: 3 rules, no unique-maximizer requirement, success and distance

**Current E3 instance** (`e3/L/R5/0`): 5 rules of length 4, unique maximizer, small gap
((f* − f(x₂))/σ_f = 0.11), and the maximum is not preserved by G for either sketch
(`F_argmax_in_maximizers = False`), so success stays at 5–12 %.

**Simulation changes.**

- `params.py`: `E3_R = 5` → `E3_R = 3`. Remove `E3_MAX_DRAWS` (and from the required names in
  `mcco_sim/params.py`).
- `mcco_sim/stages.py` (l. 22–34): drop the unique-maximizer loop; the E3 instance is draw 0,
  `e3_descriptor(params, 0)`, whatever its number of maximizers.
- `simulation_plan.md` §E3: remove "with a unique maximizer", |R| = 3.
- Needs a new run of the E3 parts (E3 instance, its theory checks, the E3 sweep). `params.py` is
  locked per output directory, so this means a new `--out`, or a separate directory for E3.
- E5 panel (a) uses the E3 instance, so it changes too (title, and whether the max is preserved by G).

**Metrics** (same as E1):

- Success: f(x̂) = f*, x̂ = best of sampled strings ∪ MP candidates (edit 2). Multiple maximizers
  are already handled (`success` compares values, not strings).
- Distance: (f* − f(x̂)) / σ_f, median with IQR band.
- x axis: decade ticks (edit 1).

**Figure (`fig_e3`, `plot.py` l. 278–302).** Two panels side by side, shared x axis:
(a) success probability (Wilson intervals, as now), (b) distance to optimum. Title with the new key.
Add the distance median / quartiles per (sketch, n) to `e3_mismatch.csv`.

## 6. E5 panel (a): instances where the bound of Eq. (6) is informative

**Problem.** The panel (a) instances are the E2 instances (median gap at |R| = 5). Their
n* = N ln 2 · (2σ²/Θ² + 2M/(3Θ)), the n at which Eq. (6) falls below 1, is ≈ 1–380 × 2²⁰, so
the bound stays near 2²⁰ ≈ 10⁶ over the whole grid. Across E1 the smallest n* is ≈ 116k, above
N_MAX = 102.4k: no instance shows the bound below 1 on the current grid.

**New instances** (from the existing E1 records, t = exact 85th percentile, M = "valid"):

| | `e1/L/R3/35` | `e1/W/R4/34` |
|---|---|---|
| rules (pattern, reward) | 001001 0.11, 11101 0.06, 1100 0.42 | 0011 0.99, 0\*100 0.13, 001010 0.81, 1\*\*\*01 0.05 |
| maximizers | 1 | 1 |
| (f* − f(x₂)) / σ_f | 0.95 | 1.12 |
| n*, quadruplet | ≈ 118k (0.11 · 2²⁰) | ≈ 193k (0.18 · 2²⁰) |
| n*, quintuplet | ≈ 300k (0.29 · 2²⁰) | ≈ 341k (0.33 · 2²⁰) |
| max preserved by G | yes, both sketches (Θ_min > 0) | yes, both sketches (Θ_min > 0) |

Bound 2^{N(1 − n/n*)} on the grid (orders of magnitude, from rounded n*):

| | 102.4k | 204.8k | 409.6k |
|---|---|---|---|
| L/R3/35 quadruplet | ≈ 6 | ≈ 4·10⁻⁵ | ≈ 10⁻¹⁵ |
| W/R4/34 quadruplet | ≈ 1.3·10³ | ≈ 0.4 | ≈ 2·10⁻⁷ |
| L/R3/35 quintuplet | ≈ 10⁴ | ≈ 80 | ≈ 6·10⁻³ |
| W/R4/34 quintuplet | ≈ 2·10⁴ | ≈ 250 | ≈ 0.06 |

Empirical Problem II failure in E1 (10 runs): 0/10 from 12.8k–25.6k on (quintuplet on L/R3/35:
0/10 at 102.4k). Expected picture: the bound becomes informative, ≈ 10× looser than the observed failure.

**Selection rule** (stated in the paper, not hard-coded keys): per family, among E1 instances with
|R| ≥ 2, a unique maximizer and Θ_min > 0 for both sketches, the one minimizing
max(n*_quadruplet, n*_quintuplet). |R| = 1 instances are excluded: they reach the same n* but are
all the same function up to the reward (single rule 0011 or 1100).

**Changes.**

- `params.py`: new `E5_N_MAX = 409_600` (= N_MIN · 2¹³ ≈ 39 % of 2²⁰), used only by the E5a sweep.
  The E1 grid stays at 102.4k.
- `mcco_sim/stages.py`, `stage_sweep` (l. 106): select the E5 instances from `theory.jsonl` with the
  rule above (theory runs before the sweep), record them in `selection.json` (`"e5"`), and add
  sweep targets `e5_L`, `e5_W` with budgets up to `E5_N_MAX`. The E2 instances stay in the sweep
  (the E2 budget choice uses it).
- `mcco_sim/runner.py`, `unit_sweep` (l. 72): for the E5 roles, only the `q_exact` threshold in
  Problem II mode (`problem2_only`), no decoding: panel (a) uses Problem II only.
- `plot.py`, `fig_e5` (l. 347): panel (a) roles `e5_L`, `e5_W` instead of the E2 instances; budget
  axis up to `E5_N_MAX` (decade ticks, edit 1); the Eq. (7) line test uses this max instead of
  `res.budgets[-1]` (Eq. (7) sizes ≈ 1.17 n*, now inside the range). Remove the third panel
  (E3 instance, max not preserved by G): drop the `"e3"` role from panel (a) in `fig_e5`
  (l. 348) and from `e5a_table`. Panel (a) then has two columns (L, W).
- `simulation_plan.md` §E5 (a): new instances and selection rule, extended budget.
- Run: sweep of the two E5 instances only (theory already in the records), in the new `--out`
  of edit 5.

## 7. E4 cost table: no n₉₀, every method and family

**Problem.** `e4_cost.tex` reports times at n₉₀ only: every MCCO quadruplet and random row, and
MCCO quintuplet for |R| ≥ 2, is "not reached" with no time.

**Change (`e4_tables`, `write_e4_tex`, `plot.py` l. 430–460).**

- Remove the n₉₀ column and the dependence on `n90` (`e4_tables(res)` without the `n90` argument).
- Rows: every method (annealing, MCCO quadruplet, quintuplet, random) × family; columns: the MCCO
  stages (sampling, sketching, decoding, candidates) and total, median wall-clock (s).
- Budget(s) at which times are reported: to decide (e.g. n = 102.4k, or 12.8k and 102.4k).
  `e4_cost_by_budget.csv` keeps every budget.
- Times vary little with |R| (at 102.4k: MCCO totals 6.4–8.6 s, annealing 0.16–0.22 s,
  random 18–21 s): |R| can be pooled or kept as rows — to decide.
- `README.md` (outputs table) and `simulation_plan.md` l. 245 ("median wall-clock per method at
  n_90"): update.

**Note on the comparison.** The MCCO sampling stage (≈ 90 % of the MCCO time) evaluates f through a
Python call per string (`CountingOracle`, via troma `sampling`), while annealing evaluates f by
vectorized array lookups (`f[neighbours]`): ≈ 70 µs vs ≈ 2 µs per query at 102.4k. The wall-clock
gap between MCCO and annealing is mostly this implementation difference, not the algorithms.
