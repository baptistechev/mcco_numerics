# MCCO revision — simulation plan

Scientific Reports manuscript, *A Compressive Sensing Inspired Monte-Carlo Method for
Combinatorial Optimization*. This document plans the new numerics. Supplementary
experiments are out of scope for now.

---

## 0. Scope

The numerics test the two claims of the paper:

1. MCCO finds the maximum of compressible functions with fewer queries to `f` than
   generic black-box search.
2. The maximum is preserved under sampling and thresholding, as analyzed in Theorem 1,
   Proposition `prop:threshold` and Corollary `cor:threshold`.

| Reviewer point | Addressed by |
|---|---|
| R1-3 / R2-4: many instances, error bars | E1 (ensemble, confidence intervals) |
| R1-4 / R2-4: computational cost | E4 (wall-clock table) |
| R1-5 / R1-6 / R2-6 (edit 8): threshold sweep, report `t` | E2, fixed `q` in §2 |
| R1-5: sketch/rule mismatch | E3 |
| R1-6: baseline tuned comparably | §6 |
| R1-7: Quadruplets missing from Fig. 3 | All sketches in every figure |
| R1-8: discrete baseline, budget matched on real evaluations | Digital annealing, query counting in §2 |
| R2-5: finite-sample guarantee (Theorem 1) | E5 (theory check) |
| R2-3: DP tractability of the benchmark class | Black-box framing, two families (§1); see `paper_revision_summary.md` |

Dual annealing is dropped.

---

## 1. Instance generator

Two families of rules, same lengths, differing only in the pattern alphabet. The solver
is not told which family an instance comes from.

Rule `r ∈ R` has:

- **length** `k_r`, uniform in {4, 5, 6};
- **pattern** `p_r` of length `k_r`;
- **reward** `w_r`, uniform in `(0, 1]` (positive reals).

The rule is applied on consecutive bits at every starting position `j`, with open
boundary (`j = 1, …, N − k_r + 1`, the last window ending on the last bit):
`f(x) = Σ_{r∈R} w_r · Σ_j 1[x_{j..j+k_r−1} matches p_r]`.

**Family L (local).** Pattern uniform in `{0,1}^{k_r}`.

**Family W (wildcards, non-local).** Pattern in `{0,1,*}^{k_r}`, where `*` matches either
bit value. First and last symbols in `{0,1}` (so that `k_r` is the actual range of the
rule), interior symbols uniform in `{0,1,*}`.

A rule with `m_r` fixed symbols contributes at most `(N − k_r + 1) · 2^{m_r}`
Walsh–Hadamard coefficients, so `s ≤ Σ_r (N − k_r + 1) · 2^{m_r}`.

**Parameters (N = 20 to start)**

- `|R| ∈ {1, …, 5}`, one ensemble of `I = 100` instances per family and per value of `|R|`.

**Validity conditions**

- Degenerate maxima are allowed: success is counted as `f(x̂) = f*` (any maximizer).
  The number of maximizers is recorded per instance.
- Quick check (30 draws per value, `N = 20`):

| | `|R|` = 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| L: unique maximizer | 43% | 47% | 60% | 70% | 73% |
| W: unique maximizer | 33% | 60% | 60% | 67% | 77% |
| L: fraction of `x` with `f > 0` (median) | 40% | 68% | 88% | 90% | 93% |
| W: fraction of `x` with `f > 0` (median) | 53% | 93% | 98% | 99% | 100% |
| L: relative gap `γ / f*` (median, unique cases) | 17% | 7% | 6% | 6% | 6% |
| W: relative gap `γ / f*` (median, unique cases) | 10% | 6% | 3% | 2% | 3% |

  The 85th-percentile threshold was nonzero in every draw of both families.
- Record per instance: `f(x*)`, `f(x₂)`, gap `γ = f(x*) − f(x₂)`, WH sparsity `s`
  (exact, from the full WH transform), mean and variance of `f`, fraction of `x` with
  `f(x) > 0`, number of maximizers.

**Ground truth.** Exhaustive enumeration: `2^20 ≈ 10^6` evaluations, negligible.

**Seeds.** Every random draw is seeded and the seed is recorded: rules (lengths,
patterns, rewards), MCCO samples, random sketches, digital annealing runs (initial
state, move choices), tuning instances. Seeds are derived deterministically from the
identifiers of the draw, e.g. `(N, family, |R|, instance id)` for an instance and
`(instance seed, run id)` for a sample or an annealing run, so every run can be
reproduced individually. See §8 for where they are stored.

---

## 2. Methods

**MCCO**

- Sketches: random, quadruplet, quintuplet. Others can be added later.
- Decoder: OMP returning 5 candidates, ranked by `f`. The 5 evaluations are counted in
  the budget.
- Threshold: `t` = 85th percentile of the sampled values (`q = 85%`, adjustable). The
  same sample sets `t` and builds the sketch, matching the "in practice" sentence of the
  adaptive-`t` clause in Methods. Record `t` per run and flag runs with `t = 0`.

**Baseline: digital annealing.** Maximizes `f` (energy `−f`). Parallel trial over the
`N` single-bit flips, with dynamic offset. Each step evaluates `N` candidates and costs
`N = 20` queries; the budget is counted in queries to `f`, so a budget `n` allows
`⌊n / N⌋` steps (2 steps at `n_min = 50`).

- Energy scale `δ`: median of the nonzero `|Δf|` over single flips from random `x`
  (queries spent on this estimate are counted in the budget, or `δ` is fixed per
  ensemble on the tuning instances, see §6).
- Temperature: geometric schedule over the `⌊n / N⌋` steps, from `T₀ = δ / ln 2`
  (uphill move of size `δ` accepted with probability 1/2) to `T_end = δ / ln 100`
  (accepted with probability 1/100).
- Offset increment: `0.1 δ`, reset to 0 after each accepted move. Relevant here
  because positive rewards leave large plateaus where `f = 0`.
- Starting values; checked on the tuning instances of §6 before being fixed.

---

## 3. Sampling protocol

- `I = 100` instances per family and per value of `|R|`, `J = 10` independent runs per instance. `J`
  counts repetitions of the whole MCCO run, each with a fresh sample; `n` is the size
  of each sample.
- Each sample is drawn i.i.d. uniformly **with replacement** at size `n_max`. Smaller
  budgets use its prefixes (valid samples under i.i.d. sampling).
- Budget grid: `n = 50 · 2^j`, `j = 0, …, 11`, i.e. 50 to 102 400 (≈ 0.1 · 2^20),
  12 points.
- The same sample is shared by every sketch and every threshold.
- Digital annealing: `J` independent runs per instance at each budget of the grid.

---

## 4. Metrics

Per run:

- **Success**: `f(x̂) = f*`, where `x̂` is the best of the 5 candidates by `f`.
- **Functional distance** `f* − f(x̂)`, in units of `σ_f` (standard deviation of `f` over
  `{0,1}^N`), as in the original Fig. 3.
- **Percentile rank** of `x̂`: fraction of `x ∈ {0,1}^N` with `f(x) ≤ f(x̂)` (exact, from
  enumeration).
- **Threshold** `t` of the run, and whether `t ≤ f(x₂)` (admissible range assumed in the
  analysis, Methods and `prop:threshold`).
- **Queries** to `f` (actual count, candidates included).
- **Wall-clock**, split into sampling, sketching, decoding, and candidate evaluation.

Aggregates:

- Success rate vs `n`, with 95% intervals from a two-level bootstrap (resample
  instances, then samples within instances).
- `n_90`: smallest budget reaching 90% ensemble success rate.
- Median functional distance and median percentile rank, with IQR.

### Theory checks

These verify the statements of the Theoretical Analysis. Statements that assume a unique
maximizer (Theorem 1, `prop:threshold`, Corollary `cor:threshold`, Proposition `prop:gap`)
are checked only on instances with a unique maximizer.

Per instance and per sketch (all instances, exact):

- **Maximum preserved by `G`**: whether `argmax_x F(x) = x*`, with `F = 2^{−N} G f`
  (hypothesis of Theorem 1), on `f` and on `T_t f` for `t` = 85th percentile of `f` over
  `{0,1}^N` (hypothesis of Corollary `cor:threshold`).
- **Surrogate gap** `Θ_min = F(x*) − max_{x≠x*} F(x)`, on `f` and on `T_t f`.
- Structured sketches: **ℓ1 mass of the coefficients outside `K`**, `Σ_{S∉K} |a_S|`,
  in the basis of Eq. (8), compared with `Δ/4` (sufficient condition after Proposition
  `prop:gap`).

Per run (E2 and E3 instances, `J = 300`):

- **Problem II success**: whether `argmax_x F̃_{S_n}(x) = x*` (the event bounded by
  Theorem 1 / Corollary `cor:threshold`), recorded separately from the OMP output.
- **Bound of Eq. (6)** and **sample size of Eq. (7)**, computed from `Θ_min`, `σ²`, `M`
  of `T_t f`; compared with the empirical failure rate of Problem II vs `n`.
- Structured sketches: **coefficient error** `ε = max_{S∈K} |a_S − â_S|` vs `n`
  (Proposition `prop:wh_samp`), and whether `kε + Σ_{S∉K} |a_S| < Δ/2` (Lemma `lem:linf`
  + Proposition `prop:gap`) on runs where the maximum is and is not preserved.

Computation: `σ²` uses `E_s[f(s)² g^x(s) g^y(s)] = 2^{−N} (Φᵀ (Φ D_{f²} Φᵀ) Φ)_{xy}`, which
only requires the `m × m` matrix `Φ D_{f²} Φᵀ`. `M` is taken as the bound `2m‖f‖_∞` of
the proof of Theorem 1 where the exact maximum over `2^{2N}` pairs is too costly.

---

## 5. Experiments

### E1 — Success vs query budget

- Two families (L, W) × five values of `|R|` (1, …, 5), `N = 20`.
- Same MCCO settings on both families, no per-family tuning.
- Three sketches + digital annealing on the same budget axis (queries to `f`).
- **Figure**: success rate vs `n`, one panel per family and `|R|` (or `|R|` summarized
  by `n_90` if the panels are too many).
- Summary: `n_90` for each method, family and `|R|`. Similar results on L and W are
  the evidence that MCCO works without knowing the structure.

### E2 — Threshold sweep

- One instance per family from the `|R| = 5` ensembles, with a unique maximizer, whose gap `γ` and sparsity `s` are closest
  to the ensemble medians.
- Sweep `t` from 0 (no thresholding) to above `f(x₂)`, on a percentile grid of the
  values of `f`. Mark `t = f(x₂)` (limit of `prop:threshold`) on the plot.
- Fixed budget `n`, taken from the E1 grid where success is neither 0 nor 1 without
  thresholding.
- `J = 300` on these instances, since this is a per-instance probability.
- **Figure**: success probability vs `t`, three sketches.
- Also record `Var(T_t f)` for edit 6, and the theory checks of §4 at each `t`.

### E3 — Sketch/rule mismatch

- Mismatch: the shape of the rules differs from the shape of the sketch.
- The E1 ensembles mix sizes 4–6, so every sketch there is partially mismatched. E3
  isolates the effect with a dedicated instance from the same generator restricted to
  `k_r = 4` (family L, quadruplet rules only), `|R| = 5`, with a unique maximizer.
- Sketches: quadruplet (matched), quintuplet (mismatched), random (reference).
- `J = 300` on this instance.
- **Figure** (or panel): success probability vs `n` for the three sketches.
- Theory checks of §4 on this instance.

### E5 — Theory check (main text)

One figure, two panels, on instances with a unique maximizer.

- **(a) Per instance: failure vs sample size.** On the E2 and E3 instances (`J = 300`),
  empirical probability that Problem II fails, `P(argmax F̃_{S_n} ≠ x*)`, vs `n` on a log
  scale, one curve per sketch. Overlaid: the bound of Eq. (6) computed from `Θ_min`,
  `σ²`, `M` of `T_t f`, and a vertical line at the sample size of Eq. (7) for `δ = 0.1`.
  With the `2^N` union factor and the bound on `M`, the bound is likely above 1 over much
  of the grid; the comparison is then on the exponential decay rate in `n`, which is the
  content of Theorem 1.
- **(b) Across instances: success vs predicted exponent.** For each E1 instance, empirical
  Problem II success rate at a fixed `n` vs the exponent
  `n Θ_min² / (2σ² + (2/3) M Θ_min)` of Eq. (6). Theorem 1 predicts a monotone relation:
  the constants it names should order the instances by difficulty. If computing `σ²` on
  all instances is too costly, use the `|R| = 5` ensembles of both families.

The quantities of Proposition `prop:wh_samp` and Lemma `lem:linf` (`ε` vs `n`, ℓ1 mass
outside `K`) are recorded (§4) but not plotted.

### E4 — Computational cost (table)

- From the E1 runs: median wall-clock per method at `n_90`, split by stage.
- Hardware and implementation versions reported.

### Later — scaling in N (workstation)

- `N` up to 30 with exact ground truth (`≈ 10^9` evaluations per instance), `n_90` vs
  `N` for each method, against the linear-in-`N` sample complexity of Theorem 1.

---

## 6. Hyperparameters

- MCCO: `q = 85%`, OMP with 5 candidates.
- Digital annealing: schedule and offset of §2.
- Tuning instances: 5–10 instances per family and per `|R|`, generated with seeds disjoint from the
  evaluation set. The digital annealing starting values are checked there on a small
  grid (e.g. `T₀`, `T_end` and offset × {0.5, 1, 2}) and the best setting is fixed.
- All values fixed before the evaluation runs and reported in Methods. Nothing is tuned
  on the instances being scored.

---

## 7. Compute estimate

- `f`-evaluations for MCCO: `10 × I × J × n_max ≈ 10 × 100 × 10 × 10^5 = 10^9`
  (shared across sketches and thresholds), plus the digital annealing runs.
- Decodes for E1: `2 (families) × 5 (|R|) × 100 × 10 × 3 (sketches) × 12 (budgets) ≈ 3.6 × 10^5`.
  E2 adds `(#instances) × 300 × 3 × (#thresholds)`.
- Fill in with measured per-decode time at `N = 20` after a pilot run
  (e.g. 10 instances, 2 samples) before launching the full grid.

---

## 8. Outputs

- One record per run: `N, family, |R|, instance_id, sample_id, method, sketch, t, q, n, x̂,
  f(x̂), functional distance, percentile rank, success, t ≤ f(x₂), queries, wall-clock by
  stage`, Problem II success and `ε` where recorded (§4), and the seeds of the
  instance, the sample and, for random sketches and annealing, of the sketch and the run.
- One record per instance: family, patterns, rewards, `f(x*)`, `f(x₂)`, `γ`, `s`, `σ_f`,
  per-instance theory checks of §4,
  mean and variance of `f`, fraction of `x` with `f > 0`, number of maximizers.
- Figures produced only from these records.
