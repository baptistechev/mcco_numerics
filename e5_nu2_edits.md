# E5 with the second moment ν² (revised Theorem 1)

Edits to regenerate the E5 figure after the change of Theorem 1. Nothing has been changed yet.

## 0. What changed in the paper

- Theorem 1 now uses the **second moment** `ν² = max_{x≠x*} E_s[Δ_x(s)²]` instead of the variance
  `σ² = max_{x≠x*} Var_s[Δ_x(s)]`, in Eq. (6) and Eq. (7). Bernstein holds with any upper bound on the
  variance, and `Var ≤ E[Δ²]`.
- Corollary 1 uses `ν_t²`, the same quantity computed from `T_t f`.
- Theorem 1 also states that the bounds hold with any upper bound on `M`; E5 already uses such a bound
  (`M["valid"] = 2 ||f||_∞ max |g^{x*} − g^x|`, called `M̄` in Theorem 2).

**Only the theory records change.** The Monte Carlo runs (`runs*.jsonl`: empirical Problem II
failure/success) are unchanged and are not rerun.

Expected effect: for each competitor `E[Δ_x²] = Var[Δ_x] + Θ_x²`, and `Θ_x²` is tiny compared with the variance, so the change is tiny. Measured on the
four E5a cases: ν² − σ² = 0.01–0.05 % of σ², and `n_eq7` shifts by at most ~90 queries:

| instance, sketch | σ² | ν² | n_eq7 (σ²) | n_eq7 (ν²) |
|---|---|---|---|---|
| e1/L/R3/35, quadruplet | 2.9772 | 2.9784 | 138 448 | 138 492 |
| e1/L/R3/35, quintuplet | 1.7733 | 1.7734 | 350 100 | 350 132 |
| e1/W/R4/34, quadruplet | 16.889 | 16.897 | 225 123 | 225 211 |
| e1/W/R4/34, quintuplet | 10.648 | 10.650 | 397 444 | 397 507 |

The stored records cannot be converted directly: they keep only `max_x Var[Δ_x]`, and the competitor
maximizing `E[Δ_x²]` need not be the one maximizing the variance. So the theory stage is recomputed.

## 1. Code changes

### 1.1 `mcco_sim/theory.py`, `theory_quantities` (l. 40–62)

Replace the variance by the second moment and rename the field:

```python
    A = phi.weighted_gram(g ** 2)
    q_diag = phi.quad_diag(A) / size                    # E_s[g(s)^2 g^x(s)^2]
    q_star = phi.adjoint(A @ phi.column(x_star)) / size  # E_s[g(s)^2 g^x*(s) g^x(s)]
    second = q_star[x_star] - 2 * q_star + q_diag        # E_s[Delta_x(s)^2]
    nu2 = float(second[others].max())
    out["nu2"] = nu2
```

and use `nu2` instead of `sigma2` in the exponent and in `n7`:

```python
            exponent = theta_min ** 2 / (2 * nu2 + (2.0 / 3.0) * M * theta_min)
            n7 = (2 * nu2 / theta_min ** 2 + 2 * M / (3 * theta_min)) * (N * math.log(2) + math.log(1 / delta))
```

Docstring: "the constants of Eq. (6)-(7)" stays correct.

### 1.2 `mcco_sim/checks.py`, `check_theory_brute_force` (l. 65–95)

The brute-force check compares `fast["sigma2"]` with the explicit variance. Compare the second moment
instead:

```python
                second = (delta_x ** 2).mean(axis=0)
                ...
                      and np.isclose(fast["nu2"], second[mask].max(), rtol=1e-8, atol=1e-12)
```

and update the docstring ("Theta_min, nu^2 and the M bounds …") and the message (`nu2 …`).
`var` is no longer needed.

### 1.3 `plot.py`

- l. 530, E5b axis label: `"predicted exponent  n Θ²/(2σ² + ⅔MΘ)"` →
  `"predicted exponent  n Θ²/(2ν² + ⅔M̄Θ)"`.
- New option `--theory DIR`: read `theory.jsonl` from DIR and use it in place of the theory records of
  the results directory and of the `--e3-e5` directory (i.e. for E1/E5b and E3/E5a). E2 keeps the
  theory of `--e2` (it only uses `var_sampled_function`, `theta_min`, `max_preserved`, which do not
  change). In `make_figures`, after building `res` and `res_new`:

```python
    if theory_dir:
        th = pd.DataFrame(RecordStore(Path(theory_dir)).load("theory.jsonl"))
        res.theory = th
        res_new.theory = th
```

  with a new argument `theory_dir: Path | None = None` of `make_figures`, passed from `main()`
  (`parser.add_argument("--theory", …)` next to `--e2`, l. 659), and add `"theory_dir"` to `summary.json`. This keeps the rule that results directories are only
  read.

Nothing else reads `sigma2` (checked: `plot.py`, `supplementary.py`, `mcco_sim/*`).

## 2. Recompute the theory records

Run in a **fresh directory**, on the machine of the original runs:

```bash
python run.py --stage selftest
python run.py --stage instances theory --out results_theory_nu2 --workers 48
```

- A fresh directory is needed: in `results` and `results_v2` the theory units are already in
  `progress.jsonl` and would be skipped, and those directories stay untouched.
- Same `params.py` → same instances (each descriptor is checked against its recorded seed).
- Cost: the original theory stage recorded ~16 CPU-hours (quadruplet ~7 h, quintuplet ~6.5 h,
  random ~2.5 h), i.e. ~20 min with 48 workers. It computes `t = 0` and `t = q_exact`; E5 only needs
  `q_exact`, so restricting the stage to it would halve the cost, but is not worth a code change.

## 3. Regenerate the figure

```bash
python plot.py results --e3-e5 results_v2 --e2 results_e2 --theory results_theory_nu2 \
    --figures figures_nu2
```

Only `e5_theory.pdf` (and `e5a_problem2.csv`, `e5b_problem2.csv`) is expected to differ from
`results_v2/figures`; the other figures should be identical.

## 4. Checks

- `selftest` passes, including the brute-force check with ν².
- **E5a instances.** They were selected from the σ² records (`results_v2/selection.json`:
  `e1/L/R3/35`, `e1/W/R4/34`, rule: min over instances of max over sketches of `N ln 2 / exponent`).
  Run `select_e5_instances` on the new records and check that it returns the same instances. With a
  0.05 % change this is expected; if it differs, keep the original instances (the selection rule is a
  choice of example, not part of the theorem) and say so in `summary.json`.
- `n_eq7` of the four E5a cases matches the table of §0.
- E5b Spearman ρ (in `summary.json`) unchanged to the third decimal.

## 5. Optional, same pass: statistics for Theorem 2

Theorem 2 requires that `x*` minimizes `F_t^∁`, the surrogate of the discarded part `f − T_t f`. In
`runner.unit_theory`, for `t > 0` and a unique maximizer, compute it with one extra `apply`/`adjoint`
(~0.2 s per record) and store:

- `discarded_min_at_xstar`: whether `x*` minimizes `F_t^∁` (exactly, up to floating tolerance);
- `discarded_rank`: fraction of `x ≠ x*` with `F_t^∁(x) < F_t^∁(x*)` (0 when the condition holds).

Then report, over the E1 instances with a unique maximizer and for each sketch, the fraction where the
condition holds at `q_exact`, and the fraction where the exponent at `q_exact` is at least the one at
`t = 0`. One sentence in the text, no figure.
