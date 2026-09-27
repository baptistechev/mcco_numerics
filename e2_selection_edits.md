# E2: selection of typical instances

Nothing has been changed yet.

## Problem

The E2 instances are chosen for their gap and WH sparsity being close to the ensemble medians
(`select_e2_instances`, `mcco_sim/instances.py` l. 195–222). These two quantities don't predict
success. In L the sparsity is almost constant (IQR = 1.5), so the choice is made on the gap alone.
Measured by their E1 success, the current instances are not typical:

- **L/32** ranks 24th of 67 by distance to the ensemble median success curve (defined below), and
  is easier than typical: at every budget its success is between the 57th and 84th percentile of
  the ensemble.
- **W/34** is typical in success (5th of 68), but its quadruplet E2 curve depends on a rare
  case: with the quadruplet sketch, it is 1 of the 33 W instances where G does not keep the maximum
  at t = 0, and the only one of them where thresholding at the 85th percentile makes G keep it.

The E2 budget is also chosen per instance from its t = 0 sweep (`choose_e2_budgets`,
`mcco_sim/stages.py` l. 123–144). For W no budget brings t = 0 success near 0.5, so the choice fell
to n = 102 400.

## New rule

The instance should have a typical success probability over the ensemble. The E1 records already
hold this: every E1 instance has J = 10 runs at the adaptive threshold (85th percentile of the
sampled values), at every budget of the grid.

- **Pool:** per family, the E1 instances with `n_rules = E2_R` and a unique maximizer (67 in L,
  68 in W).
- **Success rate:** s_ik(n) = MP-only success rate of instance i, sketch k (single-instance
  sketches), budget n, over the J runs of E1 (`experiment = "e1"`, `method = "mcco"`,
  `record = "run"`, `threshold_mode = "adaptive"`). MP-only, the same outcome as the E2 figure:
  the field `success_mp`, or `success` in records written before edit 2 of `figure_edits.md`.
- **Budget:** m_k(n) = median of s_ik(n) over the pool; n* = the budget where the mean of m_k(n)
  over the sketches is closest to 0.5 (ties: smaller n).
- **Instance:** the one closest to the median curve, d_i = Σ_k Σ_n |s_ik(n) − m_k(n)| (ties:
  smaller instance id). The whole grid is used because at J = 10 a rate at one budget has a
  standard error of about 0.15. At n* alone the rule would pick W/83, which is far from the median
  at the other budgets.

Result on `results/`:

| Family | n* | Instance | d_i | Median d_i of the pool | Success at n* (quad., quint.) | Ensemble median at n* |
|---|---|---|---|---|---|---|
| L | 6400 | **L/48** | 1.30 | 6.9 | 0.3, 0.4 | 0.4, 0.7 |
| W | 6400 | **W/26** | 1.45 | 6.8 | 0.3, 0.3 | 0.4, 0.4 |

## Code changes

1. **`mcco_sim/instances.py`:** replace `select_e2_instances` with a function taking the instance
   records and the E1 run records, implementing the rule above. It returns, per family, the
   descriptor, n*, d_i, the pool size, m_k(n) and s_ik(n) of the chosen instance.
2. **`mcco_sim/stages.py`:**
   - New stage `e2select`, between `sweep` and `e2` in `STAGES`. It reads the E1 records (of `out`, or
     of `--e1-records DIR`), writes `selection["e2"]` and `e2_budget_choice.json` (same format: the
     plot reads `[family]["budget"]`; add the ensemble medians and the rule). Written once, like
     the other choice files. With `--e1-records`, check that the pool's `instance_seed` values agree
     between the two directories.
   - Reading the E1 records: filter the lines on `"experiment": "e1"` and `"n_rules": <E2_R>`
     before `json.loads` (the runs files hold ~600 MB).
   - `stage_instances` (l. 24–28): write only `selection["e3"]`.
   - `stage_theory` (l. 33–36): no E2 grid (the E2 instances are not known yet).
   - `stage_sweep` (l. 114): drop the `e2_` targets. They only served `choose_e2_budgets`.
   - Delete `choose_e2_budgets`.
   - `stage_e2`: first the theory of the E2 grid on the two instances (units
     `theory_e2/<key>`, single-instance sketches, grid labels not already in `theory.jsonl`), then
     the E2 runs at n* as now.
3. **`mcco_sim/runner.py`, `unit_theory`:** a payload option to compute only the E2 grid labels
   given by the stage.
4. **`run.py`:** option `--e1-records DIR` (read-only, used by `e2select`). Docstring: stage list.
5. **`plot.py`:** option `--e2 DIR` for the directory holding E2 (default: `results`); the `sources` of
   `summary.json` and the module docstring follow.
6. **`tests/test_pipeline.py`:** record counts (sweep without the two E2 instances, theory with the
   E2 grid from `stage_e2`), `e2select` in the stage list, `e2_budget_choice.json` fields.
7. **Docs:** `README.md` (stages, outputs, what to rerun) and `simulation_plan.md` §5 E2 (selection
   rule, budget from the ensemble).

## Run

`results/` holds the E1 records but cannot be opened with the current `params.py` (different params).
Use a new directory, reading E1 from `results/`:

```bash
python run.py --stage instances e2select e2 --out results_e2 --e1-records results --workers 48
python plot.py results --e2 results_e2 --e3-e5 results_v2 --workers 8
```

`instances` takes ~2 min on 48 workers. `e2` is 24 units plus the theory of 2 instances, ~7 min.
Nothing else is rerun.
