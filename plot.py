"""Figures and tables of the MCCO revision, from the records of results directories only.

    python plot.py results --e2 results_e2 --e3-e5 results_v2   # E1, E4, E5b from results; E2 from
                                                                # results_e2; E3, E5a from results_v2
    python plot.py results_v2                      # everything from one directory
    python plot.py results --figures figs          # other output directory
    python plot.py results --e2 results_e2 --e3-e5 results_v2 --theory results_theory_nu2
                                                   # theory records (E1, E3, E5) from results_theory_nu2

Figures (vector PDF): e1_success, e1_distance, e2_threshold, e3_mismatch, e5_theory.
Tables: e1_success.csv, e2_threshold.csv, e3_mismatch.csv, e4_cost.csv/.tex, e4_cost_by_budget.csv,
e5a_problem2.csv, e5b_problem2.csv, and summary.json (settings and seeds of the aggregates).

Result directories are only read. Figures go to --figures (default: FIGURES under the E3/E5
directory if given, else under the results directory); an output directory that already holds
figures is refused unless --overwrite, so earlier figures are never replaced by accident.

MCCO estimate (figure_edits.md, edit 2): best of the sampled strings and the matching-pursuit
candidates. Records written before that change hold the MP-only outcome; for them the best sampled
string is recomputed from the recorded seeds (mcco_sim/posthoc.py, cached in sample_best.csv).
E1 and E3 use this combined outcome; E2 uses the MP-only outcome (effect of the threshold on the
decoding); E5 uses Problem II.

The plotting settings below only affect the figures, not the simulation.
"""

from __future__ import annotations

import os

# One BLAS thread per process (the process pool provides the parallelism), set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import json
import math
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from mcco_sim.aggregates import percentile_interval, two_level_bootstrap, wilson_interval  # noqa: E402
from mcco_sim.params import doubling_grid  # noqa: E402
from mcco_sim.posthoc import sample_best_task  # noqa: E402
from mcco_sim.records import RecordStore  # noqa: E402

# --- Plotting settings --------------------------------------------------------------------------
BOOTSTRAP_SAMPLES = 2000        # two-level bootstrap replicates (95% intervals)
BOOTSTRAP_SEED = 20260927
E5B_BUDGETS = (800, 6400, 51200)  # budgets of the E5b panels (nearest grid budgets if absent)
E5_M_BOUND = "valid"            # M used for the Eq. (6) bound: "valid" or "plan_2m_sup"

# --- Style: fixed color per entity (validated categorical slots 1-3; annealing is the neutral
# baseline), plus a marker per method so identity never relies on color alone. ---------------------
METHODS = ["quadruplet", "quintuplet", "random", "annealing"]
LABELS = {"quadruplet": "MCCO quadruplet", "quintuplet": "MCCO quintuplet", "random": "MCCO random",
          "annealing": "Digital annealing"}
COLORS = {"quadruplet": "#2a78d6", "quintuplet": "#eb6834", "random": "#1baf7a", "annealing": "#52514e"}
MARKERS = {"quadruplet": "o", "quintuplet": "s", "random": "^", "annealing": "D"}
TEXT, TEXT_2, RULE = "#0b0b0b", "#52514e", "#dcdbd7"
FULL_WIDTH, HALF_WIDTH = 7.2, 3.5      # inches: Scientific Reports double and single column

FIGURES = ["e1_success", "e1_distance", "e2_threshold", "e3_mismatch", "e5_theory"]
RUN_FIELDS = ["record", "method", "experiment", "role", "family", "n_rules", "instance_id", "instance_key",
              "sample_id", "run_id", "sketch", "threshold_mode", "threshold_label", "t", "n", "n_kept", "queries",
              "success", "functional_distance", "percentile_rank", "f_x_hat",
              "success_mp", "functional_distance_mp", "percentile_rank_mp", "f_x_hat_mp",
              "sample_best_f", "problem2_success", "time"]


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": TEXT_2,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def budget_label(b: float) -> str:
    return f"{b:g}" if b < 1000 else f"{b / 1000:g}k"


def budget_axis(ax, budgets: list[int]) -> None:
    """Log axis with decade ticks (10^2 ... 10^5) and unlabeled minor ticks, independent of the grid."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(budgets[0] / 1.3, budgets[-1] * 1.3)


def shared_legend(fig, methods: list[str], y: float = 1.0) -> None:
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(methods),
               handlelength=1.8, columnspacing=1.2)


def band(ax, x, y, low, high, m: str, **kwargs) -> None:
    ax.fill_between(x, low, high, color=COLORS[m], alpha=0.15, linewidth=0)
    ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], markeredgewidth=0, **kwargs)


# =============================================================================
# Loading (read-only)
# =============================================================================

def stored_params(results_dir: Path) -> SimpleNamespace:
    """The resolved params of a results directory, as stored when it was written. Read as plain
    values (no validation against the current params.py), so older directories still load."""
    values = json.loads((results_dir / "params.json").read_text())
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items() if " " not in k}
    params = SimpleNamespace(**values)
    params.budgets = doubling_grid(params.N_MIN, params.N_MAX)
    params.e5_budgets = doubling_grid(params.N_MIN, params.E5_N_MAX) if hasattr(params, "E5_N_MAX") else None
    return params


def normalize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Records written before the combined estimate (edit 2) hold the MP-only outcome in
    success / functional_distance / percentile_rank / f_x_hat: copy it to the *_mp fields and mark
    the combined outcome as missing (filled by ``add_sample_best``)."""
    runs = runs.copy()
    old = (runs.method == "mcco") & (runs.record == "run") & runs.success_mp.isna()
    for field in ("success", "functional_distance", "percentile_rank", "f_x_hat"):
        runs[f"{field}_mp"] = runs[f"{field}_mp"].astype(object)
        runs.loc[old, f"{field}_mp"] = runs.loc[old, field]
        runs.loc[old, field] = None
    runs["combined_missing"] = old
    runs["m"] = runs["sketch"].where(runs["method"] == "mcco", "annealing")
    runs["rep"] = runs["sample_id"].where(runs["method"] == "mcco", runs["run_id"])
    return runs


class Results:
    """One results directory, read through RecordStore (nothing is ever written there)."""

    def __init__(self, results_dir: Path) -> None:
        self.dir = results_dir
        self.store = RecordStore(results_dir)
        self.params = stored_params(results_dir)
        self.budgets = self.params.budgets
        self.runs = normalize_runs(pd.DataFrame(self.store.load("runs.jsonl", fields=RUN_FIELDS)))
        self.theory = pd.DataFrame(self.store.load("theory.jsonl"))
        self.instances = pd.DataFrame(self.store.load("instances.jsonl"))
        self.selection = self.store.read_json("selection.json")
        path = results_dir / "e2_budget_choice.json"
        self.e2_choice = json.loads(path.read_text()) if path.exists() else None

    def instance(self, key: str) -> pd.Series:
        return self.instances[self.instances.instance_key == key].iloc[0]


def add_sample_best(res: Results, experiments: set[str], cache: Path, workers: int,
                    roles: set[str] | None = None) -> None:
    """Fill the combined outcome (best of sample and MP candidates) of old MCCO records of the given
    experiments ("e1", "sweep") by regenerating each run's sample from its seed. Only experiments whose
    samples were drawn for the full budget grid (n_max = N_MAX) qualify; E2 keeps the MP-only outcome."""
    runs = res.runs
    need = runs.combined_missing & runs.experiment.isin(experiments)
    if roles is not None:
        need &= runs.role.isin(roles)
    if not need.any():
        return
    wanted = runs.loc[need, ["instance_key", "sample_id"]].drop_duplicates()
    cached = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["instance_key", "sample_id", "n"])
    have = set(zip(cached.instance_key, cached.sample_id))
    todo = wanted[[(k, s) not in have for k, s in zip(wanted.instance_key, wanted.sample_id)]]
    if len(todo):
        records = {r["instance_key"]: r for r in res.store.load("instances.jsonl")}
        tasks = [(res.params, records[key], sorted(int(s) for s in group.sample_id), res.budgets)
                 for key, group in todo.groupby("instance_key")]
        print(f"[plot] best sampled string of {len(todo)} old samples ({len(tasks)} instances, "
              f"{workers} workers)", flush=True)
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                rows = [r for chunk in pool.imap_unordered(sample_best_task, tasks) for r in chunk]
        else:
            rows = [r for task in tasks for r in sample_best_task(task)]
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(cache, index=False)
    merged = runs.loc[need, ["instance_key", "sample_id", "n"]].merge(
        cached, on=["instance_key", "sample_id", "n"], how="left")
    if merged.sample_best_f.isna().any():
        raise ValueError("Best sampled string missing for some records.")
    inst = res.instances.set_index("instance_key")
    f_star = inst.loc[merged.instance_key, "f_star"].to_numpy(float)
    sigma = inst.loc[merged.instance_key, "sigma_f"].to_numpy(float)
    f_mp = pd.to_numeric(runs.loc[need, "f_x_hat_mp"], errors="coerce").to_numpy(float)
    rank_mp = pd.to_numeric(runs.loc[need, "percentile_rank_mp"], errors="coerce").to_numpy(float)
    f_sample = merged.sample_best_f.to_numpy(float)
    # Ties keep the MP estimate (same value either way); the percentile rank is monotone in f.
    f_hat = np.where(np.isnan(f_mp), f_sample, np.maximum(f_mp, f_sample))
    rank = np.where(np.isnan(rank_mp), merged.sample_best_rank, np.maximum(rank_mp, merged.sample_best_rank))
    runs.loc[need, "sample_best_f"] = f_sample
    runs.loc[need, "f_x_hat"] = f_hat
    runs.loc[need, "success"] = f_hat == f_star
    runs.loc[need, "functional_distance"] = (f_star - f_hat) / sigma
    runs.loc[need, "percentile_rank"] = rank
    runs.loc[need, "combined_missing"] = False


def outcome_cube(df: pd.DataFrame, value: str, budgets: list[int]) -> tuple[np.ndarray, list[str]]:
    """(instances, replicates, budgets) array of a per-run value; checks the design is complete."""
    table = df.pivot_table(index=["instance_key", "rep"], columns="n", values=value, aggfunc="first")
    table = table.reindex(columns=budgets)
    keys = sorted(table.index.get_level_values(0).unique())
    reps = sorted(table.index.get_level_values(1).unique())
    table = table.reindex(pd.MultiIndex.from_product([keys, reps]))
    if table.isna().any().any():
        raise ValueError(f"Incomplete design for {value}: missing (instance, run, budget) records.")
    return table.to_numpy(dtype=float).reshape(len(keys), len(reps), len(budgets)), keys


def quartiles(values) -> tuple[float, float, float]:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    return tuple(float(v) for v in np.quantile(values, [0.25, 0.5, 0.75]))


def rate_table(df: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    """Rate of a boolean per group with a Wilson interval (one instance: binomial trials)."""
    rows = []
    for key, d in df.groupby(by):
        k, trials = int(d[value].astype(bool).sum()), len(d)
        low, high = wilson_interval(k, trials)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                         rate=k / trials, ci_low=low, ci_high=high, trials=trials))
    return pd.DataFrame(rows)


def distance_table(df: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    rows = []
    for key, d in df.groupby(by):
        q25, q50, q75 = quartiles(d[value])
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                         distance_median=q50, distance_q25=q25, distance_q75=q75))
    return pd.DataFrame(rows)


def methods_present(values) -> list[str]:
    return [m for m in METHODS if m in set(values)]


# =============================================================================
# E1: success and distance vs budget
# =============================================================================

def e1_table(res: Results, rng: np.random.Generator) -> pd.DataFrame:
    e1 = res.runs[(res.runs.experiment == "e1") & (res.runs.record == "run")]
    rows = []
    for (family, R, m), df in e1.groupby(["family", "n_rules", "m"]):
        success, keys = outcome_cube(df.assign(success=df.success.astype(float)), "success", res.budgets)
        reps = two_level_bootstrap(success, BOOTSTRAP_SAMPLES, rng)
        low, high = percentile_interval(reps)
        point = success.mean(axis=(0, 1))
        for k, n in enumerate(res.budgets):
            at_n = df[df.n == n]
            d25, d50, d75 = quartiles(at_n.functional_distance)
            r25, r50, r75 = quartiles(at_n.percentile_rank)
            row = {"family": family, "n_rules": R, "method": m, "n": n, "instances": len(keys),
                   "runs_per_instance": success.shape[1], "success": point[k],
                   "success_ci_low": low[k], "success_ci_high": high[k],
                   "distance_median": d50, "distance_q25": d25, "distance_q75": d75,
                   "rank_median": r50, "rank_q25": r25, "rank_q75": r75}
            if m != "annealing":                      # MP-only outcome, for reference
                row["success_mp"] = float(at_n.success_mp.astype(float).mean())
                row["distance_mp_median"] = quartiles(at_n.functional_distance_mp)[1]
            rows.append(row)
    return pd.DataFrame(rows)


def fig_e1_grid(res: Results, table: pd.DataFrame, value: str, path: Path) -> None:
    """2 x |R| grid; value = "success" (bootstrap band) or "distance" (median, IQR band)."""
    families, R_values = list(res.params.FAMILIES), list(res.params.R_VALUES)
    methods = methods_present(table.method)
    fig, axes = plt.subplots(len(families), len(R_values), figsize=(FULL_WIDTH, 1.55 * len(families) + 0.3),
                             sharex=True, sharey=True, squeeze=False)
    for i, family in enumerate(families):
        for j, R in enumerate(R_values):
            ax = axes[i, j]
            for m in methods:
                d = table[(table.family == family) & (table.n_rules == R) & (table.method == m)].sort_values("n")
                if d.empty:
                    continue
                if value == "success":
                    band(ax, d.n, d.success, d.success_ci_low, d.success_ci_high, m)
                else:
                    band(ax, d.n, d.distance_median, d.distance_q25, d.distance_q75, m)
            budget_axis(ax, res.budgets)
            if value == "success":
                ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(f"|R| = {R}")
            if j == 0:
                ax.set_ylabel(f"Family {family}\n" + ("success rate" if value == "success"
                                                      else "distance to optimum (σ_f)"))
            if i == len(families) - 1:
                ax.set_xlabel("queries n")
    shared_legend(fig, methods)
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# E2: threshold sweep (MP-only outcome)
# =============================================================================

def threshold_percentile(label: str) -> float:
    """Percentile of t among the 2^N values of f, from the threshold label (p<P> -> P, zero -> 0)."""
    if label == "zero":
        return 0.0
    if label.startswith("p"):
        return float(label[1:])
    return math.nan                                   # f_x2, mid_x2_max: not on this axis


def e2_table(res: Results) -> pd.DataFrame:
    e2 = res.runs[res.runs.experiment == "e2"]
    by = ["instance_key", "sketch", "threshold_label"]
    table = rate_table(e2, by, "success_mp").merge(distance_table(e2, by, "functional_distance_mp"), on=by)
    extra = e2.groupby(by).agg(t=("t", "first"), n=("n", "first"), mean_kept=("n_kept", "mean")).reset_index()
    table = table.merge(extra, on=by)
    table["percentile"] = table.threshold_label.map(threshold_percentile)
    th = res.theory[["instance_key", "sketch", "t_label", "var_sampled_function", "max_preserved", "theta_min"]]
    th = th.drop_duplicates(["instance_key", "sketch", "t_label"]).rename(columns={"t_label": "threshold_label"})
    return table.merge(th, on=by, how="left").sort_values(["instance_key", "sketch", "t"])


def fig_e2(res: Results, table: pd.DataFrame, path: Path) -> None:
    items = list(res.selection["e2"].items())
    methods = methods_present(table.sketch)
    fig, axes = plt.subplots(2, len(items), figsize=(FULL_WIDTH * 0.62, 3.4), sharex=True, squeeze=False)
    q = res.params.Q
    for col, (family, sel) in enumerate(items):
        key = sel["descriptor"]["key"]
        top, bottom = axes[0, col], axes[1, col]
        for m in methods:
            d = table[(table.instance_key == key) & (table.sketch == m) & table.percentile.notna()]
            d = d.sort_values("percentile")
            band(top, d.percentile, d.rate, d.ci_low, d.ci_high, m)
            band(bottom, d.percentile, d.distance_median, d.distance_q25, d.distance_q75, m)
        for ax in (top, bottom):
            ax.axvline(q, color=TEXT_2, linewidth=0.6, linestyle=(0, (3, 2)))
        top.set_ylim(-0.02, 1.05)
        n = res.e2_choice[family]["budget"]
        top.set_title(f"Family {family}: {key.split('/', 1)[1]}, n = {budget_label(n)}")
        bottom.set_xlim(-2, 102)
    axes[0, 0].set_ylabel("success probability")
    axes[1, 0].set_ylabel("distance to optimum (σ_f)")
    fig.supxlabel(f"threshold percentile (% of the 2$^{{{res.params.N}}}$ strings)", fontsize=7, y=0.02)
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in methods]
    handles.append(plt.Line2D([], [], color=TEXT_2, linewidth=0.6, linestyle=(0, (3, 2)), label=f"Q = {q:g}"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# E3: sketch/rule mismatch (combined outcome)
# =============================================================================

def e3_table(res: Results) -> pd.DataFrame:
    sweep = res.runs[(res.runs.experiment == "sweep") & (res.runs.role == "e3")
                     & (res.runs.threshold_mode == "adaptive")]
    table = rate_table(sweep, ["sketch", "n"], "success").merge(
        distance_table(sweep, ["sketch", "n"], "functional_distance"), on=["sketch", "n"])
    mp = rate_table(sweep, ["sketch", "n"], "success_mp").rename(columns={"rate": "success_mp_rate"})
    table = table.merge(mp[["sketch", "n", "success_mp_rate"]], on=["sketch", "n"])
    key = res.selection["e3"]["descriptor"]["key"]
    th = res.theory[(res.theory.instance_key == key) & (res.theory.t_label == "q_exact")]
    preserved = dict(zip(th.sketch, th.get("max_preserved", th.get("F_argmax_in_maximizers"))))
    table["max_preserved_by_G"] = table.sketch.map(preserved)
    table["instance_key"] = key
    return table


def fig_e3(res: Results, table: pd.DataFrame, path: Path) -> None:
    key = res.selection["e3"]["descriptor"]["key"]
    inst = res.instance(key)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.62, 2.0), sharex=True, squeeze=False,
                             gridspec_kw={"wspace": 0.4})
    names = {"quadruplet": " (matched)", "quintuplet": " (mismatched)"}
    methods = methods_present(table.sketch)
    for m in methods:
        d = table[table.sketch == m].sort_values("n")
        band(axes[0, 0], d.n, d.rate, d.ci_low, d.ci_high, m, label=LABELS[m] + names.get(m, ""))
        band(axes[0, 1], d.n, d.distance_median, d.distance_q25, d.distance_q75, m)
    for ax in axes[0]:
        budget_axis(ax, res.budgets)
        ax.set_xlabel("queries n")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("success probability")
    axes[0, 1].set_ylabel("distance to optimum (σ_f)")
    axes[0, 0].set_title("(a) success")
    axes[0, 1].set_title("(b) distance to optimum")
    fig.suptitle(f"E3: {int(inst.n_rules)} rules of length 4 ({key}, {int(inst.n_maximizers)} maximizer"
                 f"{'s' if inst.n_maximizers > 1 else ''})", fontsize=7, y=1.16)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=len(methods))
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# E5: theory check
# =============================================================================

def e5a_table(res: Results) -> pd.DataFrame:
    """Problem II failure vs n on the E5a instances (fixed exact threshold, Corollary 1), with the
    Eq. (6) bound. Empty if the directory has no E5a sweep."""
    runs = res.runs
    p2 = runs[(runs.experiment == "sweep") & runs.role.fillna("").str.startswith("e5_")
              & (runs.threshold_label == "q_exact")].copy()
    if p2.empty:
        return pd.DataFrame()
    p2["failure"] = ~p2.problem2_success.astype(bool)
    table = rate_table(p2, ["role", "instance_key", "sketch", "n"], "failure")
    th = res.theory[res.theory.t_label == "q_exact"].drop_duplicates(["instance_key", "sketch"])
    N = res.params.N
    bounds, n7 = [], []
    for _, row in table.iterrows():
        r = th[(th.instance_key == row.instance_key) & (th.sketch == row.sketch)]
        exponent = r.eq6_exponent.iloc[0][E5_M_BOUND] if not r.empty and isinstance(r.eq6_exponent.iloc[0], dict) else None
        bounds.append(math.exp(N * math.log(2) - row.n * exponent) if exponent else math.nan)
        n7.append(r.eq7_n.iloc[0][E5_M_BOUND] if exponent else math.nan)
    table["bound_eq6"], table["n_eq7"] = bounds, n7
    return table


def e5b_budgets(res: Results) -> list[int]:
    grid = np.array(res.budgets)
    return sorted({int(grid[np.argmin(np.abs(np.log(grid) - math.log(b)))]) for b in E5B_BUDGETS})


def e5b_table(res: Results) -> pd.DataFrame:
    """Per E1 instance: Problem II success rate at the fixed exact threshold vs the Eq. (6) exponent."""
    p2 = res.runs[(res.runs.experiment == "e1") & (res.runs.record == "problem2")
                  & (res.runs.threshold_label == "q_exact") & res.runs.n.isin(e5b_budgets(res))]
    rates = p2.groupby(["instance_key", "family", "n_rules", "sketch", "n"]).problem2_success.mean().reset_index()
    th = res.theory[(res.theory.ensemble == "e1") & (res.theory.t_label == "q_exact")]
    th = th[["instance_key", "sketch", "unique_maximizer", "max_preserved", "eq6_exponent"]].copy()
    th["rate_per_query"] = [e.get(E5_M_BOUND) if isinstance(e, dict) else None for e in th.eq6_exponent]
    table = rates.merge(th.drop(columns="eq6_exponent"), on=["instance_key", "sketch"], how="left")
    table["exponent"] = table.n * table.rate_per_query.astype(float)
    return table


def fig_e5(res_a: Results, a: pd.DataFrame, res_b: Results, b: pd.DataFrame, path: Path) -> dict:
    """(a) Problem II failure vs n on the E5a instances (from res_a); (b) Problem II success vs the
    predicted exponent on the E1 instances (from res_b)."""
    J = res_a.params.J_SINGLE
    floor = 1 / (2 * J)
    roles = [(f"e5_{fam}", s["descriptor"]["key"]) for fam, s in res_a.selection.get("e5", {}).items() if s]
    roles = [(role, key) for role, key in roles if not a.empty and (a.role == role).any()]
    budgets_a = res_a.params.e5_budgets or res_a.budgets
    budgets_b = e5b_budgets(res_b)
    ncol = max(len(roles), len(budgets_b), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(FULL_WIDTH, 4.1), squeeze=False)
    plotted = (set(a.sketch) if not a.empty else set()) | set(b[b.exponent > 0].sketch)
    methods = methods_present(plotted)
    notes = {}
    ymax = max(2.0, np.nanmax(a.bound_eq6.to_numpy(float)) * 2) if not a.empty and a.bound_eq6.notna().any() else 2.0
    for ax, (role, key) in zip(axes[0], roles):
        for m in methods:
            d = a[(a.role == role) & (a.sketch == m)].sort_values("n")
            if d.empty:
                continue
            zero = d.rate == 0
            ax.plot(d.n[~zero], d.rate[~zero], color=COLORS[m], marker=MARKERS[m], markeredgewidth=0)
            ax.plot(d.n[zero], np.full(zero.sum(), floor), linestyle="none", color=COLORS[m], marker=MARKERS[m],
                    markerfacecolor="white", markeredgewidth=0.8)
            if d.bound_eq6.notna().any():
                ax.plot(d.n, d.bound_eq6, color=COLORS[m], linewidth=0.8, linestyle=(0, (4, 2)))
                n7 = d.n_eq7.iloc[0]
                notes[f"{key}/{m}"] = {"n_eq7": n7}
                if n7 <= budgets_a[-1]:
                    ax.axvline(n7, color=COLORS[m], linewidth=0.6, linestyle=(0, (1, 2)))
        ax.axhline(1.0, color=TEXT_2, linewidth=0.5)
        budget_axis(ax, budgets_a)
        ax.set_yscale("log")
        ax.set_ylim(floor / 2, ymax)
        ax.set_title(f"(a) {key}")
        ax.set_xlabel("queries n")
    if roles:
        axes[0, 0].set_ylabel("P(Problem II fails)")
    else:
        print("[plot] no E5a sweep in this directory: panel (a) left empty", flush=True)
    for ax in axes[0, len(roles):]:
        ax.set_visible(False)

    for ax, n in zip(axes[1], budgets_b):
        d_n = b[(b.n == n) & (b.exponent > 0)]
        for m in methods:
            d = d_n[d_n.sketch == m]
            if d.empty:
                continue
            ax.plot(d.exponent, d.problem2_success, linestyle="none", color=COLORS[m], marker=MARKERS[m],
                    markersize=2.4, alpha=0.45, markeredgewidth=0)
            x, y = d.exponent.to_numpy(float), d.problem2_success.to_numpy(float)
            # Mean success in log-spaced bins of the exponent (bins with at least 5 instances).
            edges = np.logspace(np.log10(x.min()), np.log10(x.max()), 8)
            which = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
            binned = [(np.median(x[which == i]), y[which == i].mean()) for i in range(len(edges) - 1)
                      if (which == i).sum() >= 5]
            if binned:
                bx, by = zip(*binned)
                ax.plot(bx, by, color=COLORS[m], marker=MARKERS[m], markersize=3.6, markeredgecolor="white",
                        markeredgewidth=0.6)
            rho = spearmanr(x, y).statistic if len(x) > 2 else math.nan
            notes[f"n={n}/{m}"] = {"instances_with_exponent": int(len(x)), "spearman_rho": float(rho),
                                   "instances_without_exponent": int(((b.n == n) & (b.sketch == m)
                                                                      & ~(b.exponent > 0)).sum())}
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("predicted exponent  n Θ²/(2ν² + ⅔M̄Θ)")
        ax.set_title(f"(b) E1 instances, n = {budget_label(n)}")
    axes[1, 0].set_ylabel("P(Problem II succeeds)")
    for ax in axes[1, len(budgets_b):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in methods]
    handles += [plt.Line2D([], [], color=TEXT_2, linewidth=0.8, linestyle=(0, (4, 2)), label="bound, Eq. (6)"),
                plt.Line2D([], [], color=TEXT_2, marker="o", linestyle="none", markerfacecolor="white",
                           label=f"no failure in {J} runs")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    fig.tight_layout(h_pad=1.2)
    fig.savefig(path)
    plt.close(fig)
    return notes


# =============================================================================
# E4: cost table (every method at n = N_MAX, |R| pooled)
# =============================================================================

MCCO_STAGES = ["sampling", "sketching", "decoding", "candidates"]


def e4_tables(res: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    e1 = res.runs[(res.runs.experiment == "e1") & (res.runs.record == "run")].copy()
    for stage in MCCO_STAGES + ["total"]:
        e1[stage] = [t.get(stage) if isinstance(t, dict) else None for t in e1.time]
    mcco = e1.m != "annealing"
    e1.loc[mcco, "total"] = e1.loc[mcco, MCCO_STAGES].astype(float).sum(axis=1)
    cols = MCCO_STAGES + ["total"]
    e1[cols] = e1[cols].astype(float)
    by_budget = (e1.groupby(["family", "n_rules", "m", "n"])[cols].median().reset_index()
                 .rename(columns={"m": "method"}))
    n = res.params.N_MAX
    at_n = e1[e1.n == n]
    table = at_n.groupby(["m", "family"])[cols].median().reset_index().rename(columns={"m": "method"})
    table["runs"] = at_n.groupby(["m", "family"]).size().to_numpy()
    table.insert(2, "n", n)
    table["method"] = pd.Categorical(table.method, METHODS, ordered=True)
    return table.sort_values(["method", "family"]).reset_index(drop=True), by_budget


def write_e4_tex(table: pd.DataFrame, path: Path) -> None:
    n = int(table.n.iloc[0])
    lines = [r"\begin{tabular}{ll" + "r" * (len(MCCO_STAGES) + 1) + "}", r"\hline",
             "Method & Family & " + " & ".join(MCCO_STAGES + ["total"]) + r" \\",
             r" & & \multicolumn{" + str(len(MCCO_STAGES) + 1) + rf"}}{{c}}{{median wall-clock (s) at $n = {n}$}} \\",
             r"\hline"]
    for _, r in table.iterrows():
        cells = [("--" if pd.isna(r[c]) else f"{r[c]:.3g}") for c in MCCO_STAGES + ["total"]]
        lines.append(f"{LABELS[r.method]} & {r.family} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# Main
# =============================================================================

def make_figures(results_dir: Path, fig_dir: Path, e3_e5_dir: Path | None = None, overwrite: bool = False,
                 workers: int = 1, e2_dir: Path | None = None, theory_dir: Path | None = None) -> None:
    """E1, E4 and E5b from ``results_dir``; E2 from ``e2_dir`` and E3, E5a from ``e3_e5_dir``
    (both default to ``results_dir``). With ``theory_dir``, its theory.jsonl replaces the theory
    records of ``results_dir`` and ``e3_e5_dir`` (E1, E3, E5); E2 keeps those of ``e2_dir``."""
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    existing = [p.name for p in fig_dir.glob("*") if p.stem in FIGURES or p.name == "summary.json"] \
        if fig_dir.exists() else []
    if existing and not overwrite:
        raise SystemExit(f"{fig_dir} already holds figures ({', '.join(sorted(existing))}); "
                         "choose another --figures directory or pass --overwrite.")
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    res = Results(results_dir)
    res_new = Results(Path(e3_e5_dir)) if e3_e5_dir else res
    res_e2 = Results(Path(e2_dir)) if e2_dir else res
    add_sample_best(res, {"e1"}, fig_dir / "sample_best.csv", workers)
    add_sample_best(res_new, {"sweep"}, fig_dir / "sample_best_e3.csv", workers, roles={"e3"})
    if theory_dir:
        if res_e2 is res:
            res_e2 = copy.copy(res)          # E2 keeps its own theory (the E2 threshold grid)
        th = pd.DataFrame(RecordStore(Path(theory_dir)).load("theory.jsonl"))
        if th.empty:
            raise SystemExit(f"no theory records in {theory_dir}")
        res.theory = th
        res_new.theory = th

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    e1 = e1_table(res, rng)
    e1.to_csv(fig_dir / "e1_success.csv", index=False)
    fig_e1_grid(res, e1, "success", fig_dir / "e1_success.pdf")
    fig_e1_grid(res, e1, "distance", fig_dir / "e1_distance.pdf")

    e2 = e2_table(res_e2)
    e2.to_csv(fig_dir / "e2_threshold.csv", index=False)
    fig_e2(res_e2, e2, fig_dir / "e2_threshold.pdf")

    e3 = e3_table(res_new)
    e3.to_csv(fig_dir / "e3_mismatch.csv", index=False)
    fig_e3(res_new, e3, fig_dir / "e3_mismatch.pdf")

    a, b = e5a_table(res_new), e5b_table(res)
    a.to_csv(fig_dir / "e5a_problem2.csv", index=False)
    b.to_csv(fig_dir / "e5b_problem2.csv", index=False)
    e5_notes = fig_e5(res_new, a, res, b, fig_dir / "e5_theory.pdf")

    e4, e4_by_budget = e4_tables(res)
    e4.to_csv(fig_dir / "e4_cost.csv", index=False)
    e4_by_budget.to_csv(fig_dir / "e4_cost_by_budget.csv", index=False)
    write_e4_tex(e4, fig_dir / "e4_cost.tex")

    summary = {
        "results_dir": str(results_dir.resolve()),
        "e2_dir": str(res_e2.dir.resolve()),
        "e3_e5_dir": str(res_new.dir.resolve()),
        "theory_dir": str(Path(theory_dir).resolve()) if theory_dir else None,
        "sources": {"results_dir": ["E1", "E4", "E5b"], "e2_dir": ["E2"], "e3_e5_dir": ["E3", "E5a"]},
        "e2_instances": {f: {"instance": c["instance_key"], "budget": c["budget"]}
                         for f, c in (res_e2.e2_choice or {}).items()},
        "estimate": {"E1, E3": "best of sampled strings and MP candidates", "E2": "MP candidates only",
                     "E5": "Problem II"},
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "levels": "instances, then runs",
                      "interval": "95% percentile"},
        "e5_m_bound": E5_M_BOUND, "e5b_budgets": e5b_budgets(res),
        "e5a_instances": {f: (s["descriptor"]["key"] if s else None)
                          for f, s in res_new.selection.get("e5", {}).items()},
        "e3_instance": res_new.selection["e3"]["descriptor"]["key"],
        "record_counts": {"runs": int(len(res.runs)), "runs_e3_e5": int(len(res_new.runs)),
                          "theory": int(len(res.theory)), "instances": int(len(res.instances))},
        "e5": e5_notes,
    }
    (fig_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"figures and tables written to {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="results directory (E1, E2, E4, E5b; also E3 and E5a unless --e3-e5)")
    parser.add_argument("--e3-e5", help="results directory holding the E3 and E5a sweeps (new run)")
    parser.add_argument("--e2", help="results directory holding E2 (default: the results directory)")
    parser.add_argument("--theory", help="directory whose theory.jsonl replaces the theory records for "
                                         "E1, E3 and E5 (E2 keeps those of --e2)")
    parser.add_argument("--figures", help="output directory (default: <E3/E5 dir or results>/figures)")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing figures in --figures")
    parser.add_argument("--workers", type=int, default=1, help="processes for the post-hoc best sampled string")
    args = parser.parse_args()
    results_dir = Path(args.results)
    new_dir = Path(args.e3_e5) if args.e3_e5 else None
    fig_dir = Path(args.figures) if args.figures else (new_dir or results_dir) / "figures"
    make_figures(results_dir, fig_dir, new_dir, overwrite=args.overwrite, workers=args.workers,
                 e2_dir=Path(args.e2) if args.e2 else None, theory_dir=Path(args.theory) if args.theory else None)


if __name__ == "__main__":
    main()
