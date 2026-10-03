"""Figures and tables of the MCCO revision, from the records of one results directory.

    python plot.py                                 # every figure, from the latest results/resultsN
    python plot.py results/results3 --only e2      # E2 only, from an earlier results directory
    python plot.py --only e2 e5 --from-csv         # redraw E2 and E5 from the tables in figures/

Every results directory holds the records of every stage (mcco_sim/inherit.py), so one directory is
enough; its sources.json says which run each stage comes from.

--only E... makes only those experiments (e1 e2 e3 e4 e5 s1). --from-csv draws from the tables
already in the figures directory (written by an earlier run) instead of recomputing them from the
records; the run records are then never loaded, so it takes seconds. Use it after a change of the
drawing code only: a change of the data or of a table computation needs a run without --from-csv.

Outputs (OUTPUTS below), tables in data/ and figures in plot/:
    main/           e1_distance, e2_threshold, e4_cost (.csv and LaTeX table)
    supplementary/  e1_success, e3_mismatch, e5_theory (e5a/e5b tables), s1_decoders (basis vs
                    matching pursuit, if the results directory holds the S1 records)
Each data/ also holds summary.json (sources, settings and seeds of the aggregates).

Result directories are only read. Figures go to --figures (default: figures/), replacing the
previous version of the figures made; summary.json records the results directory each one is from.

MCCO estimate (figure_edits.md, edit 2): best of the sampled strings and the matching-pursuit
candidates. Records written before that change hold the MP-only outcome; for them the best sampled
string is recomputed from the recorded seeds (mcco_sim/posthoc.py, cached in main/data/sample_best.csv).
E1 and E3 use this combined outcome; E2 uses the MP-only outcome (effect of the threshold on the
decoding); E5 uses Problem II; S1 compares the decoders alone (best of the 5 candidates).

The plotting settings below only affect the figures, not the simulation.
"""

from __future__ import annotations

import os

# One BLAS thread per process (the process pool provides the parallelism), set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import logging
import math
import multiprocessing
from functools import cached_property
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
from mcco_sim.records import RecordStore, latest_results  # noqa: E402

# --- Plotting settings --------------------------------------------------------------------------
BOOTSTRAP_SAMPLES = 2000        # two-level bootstrap replicates (95% intervals)
BOOTSTRAP_SEED = 20260927
E5B_BUDGETS = (800, 6400, 51200)  # budgets of the E5b panels (nearest grid budgets if absent)
E5_M_BOUND = "valid"            # M used for the Eq. (6) bound: "valid" or "plan_2m_sup"
E5A_YMAX = 10.0                 # top of the E5a failure axis (the bound is cut above it)
E5B_SHOWN = 60                  # E5b instances drawn per sketch (uniformly, fixed seed); bins and fits use all
E5B_SHOWN_SEED = 20261003
E5B_POINT_ALPHA = 0.25          # opacity of the per-instance E5b points
E5B_XLABEL = r"$n\,\Theta^2 / (2\nu^2 + \frac{2}{3}\bar{M}\Theta)$"   # predicted exponent of Eq. (6)
E4_BUDGET = 6400                # budget of the E4 cost table (nearest grid budget)
S1_BOOTSTRAP_SEED = 20260928    # S1 intervals (bootstrap as E1, its own seed)

# --- Style: one shade of blue per method, told apart by marker and line style. Computer Modern
# text and math, as in the paper. --------------------------------------------------------------------
METHODS = ["quadruplet", "quintuplet", "random", "annealing"]
LABELS = {"quadruplet": "MCCO Quadruplets", "quintuplet": "MCCO Quintuplets", "random": "MCCO Random",
          "annealing": "Digital Annealing"}
COLORS = {"quadruplet": "#2171b5", "quintuplet": "#6baed6", "random": "#9ecae1", "annealing": "#08306b"}
MARKERS = {"quadruplet": "o", "quintuplet": "s", "random": "^", "annealing": "D"}
LINESTYLES = {"quadruplet": "-", "quintuplet": (0, (4, 1.5)), "random": (0, (1, 1.2)),
              "annealing": (0, (5, 1.5, 1, 1.5))}
FAMILY_NAMES = {"L": "Local Rules", "W": "Non-local Rules"}   # rule families, as named in the figures
E4_FAMILY_NAMES = {"L": "Local", "W": "Non-Local"}             # and in the E4 table
TEXT, TEXT_2, RULE = "#0b0b0b", "#52514e", "#dcdbd7"
FULL_WIDTH, HALF_WIDTH = 7.2, 3.5      # inches: Scientific Reports double and single column
# S1 compares decoders, not methods: shades of green. label, color, marker, line style.
DECODERS = {"mp": ("Matching Pursuit", "#74c476", "o", (0, (4, 1.5))),
            "bp": ("Basis Pursuit", "#00441b", "s", "-")}

# Files written per experiment (--only), relative to the figures directory: main/ and supplementary/,
# each with data/ (tables, read back by --from-csv) and plot/ (PDF figures, LaTeX table).
OUTPUTS = {"e1": ["main/data/e1_distance.csv", "main/plot/e1_distance.pdf",
                  "supplementary/data/e1_success.csv", "supplementary/plot/e1_success.pdf"],
           "e2": ["main/data/e2_threshold.csv", "main/plot/e2_threshold.pdf"],
           "e3": ["supplementary/data/e3_mismatch.csv", "supplementary/plot/e3_mismatch.pdf"],
           "e4": ["main/data/e4_cost.csv", "main/data/e4_cost_by_budget.csv", "main/plot/e4_cost.tex"],
           "e5": ["supplementary/data/e5a_problem2.csv", "supplementary/data/e5b_problem2.csv",
                  "supplementary/plot/e5_theory.pdf"],
           "s1": ["supplementary/data/s1_success.csv", "supplementary/data/s1_bp_diagnostics.csv",
                  "supplementary/plot/s1_decoders.pdf"]}
PATH = {Path(p).name: p for files in OUTPUTS.values() for p in files}
SECTIONS = ["main", "supplementary"]
# The E1 table, split between the distance figure (main) and the success figure (supplementary).
E1_KEYS = ["family", "n_rules", "method", "n", "instances", "runs_per_instance"]
E1_COLUMNS = {"success": E1_KEYS + ["success", "success_ci_low", "success_ci_high", "success_mp"],
              "distance": E1_KEYS + ["distance_median", "distance_q25", "distance_q75",
                                     "rank_median", "rank_q25", "rank_q75", "distance_mp_median"]}
RUN_FIELDS = ["record", "method", "experiment", "role", "family", "n_rules", "instance_id", "instance_key",
              "sample_id", "run_id", "sketch", "threshold_mode", "threshold_label", "t", "n", "n_kept", "queries",
              "success", "functional_distance", "percentile_rank", "f_x_hat",
              "success_mp", "functional_distance_mp", "percentile_rank_mp", "f_x_hat_mp",
              "sample_best_f", "problem2_success", "time"]


def set_style() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)   # the bundled cmr10 has an old timestamp
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
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
    handles = [method_handle(m) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(methods),
               handlelength=1.8, columnspacing=1.2)


def band(ax, x, y, low, high, m: str, **kwargs) -> None:
    ax.fill_between(x, low, high, color=COLORS[m], alpha=0.15, linewidth=0)
    ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m], markeredgewidth=0, **kwargs)


def method_handle(m: str) -> plt.Line2D:
    """Legend entry of a method: its color, marker and line style."""
    return plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m], label=LABELS[m])


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
    """One results directory, read through RecordStore (nothing is ever written there). The record
    files are loaded on first use, so drawing from saved tables (--from-csv) never reads the runs."""

    def __init__(self, results_dir: Path) -> None:
        self.dir = results_dir
        self.store = RecordStore(results_dir)
        self.params = stored_params(results_dir)
        self.budgets = self.params.budgets
        self.selection = self.store.read_json("selection.json")
        path = results_dir / "e2_budget_choice.json"
        self.e2_choice = json.loads(path.read_text()) if path.exists() else None

    @cached_property
    def runs(self) -> pd.DataFrame:
        return normalize_runs(pd.DataFrame(self.store.load("runs.jsonl", fields=RUN_FIELDS)))

    @cached_property
    def theory(self) -> pd.DataFrame:
        return pd.DataFrame(self.store.load("theory.jsonl"))

    @cached_property
    def instances(self) -> pd.DataFrame:
        return pd.DataFrame(self.store.load("instances.jsonl"))

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
                ax.set_title(f"$|R| = {R}$")
            if j == 0:
                ax.set_ylabel(f"{FAMILY_NAMES.get(family, family)}\n" + ("Success Rate" if value == "success"
                                                                       else r"Distance to Optimum ($\sigma_f$)"))
    fig.supxlabel("Queries $n$", fontsize=7, y=0.0)
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
    # Success only; the distance to the optimum stays in e2_threshold.csv.
    fig, axes = plt.subplots(1, len(items), figsize=(FULL_WIDTH * 0.62, 1.9), sharey=True, squeeze=False)
    q = res.params.Q
    for col, (family, sel) in enumerate(items):
        key = sel["descriptor"]["key"]
        ax = axes[0, col]
        for m in methods:
            d = table[(table.instance_key == key) & (table.sketch == m) & table.percentile.notna()]
            d = d.sort_values("percentile")
            band(ax, d.percentile, d.rate, d.ci_low, d.ci_high, m)
        ax.axvline(q, color=TEXT_2, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-2, 102)
        ax.set_title(FAMILY_NAMES.get(family, family))     # instance and budget go in the caption
    axes[0, 0].set_ylabel("Success Probability")
    fig.supxlabel("Threshold Percentile (%)", fontsize=7, y=-0.06)
    handles = [method_handle(m) for m in methods]
    handles.append(plt.Line2D([], [], color=TEXT_2, linewidth=0.6, linestyle=(0, (3, 2)), label=f"Q = {q:g} %"))
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
    names = {"quadruplet": " (Matched)", "quintuplet": " (Mismatched)"}
    methods = methods_present(table.sketch)
    for m in methods:
        d = table[table.sketch == m].sort_values("n")
        band(axes[0, 0], d.n, d.rate, d.ci_low, d.ci_high, m, label=LABELS[m] + names.get(m, ""))
        band(axes[0, 1], d.n, d.distance_median, d.distance_q25, d.distance_q75, m)
    for ax in axes[0]:
        budget_axis(ax, res.budgets)
    fig.supxlabel("Queries $n$", fontsize=7, y=-0.08)
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("Success Probability")
    axes[0, 1].set_ylabel(r"Distance to Optimum ($\sigma_f$)")
    axes[0, 0].set_title("(a) Success")
    axes[0, 1].set_title("(b) Distance to Optimum")
    # The instance and its number of maximizers go in the caption.
    fig.suptitle(f"{int(inst.n_rules)} rules of length 4", fontsize=7, y=1.16)
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


def fig_e5(res: Results, a: pd.DataFrame, b: pd.DataFrame, path: Path) -> dict:
    """(a) Problem II failure vs n on the E5a instances; (b) Problem II success vs the predicted
    exponent on the E1 instances."""
    J = res.params.J_SINGLE
    floor = 1 / (2 * J)
    roles = [(f"e5_{fam}", s["descriptor"]["key"]) for fam, s in res.selection.get("e5", {}).items() if s]
    roles = [(role, key) for role, key in roles if not a.empty and (a.role == role).any()]
    budgets_a = res.params.e5_budgets or res.budgets
    budgets_b = e5b_budgets(res)
    ncol = max(len(roles), len(budgets_b), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(FULL_WIDTH, 4.1), squeeze=False)
    plotted = (set(a.sketch) if not a.empty else set()) | set(b[b.exponent > 0].sketch)
    methods = methods_present(plotted)
    notes = {}
    for ax, (role, key) in zip(axes[0], roles):
        for m in methods:
            d = a[(a.role == role) & (a.sketch == m)].sort_values("n")
            if d.empty:
                continue
            zero = d.rate == 0
            ax.plot(d.n[~zero], d.rate[~zero], color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m],
                    markeredgewidth=0)
            ax.plot(d.n[zero], np.full(zero.sum(), floor), linestyle="none", color=COLORS[m], marker=MARKERS[m],
                    markerfacecolor="white", markeredgewidth=0.8)
            if d.bound_eq6.notna().any():
                ax.plot(d.n, d.bound_eq6, color=COLORS[m], linewidth=0.8, linestyle=(0, (1, 1)))
                n7 = d.n_eq7.iloc[0]
                notes[f"{key}/{m}"] = {"n_eq7": n7}
                if n7 <= budgets_a[-1]:
                    ax.axvline(n7, color=COLORS[m], linewidth=0.6, linestyle="-")
                    # Its value, along the line from the x axis (in the strip below the markers).
                    ax.text(n7 / 1.03, 0.015, f"{n7 / 1000:.0f}k", transform=ax.get_xaxis_transform(),
                            rotation=90, ha="right", va="bottom", fontsize=6, color=COLORS[m])
        ax.axhline(1.0, color=TEXT_2, linewidth=0.5)
        budget_axis(ax, budgets_a)
        ax.set_yscale("log")
        ax.set_ylim(floor / 10, E5A_YMAX)              # room for the Eq. (7) values below the markers
        family = role[len("e5_"):]
        ax.set_title(f"(a) {FAMILY_NAMES.get(family, family)}")   # instance in the caption
        ax.set_xlabel(" ")                                         # room for the row label
    if roles:
        axes[0, 0].set_ylabel("P(Problem II Fails)")
    else:
        print("[plot] no E5a sweep in this directory: panel (a) left empty", flush=True)
    for ax in axes[0, len(roles):]:
        ax.set_visible(False)

    # Instances whose points are drawn: a uniform sample per sketch, the same in every panel.
    rng = np.random.default_rng(E5B_SHOWN_SEED)
    shown = {}
    for m in methods:
        keys = np.sort(b[(b.sketch == m) & (b.exponent > 0)].instance_key.unique())
        shown[m] = set(rng.choice(keys, min(E5B_SHOWN, len(keys)), replace=False)) if len(keys) else set()
    for ax, n in zip(axes[1], budgets_b):
        d_n = b[(b.n == n) & (b.exponent > 0)]
        for m in methods:
            d = d_n[d_n.sketch == m]
            if d.empty:
                continue
            s = d[d.instance_key.isin(shown[m])]
            ax.plot(s.exponent, s.problem2_success, linestyle="none", color=COLORS[m], marker=MARKERS[m],
                    markersize=2.4, alpha=E5B_POINT_ALPHA, markeredgewidth=0)
            x, y = d.exponent.to_numpy(float), d.problem2_success.to_numpy(float)
            # Mean success in log-spaced bins of the exponent (bins with at least 5 instances, all
            # instances), and a least-squares line through these means in log10(exponent).
            edges = np.logspace(np.log10(x.min()), np.log10(x.max()), 8)
            which = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
            binned = [(np.median(x[which == i]), y[which == i].mean()) for i in range(len(edges) - 1)
                      if (which == i).sum() >= 5]
            fit = None
            if binned:
                bx, by = map(np.array, zip(*binned))
                ax.plot(bx, by, linestyle="none", color=COLORS[m], marker=MARKERS[m], markersize=3.6,
                        markeredgecolor="white", markeredgewidth=0.6)
                if len(bx) >= 2:
                    slope, intercept = np.polyfit(np.log10(bx), by, 1)
                    xs = np.logspace(np.log10(bx.min()), np.log10(bx.max()), 50)
                    ax.plot(xs, intercept + slope * np.log10(xs), color=COLORS[m], linestyle=LINESTYLES[m],
                            linewidth=1.0)
                    fit = {"slope_per_decade": float(slope), "intercept": float(intercept), "bins": int(len(bx))}
            rho = spearmanr(x, y).statistic if len(x) > 2 else math.nan
            notes[f"n={n}/{m}"] = {"instances_with_exponent": int(len(x)), "spearman_rho": float(rho),
                                   "instances_without_exponent": int(((b.n == n) & (b.sketch == m)
                                                                      & ~(b.exponent > 0)).sum()),
                                   "instances_shown": int(len(s)), "binned_mean_fit": fit}
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(" ")                                         # room for the row label
        ax.set_title(f"(b) $n$ = {budget_label(n)}")
    axes[1, 0].set_ylabel("P(Problem II Succeeds)")
    for ax in axes[1, len(budgets_b):]:
        ax.set_visible(False)
    handles = [method_handle(m) for m in methods]
    handles += [plt.Line2D([], [], color=TEXT_2, linewidth=0.8, linestyle=(0, (1, 1)), label="Bound, Eq. (6)"),
                plt.Line2D([], [], color=TEXT_2, linewidth=0.6, label="Sample Size, Eq. (7)"),
                plt.Line2D([], [], color=TEXT_2, marker="o", linestyle="none", markerfacecolor="white",
                           label=f"No Failure in {J} Runs"),
                plt.Line2D([], [], color=TEXT_2, marker="o", linestyle="none", markersize=3.6,
                           markeredgecolor="white", markeredgewidth=0.6, label="Binned Mean"),
                plt.Line2D([], [], color=TEXT_2, linewidth=1.0, label="Linear Fit")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles), 4))
    fig.tight_layout(h_pad=1.2)
    # One x label per row, centred under its visible panels.
    for row, label in ((0, "Queries $n$"), (1, E5B_XLABEL)):
        shown_axes = [ax for ax in axes[row] if ax.get_visible()]
        if shown_axes:
            left, right = shown_axes[0].get_position().x0, shown_axes[-1].get_position().x1
            fig.text((left + right) / 2, shown_axes[0].get_position().y0 - 0.075, label,
                     ha="center", va="center", fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return notes


# =============================================================================
# E4: cost table (every method at n = E4_BUDGET, |R| pooled)
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
    grid = np.array(res.budgets)
    n = int(grid[np.argmin(np.abs(np.log(grid) - math.log(E4_BUDGET)))])   # nearest grid budget
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
        lines.append(f"{LABELS[r.method]} & {E4_FAMILY_NAMES.get(r.family, r.family)} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# S1: basis pursuit vs matching pursuit
# =============================================================================

S1_FIELDS = ["experiment", "record", "instance_key", "family", "n_rules", "sample_id", "sketch", "n", "success_mp",
             "eta", "bp_residual_over_eta", "bp_positive", "bp_l1", "time", "unit"]


def s1_tables(res: Results, bp: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    bp = bp.copy()
    bp["decode_seconds"] = [t.get("decoding") if isinstance(t, dict) else None for t in bp.time]
    e1 = res.runs[(res.runs.experiment == "e1") & (res.runs.record == "run") & (res.runs.method == "mcco")].copy()
    e1["decode_seconds"] = [t.get("decoding") if isinstance(t, dict) else None for t in e1.time]
    keys = ["instance_key", "sample_id", "n", "sketch"]
    mp = e1[keys + ["family", "success_mp", "decode_seconds"]].rename(columns={"success_mp": "success"})
    paired = bp[keys + ["success_mp", "decode_seconds"]].rename(columns={"success_mp": "success"}).merge(
        mp, on=keys, suffixes=("_bp", "_mp"))
    rows = []
    for (family, sketch), g in paired.groupby(["family", "sketch"]):
        budgets = sorted(g.n.unique())
        for decoder in ("mp", "bp"):
            cube, instances = outcome_cube(g.assign(rep=g.sample_id, success=g[f"success_{decoder}"].astype(float)),
                                           "success", budgets)
            reps = two_level_bootstrap(cube, BOOTSTRAP_SAMPLES, rng)
            low, high = percentile_interval(reps)
            for k, n in enumerate(budgets):
                at = g[g.n == n]
                rows.append({"family": family, "sketch": sketch, "decoder": decoder, "n": n,
                             "instances": len(instances), "success": cube[:, :, k].mean(),
                             "ci_low": low[k], "ci_high": high[k],
                             "decode_seconds_median": float(pd.to_numeric(at[f"decode_seconds_{decoder}"]).median())})
    diagnostics = bp.groupby(["sketch", "n"]).agg(
        eta_median=("eta", "median"), residual_over_eta_median=("bp_residual_over_eta", "median"),
        positive_entries_median=("bp_positive", "median"), decodes=("eta", "size")).reset_index()
    return pd.DataFrame(rows), diagnostics


def fig_s1(table: pd.DataFrame, families: list[str], path: Path) -> None:
    sketches = [m for m in METHODS if m in set(table.sketch)]
    fig, axes = plt.subplots(len(families), len(sketches), figsize=(FULL_WIDTH * 0.8, 1.6 * len(families) + 0.3),
                             sharex=True, sharey=True, squeeze=False)
    for i, family in enumerate(families):
        for j, m in enumerate(sketches):
            ax = axes[i, j]
            for decoder, (label, color, marker, line) in DECODERS.items():
                d = table[(table.family == family) & (table.sketch == m) & (table.decoder == decoder)].sort_values("n")
                if d.empty:
                    continue
                ax.fill_between(d.n, d.ci_low, d.ci_high, color=color, alpha=0.15, linewidth=0)
                ax.plot(d.n, d.success, color=color, marker=marker, linestyle=line, markeredgewidth=0)
                budget_axis(ax, sorted(d.n.unique()))
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(LABELS[m])                   # instances per family: in the caption
            if j == 0:
                ax.set_ylabel(f"{FAMILY_NAMES.get(family, family)}\nSuccess Rate")
    fig.supxlabel("Queries $n$", fontsize=7, y=0.0)
    handles = [plt.Line2D([], [], color=c, marker=mk, linestyle=line, label=f"{label} (5 Candidates)")
               for label, c, mk, line in DECODERS.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def make_figures(results_dir: Path, fig_dir: Path, workers: int = 1,
                 only: list[str] | None = None, from_csv: bool = False) -> None:
    """Figures and tables of one (complete) results directory, in ``fig_dir``/main and
    ``fig_dir``/supplementary (OUTPUTS). ``only``: experiments to make (keys of OUTPUTS, default
    all). ``from_csv``: draw from the tables already in ``fig_dir`` instead of recomputing them from
    the records (only the figures, the E4 .tex and the E5 notes of the summaries are rewritten)."""
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    only = [e for e in OUTPUTS if e in only] if only else list(OUTPUTS)
    full = not from_csv and only == list(OUTPUTS)        # a full run rewrites the summaries from scratch
    if from_csv:
        missing = [name for e in only for name in OUTPUTS[e]
                   if name.endswith(".csv") and not name.endswith("s1_bp_diagnostics.csv") and not (fig_dir / name).exists()]
        if missing:
            raise SystemExit(f"--from-csv: {', '.join(missing)} not in {fig_dir}; make them once without --from-csv.")
    for section in SECTIONS:
        for kind in ("data", "plot"):
            (fig_dir / section / kind).mkdir(parents=True, exist_ok=True)
    set_style()

    def path(name: str) -> Path:
        return fig_dir / PATH[name]

    def table(name: str, compute) -> pd.DataFrame:
        """The table ``name``: read back from fig_dir (--from-csv), or computed and saved there."""
        if from_csv:
            return pd.read_csv(path(name))
        t = compute()
        t.to_csv(path(name), index=False)
        return t

    res = Results(results_dir)
    if not from_csv:
        if "e1" in only:
            add_sample_best(res, {"e1"}, fig_dir / "main" / "data" / "sample_best.csv", workers)
        if "e3" in only:
            add_sample_best(res, {"sweep"}, fig_dir / "supplementary" / "data" / "sample_best_e3.csv", workers,
                            roles={"e3"})

    if "e1" in only:
        e1 = None if from_csv else e1_table(res, np.random.default_rng(BOOTSTRAP_SEED))
        for value in ("success", "distance"):
            t = table(f"e1_{value}.csv", lambda: e1[[c for c in E1_COLUMNS[value] if c in e1]])
            fig_e1_grid(res, t, value, path(f"e1_{value}.pdf"))

    if "e2" in only:
        fig_e2(res, table("e2_threshold.csv", lambda: e2_table(res)), path("e2_threshold.pdf"))

    if "e3" in only:
        fig_e3(res, table("e3_mismatch.csv", lambda: e3_table(res)), path("e3_mismatch.pdf"))

    e5_notes = None
    if "e5" in only:
        a = table("e5a_problem2.csv", lambda: e5a_table(res))
        b = table("e5b_problem2.csv", lambda: e5b_table(res))
        e5_notes = fig_e5(res, a, b, path("e5_theory.pdf"))

    if "e4" in only:
        if from_csv:
            e4 = pd.read_csv(path("e4_cost.csv"))
        else:
            e4, e4_by_budget = e4_tables(res)
            e4.to_csv(path("e4_cost.csv"), index=False)
            e4_by_budget.to_csv(path("e4_cost_by_budget.csv"), index=False)
        write_e4_tex(e4, path("e4_cost.tex"))

    s1_runs = None
    if "s1" in only:
        if from_csv:
            s1 = pd.read_csv(path("s1_success.csv"))
        else:
            bp = pd.DataFrame(res.store.load("runs.jsonl", fields=S1_FIELDS, contains=['"experiment": "s1"']))
            s1_runs = int(len(bp))
            if bp.empty:
                print(f"[plot] no S1 records in {results_dir}: S1 skipped", flush=True)
                s1 = None
            else:
                s1, diagnostics = s1_tables(res, bp, np.random.default_rng(S1_BOOTSTRAP_SEED))
                s1.to_csv(path("s1_success.csv"), index=False)
                diagnostics.to_csv(path("s1_bp_diagnostics.csv"), index=False)
        if s1 is not None:
            fig_s1(s1, list(res.params.FAMILIES), path("s1_decoders.pdf"))

    # Summaries (main/data, supplementary/data): a partial run (--only, --from-csv) updates the
    # existing files. With --from-csv the tables, hence their sources, are those of the earlier run:
    # only the E5 notes are refreshed.
    fields = {section: {} for section in SECTIONS}
    if not from_csv:
        sources = res.store.path("sources.json")
        common = {"results_dir": str(results_dir.resolve()),
                  "stage_sources": json.loads(sources.read_text())["stages"] if sources.exists() else None}
        fields["main"].update(common)
        fields["supplementary"].update(common)
        fields["main"].update({
            "e2_instances": {f: {"instance": c["instance_key"], "budget": c["budget"]}
                             for f, c in (res.e2_choice or {}).items()},
            "estimate": {"E1": "best of sampled strings and MP candidates", "E2": "MP candidates only"},
            "intervals": {"E1 distance": "median and quartiles", "E2": "Wilson 95%"},
        })
        fields["supplementary"].update({
            "estimate": {"E1, E3": "best of sampled strings and MP candidates", "E5": "Problem II",
                         "S1": "best of the 5 candidates (decoder only)"},
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed_e1": BOOTSTRAP_SEED, "seed_s1": S1_BOOTSTRAP_SEED,
                          "levels": "instances, then runs", "interval": "95% percentile"},
            "e5_m_bound": E5_M_BOUND, "e5b_budgets": e5b_budgets(res),
            "e5a_instances": {f: (s["descriptor"]["key"] if s else None)
                              for f, s in res.selection.get("e5", {}).items()},
            "e3_instance": res.selection["e3"]["descriptor"]["key"],
        })
        if s1_runs is not None:
            fields["supplementary"]["s1_records"] = s1_runs
        if full:
            fields["main"]["record_counts"] = {"runs": int(len(res.runs)), "theory": int(len(res.theory)),
                                               "instances": int(len(res.instances))}
    if e5_notes is not None:
        fields["supplementary"]["e5"] = e5_notes
    for section in SECTIONS:
        target = fig_dir / section / "data" / "summary.json"
        summary = json.loads(target.read_text()) if target.exists() and not full else {}
        summary.update(fields[section])
        summary["last_invocation"] = {"only": only, "from_csv": from_csv}
        made = summary.setdefault("made_from", {})        # experiment -> results directory of its tables
        for e in only:
            if not from_csv and any(name.startswith(section) for name in OUTPUTS[e]):
                made[e] = results_dir.name
        target.write_text(json.dumps(summary, indent=2, default=float))
    print(f"{', '.join(only)} from {results_dir} written to {fig_dir}"
          + (" (from the saved tables)" if from_csv else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?", help="results directory (default: the latest results/resultsN)")
    parser.add_argument("--figures", default="figures", help="output directory (default: figures)")
    parser.add_argument("--workers", type=int, default=1, help="processes for the post-hoc best sampled string")
    parser.add_argument("--only", nargs="+", choices=list(OUTPUTS), metavar="E",
                        help=f"experiments to make, among {' '.join(OUTPUTS)} (default: all)")
    parser.add_argument("--from-csv", action="store_true",
                        help="draw from the tables already in --figures instead of recomputing them from the records")
    args = parser.parse_args()
    results_dir = Path(args.results) if args.results else latest_results()
    make_figures(results_dir, Path(args.figures), workers=args.workers,
                 only=args.only, from_csv=args.from_csv)


if __name__ == "__main__":
    main()
