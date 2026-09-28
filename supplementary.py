"""Supplementary figures S1-S3 (supplementary_numerics_plan.md), from the records of results
directories only (read-only, as plot.py).

    python supplementary.py results --e2 results_e2 --e3-e5 results_v2 --s1 results_s1 --figures figures_supp --workers 8

- S1 (needs --s1, the output of `run.py --stage s1`): basis pursuit vs matching pursuit, decoder-only
  outcome (best of the 5 candidates), on the same E1 samples (MP from the E1 records).
- S2 (post hoc, no troma): error budget of the structured sketches in the ±1 Walsh basis.
    S2a: coefficient error vs n (E5a instances, fixed exact threshold) and vs threshold (E2 instances);
    S2b: sketching terms L and ||f - F_K||_inf of every unique-maximizer E1 instance;
    S2c: table of the sufficient condition k eps + L < Delta/2, split by argmax F_hat_K = x* and by
         Problem II success (runs with t <= f(x2)).
- S3: per-instance E1 success at n = S_BUDGET (combined outcome for MCCO, as the main text).

The S2 per-instance and per-run tables are cached as CSV in the figures directory.
"""

from __future__ import annotations

import os

# One BLAS thread per process (the process pool provides the parallelism), set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import multiprocessing
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from mcco_sim.aggregates import percentile_interval, two_level_bootstrap
from mcco_sim.records import RecordStore
from mcco_sim.s2 import instance_task, runs_task
from plot import (COLORS, FULL_WIDTH, LABELS, MARKERS, METHODS, TEXT_2, Results, add_sample_best, budget_axis,
                  outcome_cube, set_style, threshold_percentile)

# --- Settings (figures only) ---------------------------------------------------------------------
S_BUDGET = 6400                 # budget of S2b (color) and S3: intermediate E1 success, as E2 (nearest grid budget)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260928
BLOCK = 5                       # samples per S2 task (~240 tasks: enough for many workers)
DECODERS = {"mp": ("Matching pursuit", "#2a78d6", "o"), "bp": ("Basis pursuit", "#eb6834", "s")}
FAMILY_MARKERS = {"L": "o", "W": "^"}
SEQUENTIAL = LinearSegmentedColormap.from_list(   # one hue, light -> dark (light end kept visible)
    "blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
STRUCTURED = ["quadruplet", "quintuplet"]
FIGURES_SUPP = ["s1_decoders", "s2a_coefficient_error", "s2b_sketching_error", "s3_success_distribution",
                "s3_success_distribution_all_R", "s3_success_vs_properties"]


def pool_map(function, tasks: list, workers: int) -> list:
    if workers > 1 and len(tasks) > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            return [row for chunk in pool.imap_unordered(function, tasks) for row in chunk]
    return [row for task in tasks for row in function(task)]


def cached(path: Path, compute) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    table = compute()
    table.to_csv(path, index=False)
    return table


# =============================================================================
# S3: per-instance success
# =============================================================================

def grid_budget(res: Results) -> int:
    """The E1 grid budget nearest to S_BUDGET (log scale)."""
    grid = np.array(res.budgets)
    return int(grid[np.argmin(np.abs(np.log(grid) - math.log(S_BUDGET)))])


def per_instance_success(res: Results, n: int) -> pd.DataFrame:
    """Per E1 instance and method: success rate over its runs at budget n (combined outcome for MCCO)."""
    e1 = res.runs[(res.runs.experiment == "e1") & (res.runs.record == "run") & (res.runs.n == n)]
    table = (e1.assign(success=e1.success.astype(float))
             .groupby(["instance_key", "family", "n_rules", "m"]).success.agg(["mean", "size"]).reset_index()
             .rename(columns={"m": "method", "mean": "success", "size": "runs"}))
    props = res.instances[["instance_key", "gap", "f_star", "wh_sparsity", "unique_maximizer"]]
    table = table.merge(props, on="instance_key")
    table["relative_gap"] = table.gap / table.f_star
    return table


def fig_s3_histograms(table: pd.DataFrame, families: list[str], R_values: list[int], n: int, path: Path) -> None:
    edges = np.arange(-0.05, 1.1, 0.1)
    fig, axes = plt.subplots(len(families), len(R_values), figsize=(FULL_WIDTH if len(R_values) > 1 else 4.8,
                                                                  1.5 * len(families) + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    methods = [m for m in METHODS if m in set(table.method)]
    for i, family in enumerate(families):
        for j, R in enumerate(R_values):
            ax = axes[i, j]
            for m in methods:
                values = table[(table.family == family) & (table.n_rules == R) & (table.method == m)].success
                if values.empty:
                    continue
                counts, _ = np.histogram(values, bins=edges)
                ax.stairs(counts / len(values), edges, color=COLORS[m], linewidth=1.1)
            ax.set_xlim(-0.08, 1.08)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(f"|R| = {R}")
            if j == 0:
                ax.set_ylabel(f"Family {family}\nfraction of instances")
            if i == len(families) - 1:
                ax.set_xlabel(f"per-instance success rate (n = {n})")
    counts = table[table.n_rules == R_values[0]].groupby("method").instance_key.nunique() // len(families)
    handles = [plt.Line2D([], [], color=COLORS[m], label=f"{LABELS[m]} ({counts.get(m, 0)} inst.)") for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    fig.savefig(path)
    plt.close(fig)


def fig_s3_properties(table: pd.DataFrame, n: int, path: Path) -> None:
    unique = table[table.unique_maximizer.astype(bool)]
    methods = [m for m in METHODS if m in set(unique.method)]
    fig, axes = plt.subplots(2, len(methods), figsize=(FULL_WIDTH, 3.4), sharey=True, squeeze=False)
    jitter = np.random.default_rng(0)                          # display only: rates take 0.1 steps
    for j, m in enumerate(methods):
        d = unique[unique.method == m]
        for row, (column, label) in enumerate((("relative_gap", "relative gap γ / f*"),
                                               ("wh_sparsity", "WH sparsity s"))):
            ax = axes[row, j]
            for family, marker in FAMILY_MARKERS.items():
                dd = d[d.family == family]
                ax.plot(dd[column], dd.success + jitter.uniform(-0.02, 0.02, len(dd)), linestyle="none",
                        marker=marker, markersize=2.4, color=COLORS[m], alpha=0.45, markeredgewidth=0)
            ax.set_xscale("log")
            ax.set_xlabel(label)
            if row == 0:
                ax.set_title(f"{LABELS[m]} ({d.instance_key.nunique()} inst.)")
    for row in range(2):
        axes[row, 0].set_ylabel(f"success rate (n = {n})")
    handles = [plt.Line2D([], [], color=TEXT_2, marker=mk, linestyle="none", label=f"Family {fam}")
               for fam, mk in FAMILY_MARKERS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# S2: error budget (Walsh basis)
# =============================================================================

def s2_instances(res: Results, fig_dir: Path, workers: int) -> pd.DataFrame:
    """S2b per instance and structured sketch (unique-maximizer E1 instances)."""
    def compute():
        records = [r for r in res.store.load("instances.jsonl") if r["ensemble"] == "e1" and r["unique_maximizer"]]
        return pd.DataFrame(pool_map(instance_task, [(res.params.N, r) for r in records], workers))
    return cached(fig_dir / "s2_instances.csv", compute)


def s2_groups(res: Results, runs: pd.DataFrame, source: str, n_max_of) -> list[tuple]:
    """S2 tasks: per instance, blocks of (sample_id, n, t, threshold_label, n_max) run groups."""
    records = {r["instance_key"]: r for r in res.store.load("instances.jsonl")}
    groups = runs[["instance_key", "sample_id", "n", "t", "threshold_label"]].drop_duplicates()
    tasks = []
    for key, g in groups.groupby("instance_key"):
        g = g.sort_values(["sample_id", "n", "t"])
        items = [(int(r.sample_id), int(r.n), float(r.t), r.threshold_label, n_max_of(int(r.n)))
                 for r in g.itertuples()]
        samples = sorted(g.sample_id.unique())
        for start in range(0, len(samples), BLOCK):
            block = set(samples[start:start + BLOCK])
            tasks.append((res.params, res.params.N, records[key], source,
                          [it for it in items if it[0] in block]))
    return tasks


def s2_runs(res_e5: Results, res_e2: Results, fig_dir: Path, workers: int) -> pd.DataFrame:
    """S2a/S2c per run group and sketch, with the stored Problem II success of the same run."""
    e5 = res_e5.runs[res_e5.runs.role.fillna("").str.startswith("e5_") & (res_e5.runs.record == "problem2")]
    e2 = res_e2.runs[(res_e2.runs.experiment == "e2") & (res_e2.runs.record == "run")]

    def compute():
        e5_n_max = max(res_e5.params.e5_budgets or res_e5.budgets)
        tasks = s2_groups(res_e5, e5, "e5a", lambda n: e5_n_max) + s2_groups(res_e2, e2, "e2", lambda n: n)
        return pd.DataFrame(pool_map(runs_task, tasks, workers))

    table = cached(fig_dir / "s2_runs.csv", compute)
    p2 = pd.concat([e5.assign(source="e5a"), e2.assign(source="e2")])[
        ["source", "instance_key", "sample_id", "n", "threshold_label", "sketch", "problem2_success"]]
    return table.merge(p2, on=["source", "instance_key", "sample_id", "n", "threshold_label", "sketch"], how="left")


def hoeffding(B: float, k: int, n, delta: float):
    """Hoeffding + union bound over the k coefficients: max error <= B sqrt(2 ln(2k/delta) / n) w.p. 1 - delta."""
    return B * np.sqrt(2 * np.log(2 * k / delta) / np.asarray(n, dtype=float))


def fig_s2a(runs: pd.DataFrame, delta: float, sources: dict, path: Path) -> dict:
    notes = {}
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.0), squeeze=False)
    for col, (key, ax) in enumerate(zip(sources["e5a"], axes[0])):
        d = runs[(runs.source == "e5a") & (runs.instance_key == key)]
        for m in STRUCTURED:
            dm = d[d.sketch == m]
            if dm.empty:
                continue
            q = dm.groupby("n").eps_samp.quantile([0.1, 0.5, 0.9]).unstack()
            ax.fill_between(q.index, q[0.1], q[0.9], color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(q.index, q[0.5], color=COLORS[m], marker=MARKERS[m], markeredgewidth=0)
            k, B, gap = int(dm.k.iloc[0]), float(dm.B.iloc[0]), float(dm.gap.iloc[0])
            ax.plot(q.index, hoeffding(B, k, q.index, delta), color=COLORS[m], linewidth=0.8, linestyle=(0, (4, 2)))
            ax.axhline(gap / (4 * k), color=COLORS[m], linewidth=0.7, linestyle=(0, (1, 2)))
            ax.axhline(float(dm.b_t.iloc[0]), color=COLORS[m], linewidth=0.8, linestyle=(0, (5, 2, 1, 2)))
            n_req = 32 * B ** 2 * k ** 2 * math.log(2 * k / delta) / gap ** 2
            notes[f"{key}/{m}"] = {"k": k, "delta_over_4k": gap / (4 * k), "b_t": float(dm.b_t.iloc[0]),
                                   "n_required_prop2": n_req}
            if n_req <= q.index.max():
                ax.axvline(n_req, color=COLORS[m], linewidth=0.6, linestyle=(0, (1, 2)))
        budget_axis(ax, sorted(d.n.unique()))
        ax.set_yscale("log")
        ax.set_xlabel("queries n")
        ax.set_title(f"(a) {key}, t = exact 85th percentile")
    for col, (key, ax) in enumerate(zip(sources["e2"], axes[1])):
        d = runs[(runs.source == "e2") & (runs.instance_key == key)].copy()
        d["percentile"] = d.threshold_label.map(threshold_percentile)
        d = d[d.percentile.notna()]
        for m in STRUCTURED:
            dm = d[d.sketch == m]
            if dm.empty:
                continue
            q = dm.groupby("percentile").eps_samp.quantile([0.1, 0.5, 0.9]).unstack()
            bias = dm.groupby("percentile").b_t.first()
            ax.fill_between(q.index, q[0.1], q[0.9], color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(q.index, q[0.5], color=COLORS[m], marker=MARKERS[m], markeredgewidth=0)
            ax.plot(bias.index, bias.values, color=COLORS[m], linewidth=0.8, linestyle=(0, (5, 2, 1, 2)))
            k, gap = int(dm.k.iloc[0]), float(dm.gap.iloc[0])
            ax.axhline(gap / (4 * k), color=COLORS[m], linewidth=0.7, linestyle=(0, (1, 2)))
        ax.set_yscale("log")
        ax.set_xlim(-2, 102)
        ax.set_xlabel("threshold percentile (% of the strings)")
        n = int(d.n.iloc[0]) if not d.empty else 0
        ax.set_title(f"(b) {key}, n = {n}")
    for row in range(2):
        axes[row, 0].set_ylabel("max coefficient error over K")
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in STRUCTURED]
    handles += [plt.Line2D([], [], color=TEXT_2, label="ε_samp (median, 10–90 %)"),
                plt.Line2D([], [], color=TEXT_2, linewidth=0.8, linestyle=(0, (4, 2)), label="Hoeffding + union bound"),
                plt.Line2D([], [], color=TEXT_2, linewidth=0.7, linestyle=(0, (1, 2)), label="Δ / (4k)"),
                plt.Line2D([], [], color=TEXT_2, linewidth=0.8, linestyle=(0, (5, 2, 1, 2)), label="bias b_t")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(path)
    plt.close(fig)
    return notes


def fig_s2b(instances: pd.DataFrame, success: pd.DataFrame, n: int, path: Path) -> None:
    d = instances.merge(success.rename(columns={"method": "sketch"})[["instance_key", "sketch", "success"]],
                        on=["instance_key", "sketch"], how="left")
    d = d[d.gap > 0]
    fig, axes = plt.subplots(1, len(STRUCTURED), figsize=(FULL_WIDTH * 0.75, 2.6), sharex=True, sharey=True,
                             squeeze=False)
    scatter = None
    for ax, m in zip(axes[0], STRUCTURED):
        dm = d[d.sketch == m]
        for family, marker in FAMILY_MARKERS.items():
            df = dm[dm.family == family]
            scatter = ax.scatter(df.L / df.gap, df.sup_error / df.gap, c=df.success, cmap=SEQUENTIAL, vmin=0, vmax=1,
                                 marker=marker, s=9, linewidths=0)
        ax.axvline(0.25, color=TEXT_2, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.axhline(0.5, color=TEXT_2, linewidth=0.6, linestyle=(0, (1, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("L / Δ  (ℓ1 mass outside K)")
        ax.set_title(f"{LABELS[m]} ({dm.instance_key.nunique()} inst.)")
    axes[0, 0].set_ylabel("‖f − F_K‖∞ / Δ")
    if scatter is not None:
        bar = fig.colorbar(scatter, ax=axes[0].tolist(), shrink=0.9, pad=0.02)
        bar.set_label(f"E1 success rate (n = {n})")
    handles = [plt.Line2D([], [], color=TEXT_2, marker=mk, linestyle="none", label=f"Family {fam}")
               for fam, mk in FAMILY_MARKERS.items()]
    handles += [plt.Line2D([], [], color=TEXT_2, linewidth=0.6, linestyle=(0, (3, 2)), label="L/Δ = 1/4"),
                plt.Line2D([], [], color=TEXT_2, linewidth=0.6, linestyle=(0, (1, 2)), label="‖f − F_K‖∞/Δ = 1/2")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=len(handles))
    fig.savefig(path)
    plt.close(fig)


def s2c_table(runs: pd.DataFrame) -> pd.DataFrame:
    d = runs[runs.t_le_f_x2.astype(bool)].copy()
    rows = []
    for (source, m), g in d.groupby(["source", "sketch"]):
        fh, p2 = g.F_hat_success.astype(bool), g.problem2_success.astype(bool)

        def frac(mask):
            return float(mask.mean()) if len(mask) else math.nan

        rows.append({"source": source, "sketch": m, "runs": len(g),
                     "cond_T": frac(g.cond_T), "cond_f": frac(g.cond_f),
                     "F_hat_success": frac(fh), "cond_T_given_F_hat_success": frac(g.cond_T[fh]),
                     "cond_T_given_F_hat_failure": frac(g.cond_T[~fh]),
                     "problem2_success": frac(p2), "cond_T_given_problem2_success": frac(g.cond_T[p2]),
                     "cond_T_given_problem2_failure": frac(g.cond_T[~p2])})
    return pd.DataFrame(rows)


def write_s2c_tex(table: pd.DataFrame, path: Path) -> None:
    cols = [("runs", "runs"), ("cond_T", r"$k\varepsilon + L < \Delta/2$"), ("F_hat_success", r"$\hat F_K$ ok"),
            ("cond_T_given_F_hat_success", r"cond.\,$|$\,ok"), ("cond_T_given_F_hat_failure", r"cond.\,$|$\,fail"),
            ("problem2_success", "Problem II ok"), ("cond_T_given_problem2_success", r"cond.\,$|$\,II ok"),
            ("cond_T_given_problem2_failure", r"cond.\,$|$\,II fail")]
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\hline",
             "Runs & Sketch & " + " & ".join(c[1] for c in cols) + r" \\", r"\hline"]
    for _, r in table.iterrows():
        cells = [f"{int(r.runs)}"] + [("--" if pd.isna(r[c]) else f"{r[c]:.3f}") for c, _ in cols[1:]]
        lines.append(f"{r.source} & {r.sketch} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# S1: basis pursuit vs matching pursuit
# =============================================================================

S1_FIELDS = ["experiment", "record", "instance_key", "family", "n_rules", "sample_id", "sketch", "n", "success_mp",
             "eta", "bp_residual_over_eta", "bp_positive", "bp_l1", "time", "unit"]


def s1_tables(res: Results, s1_dir: Path, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    bp = pd.DataFrame(RecordStore(s1_dir).load("runs.jsonl", fields=S1_FIELDS, contains=['"experiment": "s1"']))
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
            for decoder, (label, color, marker) in DECODERS.items():
                d = table[(table.family == family) & (table.sketch == m) & (table.decoder == decoder)].sort_values("n")
                if d.empty:
                    continue
                ax.fill_between(d.n, d.ci_low, d.ci_high, color=color, alpha=0.15, linewidth=0)
                ax.plot(d.n, d.success, color=color, marker=marker, markeredgewidth=0)
                budget_axis(ax, sorted(d.n.unique()))
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                instances = table[(table.sketch == m)].instances.max()
                ax.set_title(f"{LABELS[m]} ({instances} inst. per family)")
            if j == 0:
                ax.set_ylabel(f"Family {family}\nsuccess rate")
            if i == len(families) - 1:
                ax.set_xlabel("queries n")
    handles = [plt.Line2D([], [], color=c, marker=mk, label=f"{label} (5 candidates)")
               for label, c, mk in DECODERS.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def make_supplementary(results_dir: Path, fig_dir: Path, e2_dir: Path | None = None, e3_e5_dir: Path | None = None,
                       s1_dir: Path | None = None, overwrite: bool = False, workers: int = 1) -> None:
    results_dir, fig_dir = Path(results_dir), Path(fig_dir)
    existing = [p.name for p in fig_dir.glob("*.pdf") if p.stem in FIGURES_SUPP] if fig_dir.exists() else []
    if existing and not overwrite:
        raise SystemExit(f"{fig_dir} already holds figures ({', '.join(sorted(existing))}); "
                         "choose another --figures directory or pass --overwrite.")
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    res = Results(results_dir)
    res_e2 = Results(Path(e2_dir)) if e2_dir else res
    res_e5 = Results(Path(e3_e5_dir)) if e3_e5_dir else res
    cache = fig_dir / "sample_best.csv"
    earlier = Path(e3_e5_dir) / "figures" / "sample_best.csv" if e3_e5_dir else None
    if not cache.exists() and earlier is not None and earlier.exists():
        shutil.copy(earlier, cache)                  # same E1 records: reuse plot.py's cache
    add_sample_best(res, {"e1"}, cache, workers)
    families, R_values = list(res.params.FAMILIES), list(res.params.R_VALUES)

    # S3
    n_s = grid_budget(res)
    success = per_instance_success(res, n_s)
    success.to_csv(fig_dir / "s3_per_instance_success.csv", index=False)
    fig_s3_histograms(success, families, [max(R_values)], n_s, fig_dir / "s3_success_distribution.pdf")
    fig_s3_histograms(success, families, R_values, n_s, fig_dir / "s3_success_distribution_all_R.pdf")
    fig_s3_properties(success, n_s, fig_dir / "s3_success_vs_properties.pdf")

    # S2
    instances = s2_instances(res, fig_dir, workers)
    fig_s2b(instances, success, n_s, fig_dir / "s2b_sketching_error.pdf")
    runs = s2_runs(res_e5, res_e2, fig_dir, workers)
    sources = {"e5a": [s["descriptor"]["key"] for s in res_e5.selection.get("e5", {}).values() if s],
               "e2": [s["descriptor"]["key"] for s in res_e2.selection["e2"].values()]}
    delta = getattr(res_e5.params, "E5_DELTA", 0.1)
    s2a_notes = fig_s2a(runs, delta, sources, fig_dir / "s2a_coefficient_error.pdf")
    table = s2c_table(runs)
    table.to_csv(fig_dir / "s2c_condition.csv", index=False)
    write_s2c_tex(table, fig_dir / "s2c_condition.tex")

    # S1
    if s1_dir:
        s1, diagnostics = s1_tables(res, Path(s1_dir), np.random.default_rng(BOOTSTRAP_SEED))
        s1.to_csv(fig_dir / "s1_success.csv", index=False)
        diagnostics.to_csv(fig_dir / "s1_bp_diagnostics.csv", index=False)
        fig_s1(s1, families, fig_dir / "s1_decoders.pdf")

    summary = {
        "results_dir": str(results_dir.resolve()), "e2_dir": str(res_e2.dir.resolve()),
        "e3_e5_dir": str(res_e5.dir.resolve()), "s1_dir": str(Path(s1_dir).resolve()) if s1_dir else None,
        "budget_s2b_s3": n_s, "basis": "±1 Walsh (chi_S)", "hoeffding_delta": delta,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "s2a": s2a_notes, "s2_sources": sources,
        "s2b_instances": int(instances.instance_key.nunique()),
        "s2b_fraction_L_below_quarter_gap": {m: float((g.L / g.gap < 0.25).mean())
                                             for m, g in instances[instances.gap > 0].groupby("sketch")},
    }
    (fig_dir / "summary_supplementary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"supplementary figures and tables written to {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="results directory with the E1 runs")
    parser.add_argument("--e2", help="results directory holding E2 (default: results)")
    parser.add_argument("--e3-e5", help="results directory holding the E5a sweep (default: results)")
    parser.add_argument("--s1", help="results directory of `run.py --stage s1` (S1 plotted only if given)")
    parser.add_argument("--figures", default="figures_supp", help="output directory (default: figures_supp)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    make_supplementary(Path(args.results), Path(args.figures), e2_dir=args.e2 and Path(args.e2),
                       e3_e5_dir=args.e3_e5 and Path(args.e3_e5), s1_dir=args.s1 and Path(args.s1),
                       overwrite=args.overwrite, workers=args.workers)


if __name__ == "__main__":
    main()
