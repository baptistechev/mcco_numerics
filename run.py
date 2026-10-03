"""Run the MCCO revision numerics. Every hyperparameter is in params.py.

    python run.py --stage selftest
    python run.py --stage pilot --out results/pilot
    python run.py --stage instances theory --workers 48    # into the next results/resultsN
    python run.py --stage all --out results/other --workers 16
    python run.py --reproduce --workers 48                 # rerun the latest results/resultsN, then compare
    python run.py --reproduce results/results3 --stage e2  # rerun E2 of results3 only

Outputs go to results/resultsN, one directory per run: without --out, a run goes to a new
directory numbered after the latest one (results6 after results5). If the latest run was
interrupted, the same command goes back to its directory and resumes it.
A new results/resultsN starts as a copy of the previous one (highest number below N, or --from
DIR) for every stage not given in --stage (mcco_sim/inherit.py), so each directory holds the
records of every stage and plot.py reads the latest one alone. Give every stage to rerun in the
first command of a new directory: a stage inherited there cannot be run there afterwards.

Stages, run in this order by ``--stage all``:
    instances  every instance, exact ground truth, E2/E3 instance selection
    theory     per-instance checks of Theorem 1 / Corollary 1 (section 4)
    tuning     digital-annealing energy scale per ensemble and tuning grid (section 6)
    e1         MCCO (all sketches) and digital annealing on the E1 ensembles
    sweep      J_SINGLE runs on the E3 instance (all budgets) and the E5a instances
               (budgets up to E5_N_MAX, Problem II only); selects the E5a instances from theory
    e2select   E2 instances and budget with typical E1 success curves (E1 runs of this directory,
               or of --e1-records DIR, read-only)
    e2         theory of the E2 threshold grid, then the threshold sweep at that budget
Supplementary (not run by ``all``):
    s1         basis-pursuit decoding of the E1 samples (S1), |R| in S1_R_VALUES
Also: ``pilot`` (checks + timing run + compute estimate) and ``selftest`` (checks only).

Reproduction: --reproduce [DIR] reruns the stages held by DIR (default: the latest results/resultsN)
with the params file saved in DIR, hence the same seeds, into DIR_repro (resumable like any run),
then compares the records unit by unit with DIR (timings aside) and writes reproduction.json.
--stage restricts the stages rerun; the others are copied from DIR. Only the records of the
stages rerun are compared. The comparison alone: python -m mcco_sim.reproduce DIR OTHER_DIR.

A stage can be interrupted and restarted with the same command: finished work units are
listed in ``progress.jsonl`` and skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _set_blas_threads_from_argv() -> None:
    """BLAS threads must be fixed before numpy is imported (one thread per worker by default)."""
    threads = "1"
    for i, arg in enumerate(sys.argv):
        if arg == "--blas-threads" and i + 1 < len(sys.argv):
            threads = sys.argv[i + 1]
        elif arg.startswith("--blas-threads="):
            threads = arg.split("=", 1)[1]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, threads)


_set_blas_threads_from_argv()

from mcco_sim.checks import run_all  # noqa: E402
from mcco_sim.inherit import add_stages, expand, inherit, inherited_stages, previous_results, stage_sources  # noqa: E402
from mcco_sim.params import load_params  # noqa: E402
from mcco_sim.pilot import pilot  # noqa: E402
from mcco_sim.records import Output, RecordStore, latest_results, next_results, provenance  # noqa: E402
from mcco_sim.reproduce import compare, print_report  # noqa: E402
from mcco_sim.stages import STAGES  # noqa: E402

DEFAULT_PARAMS = Path(__file__).resolve().parent / "params.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="+", choices=[*STAGES, "all", "pilot", "selftest"],
                        help="one or more stages, run in the order of the pipeline")
    parser.add_argument("--reproduce", nargs="?", const="latest", metavar="DIR",
                        help="rerun the stages of a results directory (default: the latest results/resultsN) "
                             "with its params, into DIR_repro, and compare the records with the original; "
                             "--stage restricts the stages rerun (the others are copied)")
    parser.add_argument("--out", help="output directory (default: a new results/resultsN, or the latest one "
                                      "if its last run was interrupted, so the same command resumes it)")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS), help="hyperparameter file (default: params.py)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--blas-threads", default="1", help="BLAS threads per worker")
    parser.add_argument("--e1-records", help="results directory whose E1 runs e2select reads (read-only)")
    parser.add_argument("--from", dest="previous",
                        help="results directory a new --out copies its other stages from "
                             "(default: the previous results/resultsN; 'none' to start empty)")
    args = parser.parse_args()

    source = None
    if args.reproduce:
        source = latest_results() if args.reproduce == "latest" else Path(args.reproduce)
        args.stage = args.stage or list(stage_sources(RecordStore(source)))
        args.params = str(source / "params.py")          # the params the original ran with (same seeds)
        args.out = args.out or str(source.parent / f"{source.name}_repro")
        args.previous = str(source)                       # stages not rerun are copied from the original
        print(f"[run] reproducing {', '.join(args.stage)} of {source} into {args.out}", flush=True)
    elif not args.stage:
        parser.error("--stage (or --reproduce) is required")

    params = load_params(args.params)
    special = {"all", "pilot", "selftest"} & set(args.stage)
    if special and len(args.stage) > 1:
        parser.error(f"--stage {special.pop()} cannot be combined with other stages")
    stage = " ".join(args.stage)
    if stage == "selftest":
        results = run_all(params)
        raise SystemExit(0 if all(r["passed"] for r in results) else 1)
    if args.out is None:
        if stage == "pilot":
            parser.error("--out is required for the pilot")
        args.out = str(next_results())
        print(f"[run] output directory: {args.out}", flush=True)

    new = not (Path(args.out) / "params.json").exists()
    out = Output(args.out, params, params_file=args.params)
    stages = [] if stage == "pilot" else expand(args.stage)
    if new and stage != "pilot":
        previous = None if args.previous == "none" else (Path(args.previous) if args.previous
                                                          else previous_results(args.out))
        if previous is not None:
            sources = inherit(out.dir, previous, stages)
            copied = [s for s, origin in sources["stages"].items() if origin != out.dir.name]
            print(f"[run] {out.dir} copies {', '.join(copied) or 'nothing'} from {previous}"
                  + (f"; params changed: {', '.join(sources['params_changed'])}" if sources["params_changed"] else ""),
                  flush=True)
    clash = inherited_stages(out) & set(stages)
    if clash:
        parser.error(f"{', '.join(sorted(clash))} inherited in {out.dir} from an earlier directory: "
                     "rerun it in a new --out directory")
    out.append("invocations.jsonl", [provenance(params, stage, args.workers) | {"event": "start"}])
    if stage == "pilot":
        pilot(out)
    else:
        add_stages(out, stages)
        for name in stages:
            options = {"e1_records": args.e1_records} if name == "e2select" else {}
            STAGES[name](out, args.workers, **options)
    out.append("invocations.jsonl", [{"event": "end", "stage": stage,
                                      "timestamp": datetime.now(timezone.utc).isoformat()}])
    if source is not None:
        report = compare(source, out.dir, stages)
        print_report(report)
        out.write_json("reproduction.json", report)


if __name__ == "__main__":
    main()
