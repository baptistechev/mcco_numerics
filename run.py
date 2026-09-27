"""Run the MCCO revision numerics. Every hyperparameter is in params.py.

    python run.py --stage selftest
    python run.py --stage pilot --out pilot
    python run.py --stage all --out results --workers 16
    python run.py --stage e1 --out results --workers 16 --params my_params.py
    python run.py --stage instances theory sweep --out results_v2 --workers 48
    python run.py --stage instances e2select e2 --out results_e2 --e1-records results --workers 48

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
Also: ``pilot`` (checks + timing run + compute estimate) and ``selftest`` (checks only).

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
from mcco_sim.params import load_params  # noqa: E402
from mcco_sim.pilot import pilot  # noqa: E402
from mcco_sim.records import Output, provenance  # noqa: E402
from mcco_sim.stages import STAGES  # noqa: E402

DEFAULT_PARAMS = Path(__file__).resolve().parent / "params.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, nargs="+", choices=[*STAGES, "all", "pilot", "selftest"],
                        help="one or more stages, run in the order of the pipeline")
    parser.add_argument("--out", help="output directory (required except for selftest)")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS), help="hyperparameter file (default: params.py)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--blas-threads", default="1", help="BLAS threads per worker")
    parser.add_argument("--e1-records", help="results directory whose E1 runs e2select reads (read-only)")
    args = parser.parse_args()

    params = load_params(args.params)
    special = {"all", "pilot", "selftest"} & set(args.stage)
    if special and len(args.stage) > 1:
        parser.error(f"--stage {special.pop()} cannot be combined with other stages")
    stage = " ".join(args.stage)
    if stage == "selftest":
        results = run_all(params)
        raise SystemExit(0 if all(r["passed"] for r in results) else 1)
    if args.out is None:
        parser.error("--out is required for this stage")

    out = Output(args.out, params, params_file=args.params)
    out.append("invocations.jsonl", [provenance(params, stage, args.workers) | {"event": "start"}])
    if stage == "pilot":
        pilot(out)
    else:
        for name in (STAGES if stage == "all" else [s for s in STAGES if s in args.stage]):
            options = {"e1_records": args.e1_records} if name == "e2select" else {}
            STAGES[name](out, args.workers, **options)
    out.append("invocations.jsonl", [{"event": "end", "stage": stage,
                                      "timestamp": datetime.now(timezone.utc).isoformat()}])


if __name__ == "__main__":
    main()
