"""Run the MCCO revision numerics. Every hyperparameter is in params.py.

    python run.py --stage selftest
    python run.py --stage pilot --out pilot
    python run.py --stage all --out results --workers 16
    python run.py --stage e1 --out results --workers 16 --params my_params.py

Stages, run in this order by ``--stage all``:
    instances  every instance, exact ground truth, E2/E3 instance selection
    theory     per-instance checks of Theorem 1 / Corollary 1 (section 4)
    tuning     digital-annealing energy scale per ensemble and tuning grid (section 6)
    e1         MCCO (all sketches) and digital annealing on the E1 ensembles
    sweep      J_SINGLE budget sweep on the E2 and E3 instances (E3, E5a, E2 budget)
    e2         threshold sweep at the automatically chosen budget
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
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all", "pilot", "selftest"])
    parser.add_argument("--out", help="output directory (required except for selftest)")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS), help="hyperparameter file (default: params.py)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--blas-threads", default="1", help="BLAS threads per worker")
    args = parser.parse_args()

    params = load_params(args.params)
    if args.stage == "selftest":
        results = run_all(params)
        raise SystemExit(0 if all(r["passed"] for r in results) else 1)
    if args.out is None:
        parser.error("--out is required for this stage")

    out = Output(args.out, params, params_file=args.params)
    out.append("invocations.jsonl", [provenance(params, args.stage, args.workers) | {"event": "start"}])
    if args.stage == "pilot":
        pilot(out)
    else:
        for name in (STAGES if args.stage == "all" else [args.stage]):
            STAGES[name](out, args.workers)
    out.append("invocations.jsonl", [{"event": "end", "stage": args.stage,
                                      "timestamp": datetime.now(timezone.utc).isoformat()}])


if __name__ == "__main__":
    main()
