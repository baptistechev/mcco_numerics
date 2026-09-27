"""MCCO runs (sections 2-4) through the troma API: sample, threshold, sketch, decode."""

from __future__ import annotations

import time
import warnings

import numpy as np
from troma import CombinatorialProblem, DitString, Sample, matching_pursuit

from .instances import Instance
from .params import Params
from .posthoc import best_sampled, combined_estimate, sample_indexes
from .seeds import derive_seed
from .sketches import SketchSet, problem2


class CountingOracle:
    """Black-box f: lookup into the exact spectrum, with a query counter."""

    def __init__(self, f: np.ndarray, N: int) -> None:
        self.f = f
        self.basis = 2 ** np.arange(N - 1, -1, -1, dtype=np.int64)
        self.queries = 0

    def __call__(self, dit_array: np.ndarray) -> float:
        self.queries += 1
        return float(self.f[int(np.asarray(dit_array, dtype=np.int64) @ self.basis)])


def threshold_sample(sample: Sample, t: float) -> Sample:
    """troma threshold semantics: values < t become 0, and zero values are dropped."""
    keep = [i for i, v in enumerate(sample.values) if v >= t and v != 0]
    return Sample(indexes=[sample.indexes[i] for i in keep],
                  values=[sample.values[i] for i in keep],
                  dit_strings=[sample.dit_strings[i] for i in keep])


def mcco_sample(params: Params, inst: Instance, sketches: SketchSet, sample_id: int, budgets: list[int],
                modes: list[dict], problem2_only: list[dict] | None = None) -> list[dict]:
    """All MCCO runs of one sample: every budget (prefix), threshold mode and sketch.

    ``modes`` are decoded; ``problem2_only`` modes only record Problem II (no decoding).
    A mode is {"mode": "adaptive"} (t = Q-th percentile of the n sampled values, zeros
    included) or {"mode": "fixed", "label": str, "t": float}.
    """
    N = inst.N
    sample_seed = derive_seed(params, "sample", inst.d["instance_seed"], sample_id)
    n_max = max(budgets)
    indexes_all = sample_indexes(params, inst.d["instance_seed"], sample_id, n_max, N)
    q = params.Q
    records = []
    for n in budgets:
        prefix = indexes_all[:n]

        def prefix_sampler(n_samples, length, dimension, seed=None):
            return prefix[:n_samples], DitString.from_integers(prefix[:n_samples], length, dimension)

        oracle = CountingOracle(inst.f, N)
        problem = CombinatorialProblem(oracle, problem_size=N)
        start = time.perf_counter()
        full_sample = problem.sampling(n, sampling_function=prefix_sampler, threshold_parameter=None)
        sampled_values = np.concatenate([np.asarray(full_sample.values, dtype=float),
                                         np.zeros(n - len(full_sample.values))])
        t_adaptive = float(np.percentile(sampled_values, q))
        time_sampling = time.perf_counter() - start
        assert oracle.queries == n
        in_sample = set(prefix.tolist())

        for decode, mode in [(True, m) for m in modes] + [(False, m) for m in (problem2_only or [])]:
            if mode["mode"] == "adaptive":
                t, label = t_adaptive, f"q{q:g}_sample"
            else:
                t, label = float(mode["t"]), mode["label"]
            thresholded_sample = threshold_sample(full_sample, t)
            for name in sketches.names:
                problem.sample = thresholded_sample
                start = time.perf_counter()
                problem_sketch = problem.sketching(sketches.maps[name])
                time_sketching = time.perf_counter() - start
                start = time.perf_counter()
                p2 = problem2(sketches.phis[name], problem_sketch.sketch_values, inst.maximizers)
                time_problem2 = time.perf_counter() - start
                base = {
                    **inst.fields(),
                    "sample_id": sample_id,
                    "sample_seed": sample_seed,
                    "sketch": name,
                    "sketch_seed": sketches.seeds[name],
                    "threshold_mode": mode["mode"],
                    "threshold_label": label,
                    "q": q if mode["mode"] == "adaptive" else None,
                    "t": t,
                    "t_is_zero": bool(t == 0),
                    "t_le_f_x2": bool(t <= inst.f_x2),
                    "n": n,
                    "n_kept": len(thresholded_sample.values),
                    **p2,
                }
                if not decode:
                    records.append({"record": "problem2", "method": "mcco", **base,
                                    "time": {"sampling": time_sampling, "sketching": time_sketching,
                                             "problem2": time_problem2}})
                    continue
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if any(problem_sketch.sketch_values):
                        result = matching_pursuit(problem_sketch, iteration_number=params.MP_ITERATIONS,
                                                  optimizer=sketches.optimizers[name])
                        positions = [int(p) for p in result.positions]
                    else:
                        positions = []
                time_decoding = time.perf_counter() - start
                start = time.perf_counter()
                distinct = list(dict.fromkeys(positions))
                values = [oracle(np.asarray(DitString.from_integer(x, N))) for x in distinct]
                time_candidates = time.perf_counter() - start
                x_mp = distinct[int(np.argmax(values))] if distinct else None
                f_mp = max(values) if distinct else None
                # Estimate = best of the sampled strings and the MP candidates (no extra query:
                # the sampled values are known). The MP-only outcome is kept in the *_mp fields.
                x_sample, f_sample = best_sampled(inst.f, prefix)
                x_hat = combined_estimate(x_mp, f_mp, x_sample, f_sample)
                estimate_mp = {f"{k}_mp": v for k, v in inst.evaluate_estimate(x_mp).items()}
                records.append({
                    "record": "run",
                    "method": "mcco",
                    **base,
                    "candidates": [[x, v] for x, v in zip(distinct, values)],
                    "n_candidates_mp": len(positions),
                    "mp_early_stop": any("Early stop" in str(w.message) for w in caught),
                    "queries": n + sum(1 for x in distinct if x not in in_sample),
                    **inst.evaluate_estimate(x_hat),
                    **estimate_mp,
                    "sample_best_x": x_sample,
                    "sample_best_f": f_sample,
                    "sample_n_max": n_max,
                    "time": {"sampling": time_sampling, "sketching": time_sketching, "decoding": time_decoding,
                             "candidates": time_candidates, "problem2": time_problem2},
                })
    return records
