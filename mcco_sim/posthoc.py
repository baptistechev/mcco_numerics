"""Best sampled string of an MCCO run (edit 2 of figure_edits.md).

MCCO's estimate is the best of the sampled strings and the matching-pursuit candidates. The sampled
values are known from the sampling step, so this costs no extra query. The same functions are used
by ``mcco_sample`` for new runs and to recompute the value for records written before this field
existed: a run's sample is the prefix of length n of ``sample_indexes`` (seeded by ``sample_seed``).
"""

from __future__ import annotations

import numpy as np

from .instances import Instance, instance_descriptor
from .seeds import derive_seed


def sample_indexes(params, instance_seed: int, sample_id: int, n_max: int, N: int) -> np.ndarray:
    """The n_max indexes drawn for one sample (a run of budget n uses the first n)."""
    sample_seed = derive_seed(params, "sample", instance_seed, sample_id)
    return np.random.default_rng(sample_seed).integers(0, 2 ** N, size=n_max, dtype=np.int64)


def best_sampled(f: np.ndarray, prefix: np.ndarray) -> tuple[int, float]:
    """Best sampled string of a prefix (first occurrence of the maximum) and its value."""
    values = f[prefix]
    position = int(np.argmax(values))
    return int(prefix[position]), float(values[position])


def combined_estimate(x_mp: int | None, f_mp: float | None, x_sample: int, f_sample: float) -> int:
    """Best of the MP candidates and the sample; ties keep the MP estimate."""
    if x_mp is not None and f_mp >= f_sample:
        return x_mp
    return x_sample


def sample_best_rows(params, descriptor: dict, sample_ids: list[int], budgets: list[int]) -> list[dict]:
    """Best sampled string at every budget, for records that do not have it (old results).
    ``budgets`` must be the budget grid the samples were drawn for (n_max = max(budgets))."""
    inst = Instance(params, descriptor)
    rows = []
    for sample_id in sample_ids:
        indexes = sample_indexes(params, descriptor["instance_seed"], sample_id, max(budgets), inst.N)
        values = inst.f[indexes]
        running = np.maximum.accumulate(values)
        for n in budgets:
            best = running[n - 1]
            position = int(np.argmax(values[:n] == best))
            rows.append({"instance_key": descriptor["key"], "sample_id": sample_id, "n": n,
                         "sample_best_x": int(indexes[position]), "sample_best_f": float(best),
                         "sample_best_rank": float(np.searchsorted(inst.sorted_f, best, side="right") / inst.f.size)})
    return rows


def sample_best_task(args: tuple) -> list[dict]:
    """Picklable wrapper of ``sample_best_rows`` for a process pool: (params, instance record,
    sample ids, budgets)."""
    params, record, sample_ids, budgets = args
    return sample_best_rows(params, descriptor_from_record(params, record), sample_ids, budgets)


def descriptor_from_record(params, record: dict) -> dict:
    """Instance descriptor of an instance record, checked against its recorded seed."""
    d = instance_descriptor(params, record["ensemble"], record["family"], record["n_rules"], record["instance_id"],
                            rule_lengths=tuple(record["rule_lengths_allowed"]))
    if d["instance_seed"] != record["instance_seed"]:
        raise ValueError(f"{record['instance_key']}: regenerated seed differs from the recorded one.")
    return d
