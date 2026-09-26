"""Instances (section 1): rule generation, exact spectrum and ground truth, E2/E3 selection."""

from __future__ import annotations

import numpy as np

from .params import Params
from .seeds import ENSEMBLE_STREAM, FAMILY_CODE, derive_seed


def instance_descriptor(params: Params, ensemble: str, family: str, n_rules: int, instance_id: int,
                        rule_lengths: tuple[int, ...] | None = None) -> dict:
    """Everything needed to regenerate an instance (the rules are drawn from ``instance_seed``)."""
    N = params.N
    identifiers = [N, FAMILY_CODE[family], n_rules, instance_id]
    return {
        "ensemble": ensemble,
        "family": family,
        "n_rules": n_rules,
        "instance_id": instance_id,
        "N": N,
        "rule_lengths": list(rule_lengths or params.RULE_LENGTHS),
        "seed_identifiers": [ENSEMBLE_STREAM[ensemble], *identifiers],
        "instance_seed": derive_seed(params, ENSEMBLE_STREAM[ensemble], *identifiers),
        "key": f"{ensemble}/{family}/R{n_rules}/{instance_id}",
    }


def build_instance_descriptors(params: Params) -> dict[str, list[dict]]:
    e1 = [instance_descriptor(params, "e1", fam, R, i)
          for fam in params.FAMILIES for R in params.R_VALUES for i in range(params.I)]
    tuning = [instance_descriptor(params, "tuning", fam, R, i)
              for fam in params.FAMILIES for R in params.R_VALUES for i in range(params.I_TUNING)]
    return {"e1": e1, "tuning": tuning}


def e3_descriptor(params: Params, draw: int) -> dict:
    return instance_descriptor(params, "e3", params.E3_FAMILY, params.E3_R, draw,
                               rule_lengths=params.E3_RULE_LENGTHS)


def generate_rules(descriptor: dict) -> list[dict]:
    """Draw |R| distinct rules. Draw order per rule: length, pattern, reward."""
    rng = np.random.default_rng(descriptor["instance_seed"])
    lengths = descriptor["rule_lengths"]
    rules: list[dict] = []
    seen: set[str] = set()
    while len(rules) < descriptor["n_rules"]:
        k = int(lengths[rng.integers(len(lengths))])
        if descriptor["family"] == "L":
            symbols = rng.integers(0, 2, size=k)
            pattern = "".join(str(int(s)) for s in symbols)
        else:
            ends = rng.integers(0, 2, size=2)
            interior = rng.integers(0, 3, size=k - 2)
            pattern = str(int(ends[0])) + "".join("01*"[int(s)] for s in interior) + str(int(ends[1]))
        reward = 1.0 - rng.random()           # uniform in (0, 1]
        if pattern in seen:                   # duplicated patterns are redrawn
            continue
        seen.add(pattern)
        rules.append({"k": k, "pattern": pattern, "reward": reward})
    return rules


def _pattern_masks(pattern: str) -> tuple[int, int]:
    """Bit masks (fixed positions, fixed values) of a window read MSB first."""
    fixed_mask = int("".join("0" if c == "*" else "1" for c in pattern), 2)
    fixed_value = int("".join("1" if c == "1" else "0" for c in pattern), 2)
    return fixed_mask, fixed_value


def compute_spectrum(rules: list[dict], N: int) -> np.ndarray:
    """Exact f over {0,1}^N, index convention of troma.DitString (MSB first)."""
    xs = np.arange(2 ** N, dtype=np.int64)
    f = np.zeros(2 ** N, dtype=float)
    for rule in rules:
        k = rule["k"]
        fixed_mask, fixed_value = _pattern_masks(rule["pattern"])
        count = np.zeros(2 ** N, dtype=np.int64)
        for j in range(N - k + 1):
            window = (xs >> (N - j - k)) & ((1 << k) - 1)
            count += (window & fixed_mask) == fixed_value
        # f is a fixed-order function of the match counts, so equal counts give equal floats.
        f += count * rule["reward"]
    return f


def direct_cost(bits: np.ndarray, rules: list[dict]) -> float:
    """Per-x reference evaluation (as troma examples/problems_generator, with wildcards)."""
    reward = 0.0
    for rule in rules:
        pattern = rule["pattern"]
        k = len(pattern)
        for i in range(len(bits) - k + 1):
            if all(c == "*" or int(c) == int(b) for c, b in zip(pattern, bits[i:i + k])):
                reward += rule["reward"]
    return reward


def fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalized fast Walsh-Hadamard transform."""
    a = np.array(values, dtype=float)
    n = a.size
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        a = np.stack((a[:, 0, :] + a[:, 1, :], a[:, 0, :] - a[:, 1, :]), axis=1)
        h *= 2
    return a.reshape(n)


def int_to_bits(x: int, N: int) -> str:
    return format(int(x), f"0{N}b")


class Instance:
    """Instance, exact spectrum and ground-truth statistics."""

    def __init__(self, params: Params, descriptor: dict, with_wh: bool = False) -> None:
        self.params = params
        self.d = descriptor
        self.N = descriptor["N"]
        self.rules = generate_rules(descriptor)
        self.f = compute_spectrum(self.rules, self.N)
        self.sorted_f = np.sort(self.f)
        self.f_star = float(self.sorted_f[-1])
        self.f_x2 = float(self.sorted_f[-2])  # max over x != x*, for any maximizer x*
        self.maximizers = np.flatnonzero(self.f == self.f_star)
        self.unique = self.maximizers.size == 1
        self.x_star = int(self.maximizers[0]) if self.unique else None
        self.sigma_f = float(self.f.std())
        self.q_exact = float(np.percentile(self.f, params.Q))
        self.wh_sparsity = None
        if with_wh:
            coefficients = fwht(self.f) / 2 ** self.N
            tol = params.WH_RELATIVE_TOLERANCE * np.abs(coefficients).max()
            self.wh_sparsity = int(np.count_nonzero(np.abs(coefficients) > tol))

    def fields(self) -> dict:
        """Identifying fields copied into every record about this instance."""
        return {
            "N": self.N,
            "ensemble": self.d["ensemble"],
            "family": self.d["family"],
            "n_rules": self.d["n_rules"],
            "instance_id": self.d["instance_id"],
            "instance_key": self.d["key"],
            "instance_seed": self.d["instance_seed"],
        }

    def record(self) -> dict:
        cap = self.params.MAX_MAXIMIZERS_RECORDED
        return {
            "record": "instance",
            **self.fields(),
            "seed_identifiers": self.d["seed_identifiers"],
            "rule_lengths_allowed": self.d["rule_lengths"],
            "rules": self.rules,
            "f_star": self.f_star,
            "f_x2": self.f_x2,
            "gap": self.f_star - self.f_x2,
            "n_maximizers": int(self.maximizers.size),
            "maximizers": [int(x) for x in self.maximizers[:cap]],
            "maximizers_truncated": bool(self.maximizers.size > cap),
            "unique_maximizer": bool(self.unique),
            "wh_sparsity": self.wh_sparsity,
            "wh_sparsity_bound": int(sum((self.N - r["k"] + 1) * 2 ** (r["k"] - r["pattern"].count("*"))
                                         for r in self.rules)),
            "mean_f": float(self.f.mean()),
            "var_f": float(self.f.var()),
            "sigma_f": self.sigma_f,
            "fraction_positive": float(np.count_nonzero(self.f > 0) / self.f.size),
            "q_exact_percentile": self.params.Q,
            "t_q_exact": self.q_exact,
        }

    def evaluate_estimate(self, x_hat: int | None) -> dict:
        if x_hat is None:
            return {"x_hat": None, "x_hat_bits": None, "f_x_hat": None, "success": False,
                    "functional_distance": None, "percentile_rank": None, "hamming_distance": None}
        value = float(self.f[x_hat])
        return {
            "x_hat": int(x_hat),
            "x_hat_bits": int_to_bits(x_hat, self.N),
            "f_x_hat": value,
            "success": bool(value == self.f_star),
            "functional_distance": (self.f_star - value) / self.sigma_f,
            "percentile_rank": float(np.searchsorted(self.sorted_f, value, side="right") / self.f.size),
            "hamming_distance": int(np.bitwise_count(self.maximizers ^ int(x_hat)).min()),
        }


def select_e2_instances(params: Params, instance_records: list[dict]) -> dict:
    """One instance per family from the |R| = E2_R ensemble, unique maximizer, with gap and
    WH sparsity closest to the ensemble medians: min |gap - med| / IQR + |s - med| / IQR."""
    records = [r for r in instance_records if r["ensemble"] == "e1"]
    lookup = {d["key"]: d for d in build_instance_descriptors(params)["e1"]}
    selection = {}
    for family in params.FAMILIES:
        pool = [r for r in records if r["family"] == family and r["n_rules"] == params.E2_R
                and r["unique_maximizer"]]
        pool.sort(key=lambda r: r["instance_id"])
        gaps = np.array([r["gap"] for r in pool])
        sparsity = np.array([r["wh_sparsity"] for r in pool], dtype=float)

        def scaled_distance(values):
            q25, q50, q75 = np.percentile(values, [25, 50, 75])
            iqr = q75 - q25 if q75 > q25 else 1.0
            return np.abs(values - q50) / iqr, float(q50), float(iqr)

        d_gap, med_gap, iqr_gap = scaled_distance(gaps)
        d_s, med_s, iqr_s = scaled_distance(sparsity)
        best = int(np.argmin(d_gap + d_s))
        selection[family] = {
            "descriptor": lookup[pool[best]["instance_key"]],
            "gap": float(gaps[best]), "wh_sparsity": int(sparsity[best]),
            "median_gap": med_gap, "iqr_gap": iqr_gap, "median_wh_sparsity": med_s, "iqr_wh_sparsity": iqr_s,
            "score": float(d_gap[best] + d_s[best]), "pool_size": len(pool),
        }
    return selection
