"""S2 (supplementary_numerics_plan.md): error budget of the structured sketches, post hoc from the
records (no troma). Task functions are picklable for a process pool.

Per instance (S2b): L = sum_{S not in K} |a_S(f)|, ||f - F_K||_inf and the gap Delta.
Per run (S2a, S2c), with g = T_t f on the run's sample (values < t set to 0):
    eps_samp = max_{S in K} |a_S(T_t f) - a_hat_S|        (sampling error, Prop. 2)
    b_t      = max_{S in K} |a_S(f) - a_S(T_t f)|         (thresholding bias, depends on t only)
    eps_f    = max_{S in K} |a_S(f) - a_hat_S|             (error against the coefficients of f)
    condition "T": k eps_samp + L(T_t f) < Delta / 2   (Lemma 1 + Prop. 3 applied to T_t f, t <= f(x2))
    condition "f": k eps_f + L(f) < Delta / 2            (as written with f)
    argmax of F_hat_K = sum_{S in K} a_hat_S chi_S equals x* (the event the condition guarantees).
"""

from __future__ import annotations

import numpy as np

from .instances import compute_spectrum
from .posthoc import sample_indexes
from .walsh import empirical_coefficients, k_mask, truncation, walsh_coefficients

SKETCH_WINDOWS = {"quadruplet": 4, "quintuplet": 5}


def _masks(N: int) -> dict[str, np.ndarray]:
    return {name: k_mask(N, k) for name, k in SKETCH_WINDOWS.items()}


def instance_task(args: tuple) -> list[dict]:
    """(N, instance record) -> one row per structured sketch (S2b)."""
    N, record = args
    f = compute_spectrum(record["rules"], N)
    a = walsh_coefficients(f)
    rows = []
    for name, mask in _masks(N).items():
        rows.append({"instance_key": record["instance_key"], "family": record["family"],
                     "n_rules": record["n_rules"], "sketch": name, "k": int(mask.sum()),
                     "L": float(np.abs(a[~mask]).sum()),
                     "sup_error": float(np.abs(f - truncation(a, mask)).max()),
                     "gap": record["gap"], "f_star": record["f_star"], "sigma_f": record["sigma_f"]})
    return rows


def runs_task(args: tuple) -> list[dict]:
    """(params, N, instance record, source, [(sample_id, n, t, threshold_label, n_max)]) -> one row per
    (run group, sketch). A run group is one sample prefix and threshold, shared by the sketches."""
    params, N, record, source, groups = args
    f = compute_spectrum(record["rules"], N)
    a_f = walsh_coefficients(f)
    x_star = record["maximizers"][0]
    masks = _masks(N)
    L_f = {name: float(np.abs(a_f[~mask]).sum()) for name, mask in masks.items()}
    by_t: dict[float, tuple] = {}
    rows = []
    for sample_id, n, t, label, n_max in groups:
        if t not in by_t:
            g_full = np.where(f >= t, f, 0.0)
            a_t = walsh_coefficients(g_full)
            by_t[t] = (a_t, {name: float(np.abs(a_t[~mask]).sum()) for name, mask in masks.items()},
                       {name: float(np.abs(a_f - a_t)[mask].max()) for name, mask in masks.items()})
        a_t, L_t, b_t = by_t[t]
        prefix = sample_indexes(params, record["instance_seed"], sample_id, n_max, N)[:n]
        values = f[prefix]
        values = np.where(values >= t, values, 0.0)
        a_hat = empirical_coefficients(prefix, values, n, N)
        for name, mask in masks.items():
            k = int(mask.sum())
            eps_samp = float(np.abs(a_t - a_hat)[mask].max())
            eps_f = float(np.abs(a_f - a_hat)[mask].max())
            F_hat = truncation(a_hat, mask)
            top = F_hat.max()
            winners = np.flatnonzero(F_hat >= top - 1e-12 * max(abs(top), 1e-300))
            delta = record["gap"]
            rows.append({
                "source": source, "instance_key": record["instance_key"], "family": record["family"],
                "sample_id": sample_id, "n": n, "t": t, "threshold_label": label, "sketch": name, "k": k,
                "t_le_f_x2": bool(t <= record["f_x2"]), "gap": delta,
                "B": record["f_star"] if t <= record["f_star"] else 0.0,   # ||T_t f||_inf
                "eps_samp": eps_samp, "b_t": b_t[name], "eps_f": eps_f, "L_f": L_f[name], "L_t": L_t[name],
                "cond_T": bool(k * eps_samp + L_t[name] < delta / 2),
                "cond_f": bool(k * eps_f + L_f[name] < delta / 2),
                "F_hat_success": bool(winners.size == 1 and winners[0] == x_star),
            })
    return rows
