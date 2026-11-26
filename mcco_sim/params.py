"""Load the hyperparameter file (``params.py``) into a frozen, picklable ``Params`` object."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class Params:
    """The values of the hyperparameter file (same names), plus the derived budget grid."""

    N: int
    MASTER_SEED: int
    FAMILIES: tuple
    R_VALUES: tuple
    RULE_LENGTHS: tuple
    I: int
    J: int
    J_SINGLE: int
    N_MIN: int
    N_MAX: int
    Q: float
    MP_ITERATIONS: int
    SKETCHES: dict
    DA_T0: float
    DA_TEND: float
    DA_OFFSET_INCREMENT: float
    E2_R: int
    E2_PERCENTILES: tuple
    E3_FAMILY: str
    E3_R: int
    E3_RULE_LENGTHS: tuple
    E5_DELTA: float
    E5_N_MAX: int
    S1_R_VALUES: tuple
    S1_SKETCHES: tuple
    S1_INSTANCES: int
    S1_J: int
    S1_BUDGET_STRIDE: int
    S1_BP_ITERATIONS: int
    PILOT_INSTANCES: int
    PILOT_SAMPLES: int
    BLOCK_SIZE: int
    THEORY_CHUNK_COLUMNS: int
    WH_RELATIVE_TOLERANCE: float
    MAX_MAXIMIZERS_RECORDED: int

    @property
    def budgets(self) -> list[int]:
        """Budget grid n = N_MIN * 2^j up to N_MAX."""
        return doubling_grid(self.N_MIN, self.N_MAX)

    @property
    def e5_budgets(self) -> list[int]:
        """Budget grid of the E5a sweep: n = N_MIN * 2^j up to E5_N_MAX."""
        return doubling_grid(self.N_MIN, self.E5_N_MAX)

    @property
    def s1_budgets(self) -> list[int]:
        """Budgets of S1: every S1_BUDGET_STRIDE-th budget counted down from N_MAX (the maximum must
        stay N_MAX: the sample is drawn for the largest budget, as in E1)."""
        return self.budgets[::-1][::self.S1_BUDGET_STRIDE][::-1]

    def to_dict(self) -> dict:
        values = dataclasses.asdict(self)
        values["budgets (derived)"] = self.budgets
        values["e5_budgets (derived)"] = self.e5_budgets
        values["s1_budgets (derived)"] = self.s1_budgets
        return values

    def replace(self, **changes: Any) -> "Params":
        return validate(dataclasses.replace(self, **changes))


NAMES = [field.name for field in dataclasses.fields(Params)]


def doubling_grid(n_min: int, n_max: int) -> list[int]:
    grid = [n_min]
    while grid[-1] < n_max:
        grid.append(2 * grid[-1])
    return grid


def validate(params: Params) -> Params:
    if params.N_MIN < 1 or params.N_MAX < params.N_MIN:
        raise ValueError("Need 1 <= N_MIN <= N_MAX.")
    if params.budgets[-1] != params.N_MAX:
        raise ValueError(f"N_MAX={params.N_MAX} must be N_MIN={params.N_MIN} times a power of 2.")
    if params.E5_N_MAX < params.N_MAX or params.e5_budgets[-1] != params.E5_N_MAX:
        raise ValueError(f"E5_N_MAX={params.E5_N_MAX} must be >= N_MAX and N_MIN={params.N_MIN} times a power of 2.")
    unknown_families = set(params.FAMILIES) - {"L", "W"}
    if unknown_families:
        raise ValueError(f"Unknown families {sorted(unknown_families)}.")
    for name, spec in params.SKETCHES.items():
        if spec.get("type") not in ("nearest_neighbors", "gaussian"):
            raise ValueError(f"Sketch {name}: unknown type {spec.get('type')}.")
        subset = spec.get("e1_instances", "missing")
        if not (subset is None or (isinstance(subset, int) and not isinstance(subset, bool) and subset >= 1)):
            raise ValueError(f"Sketch {name}: 'e1_instances' must be None or an int >= 1.")
        if not isinstance(spec.get("single_instance"), bool):
            raise ValueError(f"Sketch {name}: 'single_instance' must be True or False.")
    if not 0 < params.DA_TEND <= params.DA_T0 or params.DA_OFFSET_INCREMENT < 0:
        raise ValueError("Need 0 < DA_TEND <= DA_T0 and DA_OFFSET_INCREMENT >= 0.")
    if not any(spec["single_instance"] for spec in params.SKETCHES.values()):
        raise ValueError("At least one sketch must have 'single_instance': True (sweep, E2).")
    unknown_s1 = set(params.S1_SKETCHES) - set(params.SKETCHES)
    if unknown_s1:
        raise ValueError(f"S1_SKETCHES: unknown sketches {sorted(unknown_s1)}.")
    if min(params.S1_INSTANCES, params.S1_J, params.S1_BUDGET_STRIDE) < 1:
        raise ValueError("S1_INSTANCES, S1_J and S1_BUDGET_STRIDE must be >= 1.")
    return params


def load_params(path: str | Path) -> Params:
    """Read every UPPER_CASE name of the file; all of NAMES are required and no others allowed."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location("_mcco_params", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {name: getattr(module, name) for name in dir(module) if name.isupper()}
    missing = [name for name in NAMES if name not in values]
    unknown = sorted(set(values) - set(NAMES))
    if missing or unknown:
        raise ValueError(f"{path}: missing {missing}, unknown {unknown}")
    values = {name: tuple(v) if isinstance(v, list) else v for name, v in values.items()}
    return validate(Params(**values))


def params_from_dict(values: dict) -> Params:
    """Params from the resolved values stored in an output directory (``params.json``)."""
    values = {k: v for k, v in values.items() if k in NAMES}
    missing = [name for name in NAMES if name not in values]
    if missing:
        raise ValueError(f"params.json: missing {missing}")
    values = {name: tuple(v) if isinstance(v, list) else v for name, v in values.items()}
    return validate(Params(**values))


def params_hash(params: Params) -> str:
    return hashlib.sha256(json.dumps(params.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
