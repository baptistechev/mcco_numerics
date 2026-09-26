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
    I_TUNING: int
    J: int
    J_SINGLE: int
    N_MIN: int
    N_MAX: int
    Q: float
    MP_ITERATIONS: int
    SKETCHES: dict
    DA_DELTA_POINTS: int
    DA_T0_ACCEPTANCE: float
    DA_TEND_ACCEPTANCE: float
    DA_OFFSET_INCREMENT: float
    DA_TUNING_MULTIPLIERS: tuple
    E2_R: int
    E2_PERCENTILES: tuple
    E3_FAMILY: str
    E3_R: int
    E3_RULE_LENGTHS: tuple
    E3_MAX_DRAWS: int
    E5_DELTA: float
    PILOT_INSTANCES: int
    PILOT_SAMPLES: int
    BLOCK_SIZE: int
    THEORY_CHUNK_COLUMNS: int
    WH_RELATIVE_TOLERANCE: float
    MAX_MAXIMIZERS_RECORDED: int

    @property
    def budgets(self) -> list[int]:
        """Budget grid n = N_MIN * 2^j up to N_MAX."""
        grid = [self.N_MIN]
        while grid[-1] < self.N_MAX:
            grid.append(2 * grid[-1])
        return grid

    def to_dict(self) -> dict:
        values = dataclasses.asdict(self)
        values["budgets (derived)"] = self.budgets
        return values

    def replace(self, **changes: Any) -> "Params":
        return validate(dataclasses.replace(self, **changes))


NAMES = [field.name for field in dataclasses.fields(Params)]


def validate(params: Params) -> Params:
    if params.N_MIN < 1 or params.N_MAX < params.N_MIN:
        raise ValueError("Need 1 <= N_MIN <= N_MAX.")
    if params.budgets[-1] != params.N_MAX:
        raise ValueError(f"N_MAX={params.N_MAX} must be N_MIN={params.N_MIN} times a power of 2.")
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
    if not any(spec["single_instance"] for spec in params.SKETCHES.values()):
        raise ValueError("At least one sketch must have 'single_instance': True (sweep, E2).")
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


def params_hash(params: Params) -> str:
    return hashlib.sha256(json.dumps(params.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
