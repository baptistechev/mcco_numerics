from pathlib import Path

import pytest

from mcco_sim.params import load_params

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def params():
    """The repository's params.py."""
    return load_params(ROOT / "params.py")
