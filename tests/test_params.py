"""Loading and validation of the hyperparameter file."""

import pytest

from mcco_sim.params import load_params, params_hash
from conftest import ROOT


def test_budget_grid(params):
    assert params.budgets == [50 * 2 ** j for j in range(12)]


def test_missing_and_unknown_names_rejected(tmp_path):
    source = (ROOT / "params.py").read_text()
    missing = tmp_path / "missing.py"
    missing.write_text(source.replace("J_SINGLE = 300", ""))
    with pytest.raises(ValueError, match="J_SINGLE"):
        load_params(missing)
    unknown = tmp_path / "unknown.py"
    unknown.write_text(source + "\nEXTRA = 1\n")
    with pytest.raises(ValueError, match="EXTRA"):
        load_params(unknown)


def test_n_max_must_be_on_the_grid(params):
    with pytest.raises(ValueError, match="power of 2"):
        params.replace(N_MAX=1000)


def test_hash_changes_with_values(params):
    assert params_hash(params) != params_hash(params.replace(J=11))


def test_sketch_subset_keys_required(params):
    for bad in ({"e1_instances": 0}, {"e1_instances": "10"}, {"single_instance": None}):
        sketches = {name: dict(spec) for name, spec in params.SKETCHES.items()}
        sketches["random"].update(bad)
        with pytest.raises(ValueError, match="random"):
            params.replace(SKETCHES=sketches)
