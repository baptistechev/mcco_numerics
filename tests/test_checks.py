"""Correctness checks of mcco_sim/checks.py (also run by the pilot on the target machine)."""

import pytest

from mcco_sim.checks import CHECKS


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_check(params, check):
    failed = [r for r in check(params) if not r["passed"]]
    assert not failed, failed
