"""Deterministic seeds: every random draw is seeded from the master seed and its identifiers."""

from __future__ import annotations

import numpy as np

from .params import Params

STREAM_ID = {
    "instance": 1,
    "tuning_instance": 2,
    "e3_instance": 3,
    "sample": 10,
    "random_sketch": 11,
    "da_run": 12,
    "da_delta": 13,
}
FAMILY_CODE = {"L": 0, "W": 1}
ENSEMBLE_STREAM = {"e1": "instance", "tuning": "tuning_instance", "e3": "e3_instance"}


def derive_seed(params: Params, stream: str, *identifiers: int) -> int:
    """64-bit seed derived deterministically from the master seed, a stream and identifiers."""
    entropy = [int(params.MASTER_SEED), STREAM_ID[stream], *[int(i) for i in identifiers]]
    state = np.random.SeedSequence(entropy).generate_state(2, np.uint32)
    return (int(state[0]) << 32) | int(state[1])
