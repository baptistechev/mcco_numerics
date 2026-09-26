"""MCCO revision numerics (simulation_plan.md), built on the TrOMA library.

Hyperparameters live in the repository's ``params.py``; ``run.py`` is the command-line entry point.
"""

from .params import Params, load_params

__all__ = ["Params", "load_params"]
