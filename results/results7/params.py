"""Hyperparameters of the MCCO revision numerics (see simulation_plan.md).

This file is the single place where the simulation parameters live: the code has no defaults.
Every UPPER_CASE name below is required. A copy of this file and its resolved values are
stored in each output directory, and resuming a run with different values is refused.
"""

# --- Problem and seeds -------------------------------------------------------------------
N = 20                          # bit-string length
MASTER_SEED = 20260926          # every seed is derived from this and the identifiers of the draw

# --- Instances (section 1) -----------------------------------------------------------------
FAMILIES = ("L", "W")           # L: patterns in {0,1}; W: interior symbols in {0,1,*}
R_VALUES = (1, 2, 3, 4, 5)      # number of rules |R|
RULE_LENGTHS = (4, 5, 6)        # k_r, uniform
I = 100                         # instances per family and |R| (E1)

# --- Sampling protocol (section 3) ------------------------------------------------------------
J = 10                          # independent runs per instance (E1)
J_SINGLE = 300                  # independent runs per instance (E2, E3, E5a)
N_MIN = 50                      # budget grid n = N_MIN * 2^j, j = 0, 1, ... up to N_MAX
N_MAX = 102_400                 # must be N_MIN times a power of 2

# --- MCCO (section 2) --------------------------------------------------------------------------
Q = 85.0                        # threshold t = Q-th percentile of the sampled values
MP_ITERATIONS = 5               # matching-pursuit iterations = number of candidates
# Per sketch: "e1_instances" = None (all I instances) or K (only instance ids < K in each
# family and |R|); "single_instance" = whether it runs in the sweep, E2 and the E3 theory.
SKETCHES = {
    "quadruplet": {"type": "nearest_neighbors", "k": 4, "e1_instances": None, "single_instance": True},
    "quintuplet": {"type": "nearest_neighbors", "k": 5, "e1_instances": None, "single_instance": True},
    # Gaussian sketch, one per instance; "rows" is a number or the name of a sketch to match.
    # Costly and expected to perform poorly: a subset of E1 only.
    "random": {"type": "gaussian", "rows": "quintuplet", "e1_instances": 10, "single_instance": False},
}

# --- Digital annealing baseline (sections 2 and 6) -------------------------------------------
# One fixed setting for every family and |R|, as for MCCO (no per-ensemble tuning). Absolute
# values in units of f (rewards in (0, 1]): the typical single-flip |Delta f| is about 0.5 in
# every ensemble (0.43-0.62), so T0 accepts an uphill move of 0.25 with probability 1/2, T_end
# one of 0.5 with probability 1/100, and the offset grows by 0.1 per rejected step.
DA_T0 = 0.36                    # initial temperature (geometric schedule from T0 to T_end)
DA_TEND = 0.11                  # final temperature
DA_OFFSET_INCREMENT = 0.1       # dynamic offset increment after a step with no accepted flip

# --- E2: threshold sweep -------------------------------------------------------------------------
E2_R = 5                        # E2 instances come from the |R| = E2_R ensembles
E2_PERCENTILES = (50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99, 99.5, 99.9)  # plus t = 0, f(x2), mid

# --- E3: sketch/rule mismatch ----------------------------------------------------------------------
E3_FAMILY = "L"
E3_R = 3                        # the E3 instance is draw 0 (any number of maximizers)
E3_RULE_LENGTHS = (4,)

# --- E5: theory check --------------------------------------------------------------------------------
E5_DELTA = 0.1                  # delta of the sample size of Eq. (7)
E5_N_MAX = 409_600              # budget grid of the E5a sweep: N_MIN * 2^j up to E5_N_MAX (E1 stays at N_MAX)

# --- S1 (supplementary): basis pursuit vs matching pursuit -----------------------------------------
S1_R_VALUES = (5,)              # E1 ensembles (|R|) decoded by basis pursuit, both families
S1_SKETCHES = ("quadruplet", "quintuplet")  # random dropped: ~21 h per unit (dense 4.3 GB matrix)
S1_INSTANCES = 20               # instance ids < S1_INSTANCES per family and |R| (E1 has I)
S1_J = 5                        # sample ids 0..S1_J-1 (the E1 samples of the same ids)
S1_BUDGET_STRIDE = 2            # every S1_BUDGET_STRIDE-th budget counted down from N_MAX (100, 400, ..., 102400)
S1_BP_ITERATIONS = 1000         # Chambolle-Pock iterations per decode (fixed, no stopping rule)

# --- Pilot (timings and compute estimate) ----------------------------------------------------------
PILOT_INSTANCES = 10
PILOT_SAMPLES = 2

# --- Execution settings (no effect on the results) ------------------------------------------------
BLOCK_SIZE = 25                 # samples per work unit in the sweep and E2 stages
THEORY_CHUNK_COLUMNS = 65536    # column chunk of the dense random-sketch theory (memory bound)
WH_RELATIVE_TOLERANCE = 1e-12   # nonzero threshold for the Walsh-Hadamard sparsity
MAX_MAXIMIZERS_RECORDED = 1000
