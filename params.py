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
I_TUNING = 10                   # tuning instances per family and |R| (seeds disjoint from E1)

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
DA_DELTA_POINTS = 1000          # random x per tuning instance for the energy scale delta
DA_T0_ACCEPTANCE = 0.5          # T0: uphill move of size delta accepted with probability 1/2
DA_TEND_ACCEPTANCE = 0.01       # T_end: ... accepted with probability 1/100
DA_OFFSET_INCREMENT = 0.1       # dynamic offset increment, in units of delta
DA_TUNING_MULTIPLIERS = (0.5, 1.0, 2.0)  # tuning grid on T0, T_end and offset

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

# --- Pilot (timings and compute estimate) ----------------------------------------------------------
PILOT_INSTANCES = 10
PILOT_SAMPLES = 2

# --- Execution settings (no effect on the results) ------------------------------------------------
BLOCK_SIZE = 25                 # samples per work unit in the sweep and E2 stages
THEORY_CHUNK_COLUMNS = 65536    # column chunk of the dense random-sketch theory (memory bound)
WH_RELATIVE_TOLERANCE = 1e-12   # nonzero threshold for the Walsh-Hadamard sparsity
MAX_MAXIMIZERS_RECORDED = 1000
