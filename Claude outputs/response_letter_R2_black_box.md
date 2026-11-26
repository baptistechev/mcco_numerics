# Response letter — notes on the black-box setting (R2 #1, #2, #3, minor 4)

Material to draw on for the replies. Not final wording.

## What changed in the manuscript

| Where | Change |
|---|---|
| Abstract | "a new method for black-box combinatorial optimization, which only assumes that the objective is compressible" (was: "…including Black-Box or complicated objective functions") |
| Introduction, 2nd paragraph | "much more difficult" removed (R2 #1). Dynamic programming is described as exact for recursive structures, at a cost set by the interaction range (state dimension), instead of "stops working" (R2 #2). NP-hardness of the general Ising ground state backed by Barahona (1982) (R2 minor 4). The paragraph ends on the black-box scenario. |
| Introduction | "tasks of high complexity" → "black-box combinatorial optimization tasks"; new scope statement: "The scope of this paper is black-box combinatorial optimization: the solver can only query the objective, and the only assumption made on it is that it is compressible." |
| Methods, class of compressible problems | Rules generalized to patterns in {0,1,*}^{k_r} (local and non-local rules). The NP-completeness paragraph is replaced by: when the rules are known, dynamic programming finds the maximum exactly (cost linear in N, exponential only in k = max k_r); the setting of the paper is black-box access, with neither the rules nor their number, lengths or patterns given to the solver. Examples of such objectives: simulators, experiments, learned models, reinforcement learning returns. |
| Numerical Results | 1000 instances over two families (local, non-local), N = 20; solvers are told neither the family, the rules nor their lengths; black-box baseline (digital annealing) with one fixed setting, like MCCO. |

## R2 #2 and #3 — dynamic programming and the difficulty of the benchmark

- We agree with the reviewer. Bounded-length rules on consecutive bits are solved exactly by
  transfer-matrix dynamic programming, keeping the last k − 1 bits as state (dimension 2^{k−1}).
  Non-local rules with wildcards, as now included in the benchmark, still act within a window
  of k_r consecutive bits, so the same holds. Non-nearest-neighbor interactions increase the
  state dimension; they do not make dynamic programming inapplicable.
- The difficulty addressed in the paper is therefore not the structural hardness of the class,
  and the manuscript no longer claims it. It is the black-box setting: dynamic programming
  requires the rules, while MCCO only queries f. The numerical experiments compare black-box
  methods at an equal number of queries, which is how they should be read.

## Anticipated objection — learning the rules, then dynamic programming

- A solver that assumes locality could fit all local terms by regression and then run dynamic
  programming. With a known range k, this needs about (N − k + 1)·2^k queries
  (≈ 10^3 for N = 20, k = 6); if k is unknown, increasing k until the fit is exact costs at
  most about twice that.
- This method assumes that the interactions are local and bounded by a known (or searched)
  range. It is not a black-box method in the sense of the paper.
- MCCO does not require the sketch to match the rules: on an instance with 3 rules of length 4,
  the matched sketch (Quadruplets) and a mismatched one (Quintuplets) reach comparable success
  rates (0.31 and 0.33 at n = 3200; 0.41 and 0.37 at n = 102400, 300 runs; Supplementary
  Figure 1). The structured sketches used here have windows of 4 and 5 bits, while the rules
  have lengths 4–6.
- MCCO is ahead of digital annealing below a few hundred queries (median distance at n = 200:
  1.2–1.8 σ against 1.8–3.7 σ), i.e. below the ≈ 10^3 queries a regression would need just to
  determine the local terms.
- Note for us: MCCO's structured sketches do carry a locality prior (window of the sketch). The
  argument is that the prior does not have to match the problem, not that MCCO uses no prior.
