# evolve: measured evolutionary code-search for gradwave kernels

A closed-loop kernel search where **fitness is measured wall-time** and the hard
constraint is a **correctness oracle**. Selection pressure *is* measurement, so a
regressing or physics-breaking candidate is culled by construction — the
anti-corpse property that plain propose-then-vet lacks, and the answer to the
isolated-microbench illusion that has burned past speedup campaigns.

## Pieces

- `evaluator.py` — kernel-agnostic: gate a candidate through the oracle, then time
  it min-of-N (@ 8 threads). The oracle runs strictly *before* the timer.
- `driver.py` — discover a file-per-candidate population and rank by fitness. The
  candidate entrypoint symbol is configurable (`rr`, `apply`).
- `rr_problem.py`, `happly_problem.py` — the two `Problem`s: the Davidson
  Rayleigh-Ritz step (synthetic Hermitian workload, gauge/degeneracy-robust exact
  oracle) and the batched H-apply (real Si `BatchedHamiltonian`, relative-Frobenius
  **tolerance** oracle so approximate-but-valid candidates — Toeplitz, fp32 — are
  admissible with their error visible).
- `mutate.py` + `../evolve_run.py` — the outer loop.
- `candidates_rr/`, `candidates_happly/` — one file per candidate, each exposing the
  entrypoint (`rr`/`apply`).

## Two mutator backends

Both share the one evaluator + oracle; they differ only in how new candidates are
proposed.

**1. Genome evolutionary search (autonomous).** `mutate.py` defines a genome of
independent kernel-design choices (`HAPPLY_GENES`, `RR_GENES`) and a builder that
renders a genome to a callable. `evolve()` runs mutation + crossover + elitist
selection with a genome cache and convergence patience — fully in-process, no
external model. Run it:

    scripts/gwq --host asus bench evolve_run happly 6 100   # problem gens reps

It enumerates a small genome space and genuinely evolves a large one. On the
H-apply it recombined the Toeplitz and fp32 wins (7.8x expansion-grade); on the RR
it converged to baseline and stopped early — the kernel is mined.

**2. LLM/agent-authored candidate files (open-ended).** For ideas *outside* the
genome axes, drop a new `candidates_<problem>/<name>.py` exposing the entrypoint
and re-run the population — `driver.py` discovers it automatically. This is where
non-recombination novelty comes from. `candidates_happly/cached_nl_fold.py` is one
such candidate (memoise the `dij@p` projector fold across the many applies of a
fixed H — off-genome, exact, a projector-heavy/large-cell lever).

Protocol for adding one (used to generate `cached_nl_fold`): give the author the
kernel source, the current champions, the oracle's accuracy grades, and the
interface; it writes ONE pure `def <entry>(...)`; the oracle gates it, so a wrong
candidate is culled, not shipped. Authoring needs no compute; only the evaluation
runs on asus.

## Honest scope

The kernels pilotted here (RR, small-cell H-apply) are largely mined, so the loop
mostly *rediscovers and recombines* known, size-gated wins (Toeplitz, fp32) rather
than inventing new algorithms — which is the correct outcome and validates the
method. Its value at the frontier is (a) an unattended, corpse-proof search over
real design spaces, and (b) a substrate for higher-ceiling targets (large-cell
H-apply where Toeplitz dies, the native-kernel port). All timing runs on asus; the
laptop never runs gradwave compute.
