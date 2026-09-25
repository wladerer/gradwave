"""Measured evolutionary code-search harness (AlphaEvolve-style) for gradwave.

The premise (see the session design note): gradwave's prior speedup searches ran
*propose -> vet -> build*, with measurement applied as a late filter. That loop
repeatedly manufactured "corpses" and fell for the isolated-microbench illusion,
because idea-generation and measurement were separate steps with a judgement call
in between. This harness closes the loop: a **population of candidate
implementations** of one kernel is scored by **measured wall-time** (the fitness)
subject to a **hard correctness constraint** (the oracle). Selection pressure IS
measurement, so a regression is culled automatically -- the loop cannot ship a
corpse, and a candidate that "wins" a microbench but breaks physics never passes
the gate.

Layers:

- ``evaluator`` -- kernel-agnostic. A ``Problem`` supplies a frozen workload and an
  oracle; ``evaluate`` runs a candidate, gates it, and (if admissible) times it
  min-of-N. Nothing here knows what kernel is under test.
- ``rr_problem`` -- the first ``Problem`` instance: the Davidson Rayleigh-Ritz step
  (``solvers.davidson._rr``). Chosen as the pilot because it is a genuinely
  separable pure function with a self-contained, gauge/degeneracy-robust oracle.
- ``candidates_rr/`` -- one module per candidate, each exposing ``rr(q, hq, nw)``
  with the exact ``_rr`` signature. New candidates are added as files; the driver
  discovers them. This is where the "mutation" step deposits variants.
- ``driver`` -- discovers the population, evaluates every candidate against the
  problem, ranks the admissible ones by fitness, and reports a leaderboard.

Timing methodology mirrors ``benchmarks/bench_glue_capture.py`` (warmup then
min-of-N under 8 threads); min (not mean) matches the repo's documented practice
(``docs/design/decision-records.md`` evidence lines). All timing must run on asus
via ``scripts/gwq`` -- never on the thinkpad.
"""
