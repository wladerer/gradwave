# Refactor plan

The physics is validated and the debt is structural, so the rules are
strict. No commit mixes refactoring with physics changes. Every stage ends
with the fast suite green plus a golden-energy check (recorded converged
energies for a fixed system set, compared to full float64 precision).
Stages are ordered by value over risk, and each is independently
shippable. Public entry points keep their signatures until the end.

## Stage 0 — Preconditions (blocking)

The mixer-overhaul batch (Broyden/Johnson mixers, Stoner preconditioner,
Newton finisher, rig) must land first, with its experimental verdict
recorded. Refactoring under an uncommitted feature batch is how work gets
lost. Then record the golden set, converged F for si2 kjpaw, si uspp, al
smeared, cu, ni spin, o2 triplet, nio+U, at fixed settings, committed as
a fixture. The slow suite runs once more as the behavioral baseline.

## Stage 1 — Configuration and layout objects

`scf_uspp` has ~21 keyword parameters and `scf` is close behind. The
mixing vector's layout (sphere size, spin channels, becsum slices, kerker
mask, step scales, block ids) is currently assembled inline in the loop
and re-derived independently in the rig, the Newton finisher, and the
adjoint.

- `SCFOptions` frozen dataclass, tolerances, smearing, criterion, mixer
  choice, flags. `scf_uspp(system, xc, opts=None, **kw)` keeps every
  existing call site working by folding kw into opts.
- `MixLayout` owns pack/unpack between (per-spin densities, becsum) and
  the flat composite vector, plus the kerker mask, step-scale vector, and
  block ids. The SCF loop, the rig's `Packing`, `newton._pack`, and the
  adjoint's split/join all become calls into it.

Risk low. The layout consolidation is the one piece with real bug
potential (normalization conventions), so it lands with a unit test that
round-trips against the current inline code's outputs on a recorded
state.

## Stage 2 — Mixer consolidation

Three mixer classes duplicate constructor plumbing, and JohnsonMixer
borrows `_damped` from BroydenMixer by class-attribute assignment.

- `_DampedMixerBase` holds g2/alpha/kerker/q0/step_scale/extra_precond
  and the damped-step method. Pulay, Broyden, and Johnson subclass it
  with only their own state and `step()`.
- The adaptive block-multiplier machinery stays Pulay-only (it is
  documented as an opt-in stabilizer there).
- Gate, beyond unit tests, a recorded-trajectory fixture, a short
  sequence of (rho_in, rho_out) pairs captured from a real Si run, with
  the assertion that each mixer's outputs are bit-identical before and
  after the refactor.

## Stage 3 — Split the USPP loop

`scf/uspp.py` is ~1400 lines. The split:

- `scf/uspp_setup.py`, `setup_uspp`, `AugSpecies`, augmentation tables,
  `USPPSystem`.
- `scf/uspp_loop.py`, the driver, with the loop body extracted as
  `_scf_iteration(state, ops) -> IterationResult` (densities out, becsum
  out, eigen data, energies). The driver owns mixing, convergence, trust
  logic, rescue.
- `scf/uspp.py` remains as a facade re-exporting the public names.

The extraction pays twice. The Newton finisher and the mixer rig
currently evaluate the raw map through `scf_uspp(max_iter=1,
start_from=..., etol=1e-300)` round trips that rebuild projectors and
guess machinery every call. Both switch to `_scf_iteration` directly,
which removes the sentinel-tolerance hack and roughly halves the rig's
J-apply cost.

Gate, fast suite, golden energies, plus three slow anchors (si_paw vs QE,
ni spin vs QE, batched-vs-per-k equality).

**Prepared (2026-07-13).** docs/refactor_stage3.md holds the measured
consumer map, the string-anchored anatomy of scf_uspp, the
IterOps/IterState/_scf_iteration contract, the four-commit sequence
with per-commit gates, and the trap list.

**Done (2026-07-13, commits 84f7a78 / 15c6cb6 / 4d3+30e9924).** Four
commits as planned: setup → uspp_setup.py (287 lines), driver →
uspp_loop.py with uspp.py a 22-line facade, _scf_iteration extracted
(loop body verbatim; driver keeps tolerance schedule, rescue,
convergence, trust, mixing), newton + rig switched to direct
evaluation. Every gate green including the nine slow anchors. The
sentinel-tolerance hack is gone; measured payoff: rig J-apply 23.5 s →
7.1 s warm (|F(x*)−x*| 2.4e-11 through the new path), newton polish
suite minutes → 44 s.

## Stage 4 — NC/USPP sharing (deliberately minimal)

Full unification (NC as the S=1 special case of the generalized loop) is
the architecturally right end state and QE's own structure, but it
touches everything validated on the NC path, bands, SOC, noncollinear,
+U, the NC implicit module. The cost-benefit today favors the minimal
version.

- Extract genuinely shared blocks into `scf/common.py`, occupation and
  Fermi handling, convergence criteria (the energy-tail logic is already
  duplicated in spirit), trust-region and rescue logic, mixing-vector
  helpers via `MixLayout`.
- A one-day spike measures the S=1 fast path's overhead inside the
  generalized loop on the NC benchmark set. Full unification proceeds
  only if the overhead is <10% and NC maintenance is actually hurting,
  neither is true today.

## Stage 5 — Test tiering

The "fast" suite has crept from seconds to ~10 minutes and the four
heaviest spin tests cost 2+ hours. Tiers by marker.

- `fast` (unit + cheap integration, target under 2 minutes total), the
  default for every commit gate in this plan.
- `standard` (current not-slow set), CI.
- `slow` (QE comparisons), nightly or pre-release.
- `torture` (ni spin, o2, anything over 10 minutes), manual, exercised
  when their subsystems change.

One audit pass moves tests that crept upward back down (several
"fast" tests run multi-second SCFs that a lower cutoff would test
equally well).

**Done (2026-07-13).** Measured with `--durations=0`: the 138-test
not-slow suite took 588 s, of which the 15 tests over 5 s accounted for
~510 s. Those 15 carry `standard` (file-level for metal_forces,
uspp_batched_equality, dos, golden_energies; function-level for the
stress_vs_qe CI cases and stress_with_symmetry; param-level for
gaas_pbe_ci). Five monsters moved from `slow` to `torture` (ni+U vs QE,
o2 triplet, spin-degenerate forces/stress, nio linear-response U vs hp,
nio dE/dU Hellmann-Feynman). Result: fast tier 122 tests in 81 s,
fast+standard identical to the old not-slow set (138), torture 5. Tier
commands are recorded in pyproject.toml and the README. The audit's
lower-cutoff rewrites of individual tests were not needed to reach the
2-minute target and are left for when a fast test regresses.

## Stage 6 — Benchmark hygiene

- One shared case-construction helper for the benchmark family
  (delta_factor, lejaeghere, bench_* all hand-roll the same dicts).
- `cases.py` should hold physics choices only. The per-case mixer
  workarounds (mixing_alpha 0.3, seed bumps) are deleted the moment the
  mixer overhaul's winner makes them unnecessary, which is the point of
  the overhaul. Anything that must stay gets a comment naming the reason
  and the evidence.

## Sequencing and effort

Stage 0 is blocked on the current experiments. Stages 1 and 2 fit in one
session and unblock most of the readability win. Stage 3 is a session on
its own and should not start the same day the batch lands. Stages 5 and
6 are filler-sized and can interleave. Stage 4's spike waits until the
others settle.

Nothing here changes numbers. If any stage's gate shows a numerical
difference beyond float noise, the stage stops and the difference is
understood before proceeding, per the validation culture that is the
reason this refactor is safe at all.
