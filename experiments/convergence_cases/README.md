# Convergence case studies

Research notes for validating the SCF flight recorder (`scf.recorder`, PR #203)
against the documented hard cases in `docs/manual/wisdom.md` and
`docs/manual/performance.md`, and one convergence-technique experiment driven by
what the traces showed. Everything here is a stacked archive on
`feat/scf-flight-recorder`. It is never a PR. The recorder threshold fixes below
were committed onto `feat/scf-flight-recorder` itself.

Runners live here. Traces are in `traces/`, one JSON per SCF. Per-part results
are `results_part1.json`, `results_positives.json`, `results_part2_k6.json`, and
`results_part2_k10.json`. Reproduce with `uv run python run_part1.py`,
`run_positives.py`, and `run_part2.py`.

## What the recorder does, and what broke

`diagnose()` reads the last 5 iterations and returns up to three tags. That
window is the pathological phase for a stuck SCF, but it is the settled noise
floor for a converged one. Running the documented cases surfaced three false
positives that all trace to reading the converged tail. The first pass of
validation therefore turned into a threshold audit before it could be a tag
audit.

Two guards were added to `scf.recorder.diagnose()` on `feat/scf-flight-recorder`
(commit `fix(scf/recorder): guard moment-collapse and charge-sloshing against a
converged tail`), each justified against the whole battery below.

- `_MOMENT_COLLAPSE_FLOOR = 0.1` muB. The `seed_moment` passed to the recorder
  on the USPP/PAW path is the smooth-grid moment of the SAD atomic seed. For a
  strongly seeded 3d metal that is many times the converged smooth moment (fcc
  Ni at `start_mag=0.6` seeds ~10.8 muB and converges to ~0.79 muB smooth), so
  the old test `mags[-1] < 0.1*seed_moment` was trivially true for every healthy
  ferromagnetic PAW metal. A collapse is a fall to the nonmagnetic branch, so
  the final moment now also has to be below 0.1 muB in absolute terms. The
  smallest held moment in the battery (fcc Ni, 0.79 muB smooth) sits about 8x
  above the floor, and a real collapse drives the moment below 0.05 muB.
- `_SLOSH_RES_FLOOR = 1e-4`. Once the residual reaches the noise floor the
  `|G|`-shell decomposition is roundoff and its non-monotonicity is meaningless.
  A cleanly converged fcc Ni (NC, johnson) tripped charge-sloshing at a ~1e-6
  residual for exactly this reason. The tag now requires the window residual
  above 1e-4, which sits above a smeared metal's occupation-noise floor and well
  below any genuinely stuck residual.

Both synthetic true-positive unit tests still fire (collapse to 0.02 muB, sloshing
at a 0.1 residual), and two regression tests were added. Replaying every stored
trace through the patched `SCFRecorder.diagnose()` gives zero tags on the
converged battery, matching the corrected expectation.

## Part 1, tags vs the documented hard cases

Iteration counts are the trustworthy metric here, not wall time. The "fired"
column is post-fix. Where the pre-fix behavior differed it is called out.

| case | system and settings | iters | doc iters | expected | fired (post-fix) | verdict |
|---|---|---|---|---|---|---|
| ni_nc_underseed_pulay | fcc Ni NC nspin=2, 8x8x8, 45 Ry, `start_mag=0.02`, pulay | 14 | | moment-collapse | none | moment held at 0.75 muB, no collapse to detect |
| ni_nc_underseed_johnson | same, johnson | 14 | | ? | none | pre-fix fired charge-sloshing (false), now clean |
| ni_nc_seeded_johnson | same, `start_mag=0.6`, johnson | 14 | | none | none | match |
| ni_uspp_johnson | fcc Ni PAW nspin=2, 45/360 Ry, `start_mag=0.6`, johnson | 18 | 18 | none | none | pre-fix fired moment-collapse (false), now clean, iters match |
| ni_uspp_pulay | same, pulay | 27 | 27 | none | none | pre-fix false moment-collapse, now clean, iters match |
| fe_uspp_pulay | bcc Fe PAW nspin=2, 45/360 Ry, `start_mag=0.7`, pulay | 30 | 29 | none | none | match within 1 iter |
| fe_uspp_johnson | same, johnson forced | 16 | 93 | becsum blowup? | none | blowup did NOT reproduce, see below |
| al_slab4_kerker | Al(100) 4-layer NC, 25 Ry, 4x4x1 | 21 | 21 | charge-sloshing | none | MISS, structural, see below |
| al_slab4_local_tf | same, `precond="local_tf"` | 17 | 17 | none | none | match |
| al_slab6_kerker | Al(100) 6-layer NC | 27 | 27 | charge-sloshing | none | MISS, structural |
| al_slab6_local_tf | same, local_tf | 21 | 21 | none | none | match |
| coarse_al_tiny | fcc Al NC, 2x2x2, 0.01 eV smearing | 9 | | quiet | none | match, reorder total 5 spread |
| coarse_al_wide | same, 0.5 eV smearing | 9 | | quiet | none | match |
| pt_paw_johnson | fcc Pt PAW, 40/400 Ry, 6x6x6, 0.2 eV | 12 | 13 | level-crossing where real | none | see note, 86 early reorders resolve by convergence |

### What the traces show that the scalar residual hides

- fcc Ni USPP holds the moment cleanly. The smooth-grid moment settles 2.48 to
  0.79 muB over the first few iterations and stays there. The scalar residual
  looks identical to any converged run. The moment trajectory is what shows the
  branch was never lost, and it is also what exposed the inflated seed that
  produced the false positive.
- The Al slabs are long-wavelength dominated the whole way. The 2-lowest-shell
  residual fraction rides 0.7 to 0.99 across the active phase on both kerker
  slabs, exactly the sloshing signature, while `drho` falls almost
  monotonically. The scalar residual reports healthy convergence. The shell
  decomposition reports that the convergence is a slow long-wavelength grind, and
  that is the real cost the `local_tf` remedy removes (21 to 17, 27 to 21).
- fcc Pt reorders hard early then settles. The band-reordering count is
  `[0, 64, 17, 5, 0, 0, ...]`, so 86 reorderings all sit in iterations 2 to 4
  while the eigenvalues are still moving 1.5 eV per step, then go to zero. The
  scalar residual never shows this initial subspace churn.

### Fe johnson blowup did not reproduce

`docs/manual/wisdom.md` records bcc Fe on the USPP path going 29 (pulay) to 93
(johnson) when johnson is forced, from dropping the becsum step-damping crutch.
Here pulay took 30 and forced johnson took 16, so johnson was better, not a
blowup. The likely reason is the crutch re-audit the wisdom page itself
describes, namely QE mixes becsum unscaled and matching that closed the gap, so
the becsum-oscillation the 93-iteration case exhibited is no longer present at
these settings. There was no becsum-driven blowup to characterize, so a
`becsum-oscillation` tag has no positive to key on from this system. Recorded,
not forced.

## Positive controls, the patched tags still fire

Guarding against the converged tail could have neutered the detectors, so two
constructed pathologies confirm they still fire on the real thing
(`run_positives.py`).

| control | system and settings | outcome | fired | correct |
|---|---|---|---|---|
| pos_cr_collapse | bcc Cr NC nspin=2, SpinPBE, FM `start_mag=0.7` | moment collapses to 0.000 muB | moment-collapse | yes, 0.0 muB is below the 0.1 floor |
| pos_al_slab_sloshing | Al(100) 6-layer NC, Kerker off, `mixing_alpha=0.9`, `mixing_history=2` | diverges, `drho` stalls at ~97 | charge-sloshing and level-crossing | yes, 89% low-shell weight, non-monotone, residual far above the floor, 5/5 recent reorders |

The Cr FM state collapsed under both SpinPBE and LSDA at these settings, which
differs from the wisdom page's 3.1 muB hold. That number is for the magnetic-IBZ
Cr setup, not this FM primitive cell, so the difference is the system, not the
tag. The point the control makes is narrow, namely the tag fires when and only
when the moment actually reaches zero.

## Structural gaps found, recorded not forced

The common cause is that `diagnose()` is post-hoc over the last 5 iterations, so
on a converged run it inspects the settled tail and cannot report a pathology
that happened mid-run and then resolved.

- charge-sloshing misses the documented converging slab. The kerker Al slabs are
  long-wavelength dominated for 15 or more iterations, but a competent Pulay
  mixer drives the residual down almost monotonically, so the `not falling`
  requirement suppresses the tag. The 2-lowest-shell fraction being high is the
  discriminator between healthy convergence and sloshing, not the
  non-monotonicity. Catching the slab needs either a mid-run detector or a
  metric keyed on a long-wavelength-dominated active phase rather than on
  oscillation in the tail. The Part 2 detector is the mid-run version.
- level-crossing cannot report a crossing that resolves before convergence. fcc
  Pt reorders 86 times in iterations 2 to 4 then converges clean, so the last-5
  window is all zeros and the tag stays quiet. That is arguably correct, since
  early Davidson subspace sorting is not a pathology, but it means level-crossing
  only fires on a run where reordering persists into the tail, which a converging
  run never shows. The only positive observed was the diverging sloshing control.
- moment-collapse is now correct on the battery, but its `seed_moment` reference
  differs by construction between the NC path (smooth-grid `int|m|`, which equals
  the full moment for norm-conserving) and the USPP/PAW path (smooth-grid moment
  of the SAD seed, which excludes the augmentation moment). The absolute floor
  side-steps the mismatch. A cleaner fix would define the collapse reference from
  the first post-seed iteration rather than the SAD seed, which is a larger
  change than a threshold and is left as a note.

## Part 2, trace-driven auto-remediation for charge-sloshing

The structural finding motivates the experiment. Since `diagnose()` cannot drive
a mid-run decision, `run_part2.py` prototypes a mid-run detector that reads the
same shell decomposition and, unlike `diagnose()`, does not require
non-monotonicity. It flags when the mean 2-lowest-shell fraction over the
still-active iterations (residual above 1e-4) exceeds 0.5.

Per slab we compare three policies. A is robust from the start (`local_tf`), the
documented best. B is kerker from the start, the sloshing-prone default. C runs
kerker for `k` iterations, and if the detector flags, warm-restarts with
`local_tf` and a fresh mixer via `start_from` (the mixing-history reset the task
describes). Total C iterations include the wasted k-iteration prefix. The honest
question is whether detect-plus-restart beats just always using the robust
preconditioner. Prototype only, no production path touched.

| slab | A local_tf | B kerker | C detect@6 | C detect@10 | C beats A |
|---|---|---|---|---|---|
| Al(100) 4-layer | 17 | 21 | 20 (not flagged) | 22 (flagged, 10+12) | no |
| Al(100) 6-layer | 21 | 27 | 26 (not flagged) | 24 (flagged, 10+14) | no |

Two things came out of this.

- Detection timing decides whether the detector even fires. At `k=6` the mid-run
  detector does not flag, because the long-wavelength dominance emerges only
  after iteration 5 or 6. The first six kerker iterations average a
  2-lowest-shell fraction near 0.35, below the 0.5 threshold. So C@6 fell through
  to kerker with a mixer reset at iteration 6, which marginally beat pure kerker
  (20 vs 21, 26 vs 27) but lost to `local_tf`.
- Even correct detection loses to robust-from-start. At `k=10` the detector
  flags on both slabs and C switches to `local_tf`, giving 10+12=22 and 10+14=24.
  Both exceed `local_tf` from the start (17 and 21). The wasted kerker prefix
  costs more than the switch saves. All energies match to 1e-3 eV across
  policies, so this is purely iteration count.

The answer to the task's honest question is no. Detect-and-restart does not beat
just always using the robust preconditioner on these slabs. The cheapest correct
policy is to start on `local_tf`. The mid-run detector is still the right shape
for the charge-sloshing structural gap, since it flags the sloshing the post-hoc
tag misses, but as a remediation trigger the wasted prefix sinks it. A detector
that never pays a prefix, meaning one that runs from iteration 1 and switches the
preconditioner in place rather than restarting, is the direction that could win,
and it needs an in-loop hook the owner has ruled out of production for now.

## Compute spent

All SCFs on asus (Core Ultra 7 155H, 8 threads), except a handful of local
smoke and construction checks on the thinkpad at 2 threads.

| batch | SCFs | wall (asus, 8 thr) |
|---|---|---|
| Part 1 | 14 | 231 s |
| positive controls | 2 | 212 s |
| Part 2, k=6 | 8 | 166 s |
| Part 2, k=10 | 8 | 169 s |
| total | 32 | 778 s (~13 min) |

Local validation added roughly six small SCFs (Si and Ni smoke tests, a Cr
collapse check, coarse Al) at 2 threads, a few minutes each. Hematite was left
unrun. The Part 1 battery already exercised nspin=2 NC and USPP, the slab
inhomogeneous case, a coarse-mesh metal, and a hard PAW metal, so the marginal
value of a 10-atom AFM oxide did not justify the compute.
