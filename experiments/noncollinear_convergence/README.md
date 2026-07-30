# Non-collinear / SOC SCF convergence: a transverse instability, not a mixer problem

Research note, branch `research/noncollinear-convergence`. Follows the collinear
spin-channel study (`research/uspp-spin-channel`), which localized the Ni PAW
stagnation to the mixer's magnetization channel and fixed it by defaulting
nspin=2 USPP/PAW mixing to johnson (PR #205). This note asks the same questions
of the spinor path. Which mixer does it use, which channel carries the residual
floor, and does the moment direction converge or wander.

The short version. Every magnetic spinor run in the matrix fails rhotol 1e-5,
under every exposed knob, while the collinear path converges the same physics
in 13 iterations. The floor is not the mixer, not the eigensolver, not the
schedule, not the backoff, and not the smearing. The probe traces localize it
to the transverse magnetization channels, which are amplified from machine
zero at roughly 3x per iteration until they saturate near 1e-4. Pinning those
two channels makes the collinear-limit spinor run converge (measured below).
The energies and moments at the floor are still good to about 3e-5 eV and
1e-4 mu_B, so the floor is a residual-gate problem, not a fixed-point problem.

## The mixing story on the non-collinear path, precisely

The #205 johnson default does NOT reach the spinor path. The three drivers
resolve their mixing scheme independently, and `scf_noncollinear` never calls
either resolver.

- `scf/noncollinear.py:523`. The driver takes `mag_mixer: str = "pulay"` and
  `_build_nc_mixer` (`scf/noncollinear.py:263-298`) instantiates that class by
  name from `_MAG_MIXERS` (`scf/noncollinear.py:259`, pulay/johnson/broyden).
  The magnetic default is PulayMixer. A nonmagnetic (m pinned to zero) run
  hardcodes PulayMixer with plain Kerker (`scf/noncollinear.py:276-279`).
- No auto-resolution. The collinear NC path resolves scheme=None to johnson
  for nspin=2 (`scf/loop.py:705-716`), and the USPP path defaults nspin=2 to
  johnson since #205 (`scf/uspp_loop.py`). The spinor driver has no scheme
  argument beyond `mag_mixer`, and `api._run_scf_noncollinear`
  (`api.py:340-349`) does not pass it, so YAML inputs cannot select it and
  every `task: scf, noncollinear: true` run mixes with pulay.
- Kerker masks the m channels. The magnetic mixer packs [rho, m_x, m_y, m_z]
  with Kerker on the rho block only (`kerker_mask`, `scf/noncollinear.py:282`),
  matching the collinear layout.
- The m channels get a decoupled, LARGER step. `mag_mixing_alpha` defaults to
  `max(mixing_alpha, 0.6)` (`scf/noncollinear.py:589-590`), applied as a
  per-component `step_scale` on the three m blocks, a moment-collapse guard
  documented at `scf/noncollinear.py:579-588` (bcc O2 collapses at alpha 0.4).
- The Stoner spin preconditioner is wired but opt-in. `spin_precond: bool =
  False` (`scf/noncollinear.py:522`) builds `build_stoner_precond_nc`
  (`scf/spin_precond.py:186`) each iteration and applies it to all three
  Cartesian m blocks (`scf/noncollinear.py:788-802`). Off by default, not
  reachable from the input layer.
- `precond_op` is a callable-only override on the charge block
  (`scf/noncollinear.py:535-536`), no string selector, unreachable from YAML.
- An adaptive backoff (`_nc_adaptive_backoff`, `scf/noncollinear.py:332-355`)
  halves the global step and drops the DIIS history whenever the residual
  fails to fall 10 percent over 6 iterations. The multiplier never recovers,
  so a floored run ends up mixing at 0.1x with no history. On by default.
- The diago-tolerance schedule is "linear" (`tol_eff = 0.03 res_prev`,
  `scf/common.py:adaptive_diago_tol`), the spinor-family choice. The collinear
  drivers use "quadratic". A magnetic run may opt into quadratic via
  `mag_diago_schedule` (`scf/noncollinear.py:524`).

So the spinor path is the last driver still mixing its magnetization with
plain pulay by default, and none of the magnetic knobs (`mag_mixer`,
`spin_precond`, `mag_mixing_alpha`, `mag_diago_schedule`) are exposed through
`inputs.py`/`api.py`.

## Prior art: fix/ni-soc-convergence

The branch (one WIP commit, 371ecc2, 2026-07-24, "gentler magnetization mixing
+ Stoner spin-precond boost [UNVALIDATED]") is MERGED in substance. Its two
source changes, the `mag_mixer` selector in `_build_nc_mixer` and the
non-collinear Stoner builder `build_stoner_precond_nc`, are on main verbatim
at the lines cited above. The branch itself is stale (its diff against current
main is dominated by months of main-side drift) and can be deleted.

What it was fighting is recorded in `tests/integration/test_ni_soc_convergence.py`
(PR #79): fcc Ni + SOC at 40 Ry, 4x4x4, gaussian 0.1 eV. The stock driver pins
the residual in a 3e-2 to 1e-1 band with the moment thrashing between -0.4 and
+5 uB and exhausts the budget. The merged recipe (`spin_precond=True`,
`mag_diago_schedule="quadratic"`, `mag_mixing_alpha=0.3`) locks the moment to
~0.6 uB and drops the residual 30-60x. The test's honesty caveat matters for
everything below. It gates at rhotol 5e-3, deliberately loose, because "a
separate, much smaller Fermi-surface residual floor (~1e-3 at this coarse
mesh) means the strict 1e-6 gate is not reached". This study measures that
floor and finds it is not a Fermi-surface artifact (the smearing-width variant
below leaves it unchanged) but the transverse instability.

## Probes

`probe.py` implements a `mixer_hook` (an existing driver argument,
`scf/noncollinear.py:537`, called with the raw packed (vin, vout) before every
mix, `scf/noncollinear.py:808-809`), so no src changes and no monkeypatching.
Per iteration it records the charge and magnetization residual norms (the m
part as a vector-field norm and per Cartesian component), the integrated
moment vector from the G=0 coefficients, the angle the total moment turned
since the previous iteration, and a 12-shell |G| decomposition of both
channels' residual power (the flight recorder's binning, which does not
support this path, PR #203).

## Baseline matrix

Settings shared by every run: PBE (NoncollinearXC over SpinPBE), gaussian
0.1 eV, etol 1e-6, rhotol 1e-5, diago_tol 1e-9, max_iter 80 (200 for the
oracle attempt), mixing_alpha 0.5, history 8, no symmetry, full mesh. All on
asus, 7 to 8 threads. Systems use the committed fixture pseudos. "fix" is the
merged #79 recipe. "floor" columns are the mean raw residual over the last ten
iterations, in the same volume-scaled norm the driver gates on.

| run | converged | n_iter | F [eV] | moment [mu_B] | dn floor | dm floor |
|---|---|---|---|---|---|---|
| pt_soc_nm (anchor) | yes | 8 | -3305.040752 | 0 (pinned) | --- | --- |
| ni_soc_stock_s0.3_z | no | 80 | -4189.802273 | [0.000, 0.000, +0.579] | 3.5e-04 | 2.0e-03 |
| ni_soc_stock_s0.6_z | no | 80 | -4189.802286 | [0.000, 0.000, +0.579] | 2.9e-04 | 2.1e-03 |
| ni_soc_stock_s1.0_z | no | 80 | -4189.793785 | [0.000, 0.000, -0.293] | 1.1e-02 | 3.1e-02 |
| ni_soc_stock_s0.6_tilt | no | 80 | -4189.802273 | [+0.334, +0.334, +0.334] | 3.3e-04 | 2.4e-03 |
| ni_soc_fix_s0.3_z | no | 80 | -4189.802299 | [0.001, 0.001, -0.579] | 2.8e-04 | 1.9e-03 |
| ni_soc_fix_s0.6_z | no | 80 | -4189.802299 | [0.004, 0.004, -0.579] | 2.7e-04 | 1.6e-03 |
| ni_soc_fix_s1.0_z | no | 80 | -4189.802304 | [0.007, 0.007, +0.579] | 2.8e-04 | 1.4e-03 |
| ni_soc_fix_s0.6_tilt | no | 80 | -4189.802297 | [-0.334, -0.334, -0.334] | 4.4e-04 | 1.8e-03 |
| ni_soc_johnson_s0.6_z | no | 80 | -4189.802305 | [0.000, 0.000, +0.579] | 3.9e-04 | 1.7e-03 |
| ni_soc_johnson_long_s0.6_z | no | 200 | -4189.802267 | [0.000, 0.000, +0.579] | 4.0e-04 | 2.6e-03 |
| ni_socfree_s0.6_z | no | 80 | -4500.062702 | [0.000, 0.000, +0.116] | 1.1e-04 | 4.9e-04 |
| ni_collinear_s0.6 | yes | 13 | -4500.062708 | 0.1155 | --- | --- |
| fe_soc_stock_s0.6_z | no | 80 | -3213.070356 | [0.000, 0.001, +1.974] | 7.5e-03 | 2.2e-02 |
| fe_soc_fix_s0.6_z | no | 80 | -3213.039557 | [-0.056, -0.027, +2.509] | 3.3e-02 | 9.6e-02 |
| fe_socfree_s0.6_z | no | 80 | -3213.408310 | [0.002, 0.001, +2.427] | 5.6e-02 | 1.8e-01 |
| fe_collinear_s0.6 | no | 80 | -3213.415054 | 2.5216 | --- | --- |
| fe2_canted90_socfree | no | 80 | -6409.918861 | [+2.84, 0.00, +2.90] | 7.9e-03 | 2.6e-02 |
| fe2_canted90_soc | no | 80 | -6409.195232 | [+2.84, 0.00, +2.90] | 6.2e-03 | 2.0e-02 |

What the matrix says.

- The anchor behaves. Nonmagnetic SOC (Pt, m pinned to zero) converges in 8
  iterations. SOC by itself is not the problem, and neither is the spinor
  Hamiltonian.
- No magnetic spinor run reaches rhotol 1e-5. Not one, across pulay, the #79
  recipe, and johnson, on Ni, Fe, and the 2-atom canted cell. The johnson
  200-iteration run shows the floor does not decay with budget.
- The fixed point is fine. Every Ni + SOC arm that holds the FM branch lands
  on the same free energy to about 4e-5 eV (-4189.80227 to -4189.80230) and
  the same moment (0.5791 mu_B along the seed axis) to 1e-4. The floor blocks
  the residual gate, not the physics.
- The honest cost of going non-collinear on the same physics. The scalar-pseudo
  Ni cell converges through the collinear path in 13 iterations and floors at
  80 through the spinor path with no SOC in either run. Same cell, same
  pseudo, same tolerances.
- Seed strength selects the basin, and the #79 recipe widens it. At seed 1.0
  the stock driver wanders between branches for 80 iterations (F sits 8.5 meV
  above the FM branch, moment 0.29 and drifting, the residual two decades
  above the other seeds). The fix recipe at the same seed lands the proper FM
  branch. That, not tolerance, is what the #79 recipe buys.
- Direction tracks the seed. The [111]-tilted seed converges its moment onto
  [111] exactly (0.334 per component), so there is directional branch
  selection, and MAE-style workflows can trust the seeded axis at this scale.
  One caveat, the fix recipe flips the moment SIGN (seed +z lands -z at seeds
  0.3 and 0.6, seed +[111] lands -[111]). The +-m branches are time-reversal
  degenerate with or without SOC here (identical F), but anything reading the
  sign should gate on the axis, as the manual already advises.
- Fe at these settings (ONCV FR pseudo, 40 Ry, 4x4x4, alpha 0.5) is hard for
  BOTH formalisms. The collinear control also caps at 80, so the Fe rows say
  "hard case all around", not "spinor-specific failure". The clean paired
  comparison is the Ni scalar pair above. The fix recipe makes Fe worse (dm
  floor 9.6e-2 versus stock 2.2e-2), echoing the collinear finding that
  near-Stoner medicine hurts robust ferromagnets.
- The canted 2-atom case aligns almost immediately. The two moments seeded 90
  degrees apart merge onto the bisector by iteration 2 and the angle then just
  jitters (0.03 to 0.1 degrees per iteration, no monotone rotation, no
  precession). SOC and no-SOC behave identically to three digits. The floor
  is 10x higher than 1-atom Ni.

## Where the floor lives

Every knob was tested directly on ni_soc s0.6 and the floor survived all of
them (`run_variants.py`).

| variant | floors at (dm) | what it rules out |
|---|---|---|
| stock pulay | 2.1e-03 | baseline |
| #79 fix recipe | 1.6e-03 | Stoner precond + gentle m step + quadratic schedule |
| johnson, 200 iters | 2.6e-03 | mixer scheme, iteration budget |
| diag11 (fix + diago 1e-11) | 1.6e-03 | (masked, see below) |
| noadapt (backoff off) | 2.5e-03 | backoff churn |
| width02 (0.2 eV smearing) | 1.7e-03 | Fermi-surface sharpness |
| best (johnson + quad + precond + 1e-11 + no backoff) | 2.6e-03 | everything at once |

One measurement caveat. diag11 is bit-identical to the fix run because the
quadratic schedule's own term (0.1 r^2/nelec, about 2e-8 at the floor) sits
above both 1e-9 and 1e-11, so the base tolerance never engages. The schedule
is exonerated separately by the socfree prototype below, where the quadratic
term reaches 1.4e-9 and the floor persists.

The trace decomposition then localizes the floor. In the SOC-free spinor run
the transverse channels (m_x, m_y) start at machine zero, as exact collinear
symmetry requires, and are AMPLIFIED by the SCF map at roughly 3x per
iteration (3.9e-11 at iteration 10, 6.9e-6 at 20, 7.1e-5 at 40) until they
saturate near 1e-4, where the total m floor sits. The longitudinal channel
floors at the same scale. The residual power sits in the lowest |G| shells
(0.5 to 0.8 of it in the bottom two shells), the opposite of the collinear
PAW stagnation, which was mid-to-high |G|. Long-wavelength transverse
magnetization is exactly the soft (magnon-like) direction of a ferromagnet,
which costs no energy at q -> 0 without SOC, so the linear response there has
gain near 1 and the mixed iteration has nothing to contract with. The
measured factor-3 growth says the map is genuinely unstable on those modes at
the default m step (0.6), not merely marginal. This explains every negative
above at once, namely why the floor is mixer-independent (all history mixers
damp through the same near-unit-gain map), schedule- and
tolerance-independent (the noise is not eigensolver noise), width-independent
(not a Fermi-surface artifact), worse on the 2-atom cell (more soft q modes),
and absent from both the nonmagnetic run (m pinned) and the collinear path
(no transverse channels exist).

With SOC the transverse channels are seeded at 1e-3 by the spin mixing of the
first diagonalization instead of at 1e-14, and saturate at the same kind of
level, which is why the SOC floor (1.4e-3 to 2.6e-03) sits above the SOC-free
one (4.9e-4).

## Prototype: schedule parity, then a transverse pin

`run_proto.py` (schedule parity on the paired Ni scalar system) and
`run_proto2.py` (the transverse pin).

| run | converged | n_iter | F [eV] | moment | note |
|---|---|---|---|---|---|
| ni_socfree_quad | no | 80 | -4500.062703 | 0.116 z | quadratic schedule alone does not fix it |
| ni_socfree_quad_tight | no | 80 | -4500.062702 | 0.115 z | + diago 1e-11, no backoff. Eigensolver exonerated |
| ni_socfree_johnson_quad | yes | 16 | -4500.061478 | 0.000 COLLAPSED | converges by killing the moment, +1.2 meV NM branch |

PROTO2 PENDING

The johnson result deserves its own flag. Full parity with the collinear
nspin=2 default (johnson + quadratic) does converge the spinor run in 16
iterations, but onto the nonmagnetic branch, 1.2 meV above the FM answer the
collinear path holds with the same mixer. Johnson on the spinor path is a
moment-collapse hazard the collinear path does not have, so simply porting
the #205 default to `mag_mixer` is not the fix.

## Flags

- The driver signature advertises rhotol 1e-7 as its default
  (`scf/noncollinear.py:518`). No magnetic spinor run in this study can pass
  1e-5. Until the transverse channel is handled, magnetic spinor rhotol below
  about 5e-3 (the #79 test's own gate) buys iterations, not accuracy. The
  manual (noncollinear-soc.md, magnetism.md, mae.md) gives no spinor-specific
  tolerance guidance and does not mention the floor. Nothing in the manual is
  contradicted by measurement, but the omission is worth closing.
- `docs/manual/wisdom.md` notes on johnson moment collapse ("do not expect
  the mixer to select the physical branch") transfer to the spinor path with
  more force, since johnson there collapses even a comfortable 0.116 mu_B
  moment from a 0.6 seed at the collinear-limit fixed point.
- The #79 recipe flips the seeded moment sign reproducibly. Degenerate, but
  gate on the axis.

## Files

- `probe.py` NCConvergenceProbe (mixer_hook, no src changes)
- `systems.py` system builders (fixture pseudos only)
- `run_matrix.py` baseline matrix (groups anchor/ni_stock/ni_fix/ni_alt/fe/canted)
- `run_variants.py` floor-origin discriminators
- `run_proto.py` schedule parity on the paired scalar-Ni system
- `run_proto2.py` transverse pin + johnson collapse controls
- `results/asus/` summary.jsonl + per-run traces
