# Low-q damping of the transverse magnetization channels in the spinor SCF

Research note, branch `research/transverse-damping`. Executes recommendation 4
of the noncollinear-convergence campaign (`research/noncollinear-convergence`),
which decomposed the spinor residual floor into two independent parts. The
transverse magnetization channels are amplified by the mixing map at roughly
3x per iteration from machine zero to a ~1e-4 saturation, with the residual
power in the lowest |G| shells, and the longitudinal channel keeps a separate
near-Stoner floor of a few 1e-4. The campaign's proposed cure was a
reverse-Kerker damping of the transverse channels, suppress low-|G|, pass
high-|G|, leave G=0 free. This study built that cure, measured it, and found
the design is structurally mismatched to the instability. The amplified mode
is a rigid rotation of the whole magnetization density, it spans G=0 and its
finite-G cloud coherently, and it is amplified through the DIIS state
recombination rather than the mixing step. A variant that does work on the
z-seeded cases (damping the mixer's total update in a fixed lab frame) kills
the transverse floor 13x, but the longitudinal floor then binds everywhere,
SOC barely moves, and every damping form fails the canted-cell kill criterion.
The recommendation is no-go for a production implementation.

## Summary of the verdicts

- The brief's design (reverse-Kerker on the transverse residual via
  `mixer_hook`, G != 0 only, moment frame) leaves the measured amplification
  bit-for-bit unchanged through iteration 20 and worsens the floor.
- The amplified mode is a rigid rotation of m(r). Its residual grows in
  lockstep at G=0 and at finite G (5.8e-5 vs 8.2e-5 at the floor), so no
  |G|-selective filter can isolate it, and exempting G=0 while damping the
  cloud tears the rotation apart instead of suppressing it.
- The amplification flows through the Pulay state recombination, not the
  step. Damping the residual in the lab frame does nothing (floor 3.8e-4,
  equal to baseline), damping the total update in the same frame kills the
  transverse floor 13x (dm_x 7.9e-6). Same kernel, same frame, different
  insertion point.
- The frame matters more than the kernel. The instantaneous-moment frame
  chases the very noise it should damp (the frame tilts with the wandering
  G=0 moment and reclassifies the mode as longitudinal), so every
  moment-frame arm fails. A fixed lab frame works on z-seeded cases.
- Where the damping works, reverse-Kerker beats the flat null 8x (dm_x
  7.9e-6 vs 6.4e-5 at the same insertion point and frame), so the G
  dependence is real.
- None of it lowers the total floor. The longitudinal near-Stoner floor
  (2e-4 to 5e-4) binds in every socfree arm, the best composite (3.5e-4) is
  the undamped johnson best arm, and the champion transverse fix does not
  stack with johnson (dm_x back at 1.9e-4 under it).
- SOC is untouched at 1.5e-3 to 2.0e-3 across every treatment.
- The canted 2-atom kill criterion fails for every form. The moment-frame
  hook drives the seeded 90-degree pair through 171.8 degrees, near
  anti-alignment, before a late recovery, and the lab-frame update damping
  slows alignment roughly 10x and lands the final axis off the bisector.

## The two insertion points

Everything below uses one damping kernel and two places to apply it.

The kernel. D(G) = G^2 / (G^2 + q0^2) on the transverse component of the
m-channels, D -> 0 as G -> 0, D -> 1 at high G, the mirror of charge Kerker.
The transverse component is defined per G against a unit vector n_hat, either
the current total moment direction ("moment" frame, from the G=0 coefficients
of vin) or lab z ("lab" frame). G=0 is never damped except in the explicitly
labeled all-G arms. The flat null replaces D(G) with a constant alpha_perp at
G != 0. The q0 scan brackets the observed soft sector, the density sphere
runs to |G|max ~24 1/Ang, the campaign put 0.5 to 0.8 of the transverse
residual power in the bottom two of twelve shells (up to ~4 1/Ang), so q0 =
2, 4, 8 1/Ang places the D = 0.5 crossover below, at, and above that edge.

Insertion point A, the residual hook (`damping.py`). The driver exposes
`scf_noncollinear(..., mixer_hook=...)`, called with the raw packed (vin,
vout) before `mixer.step`. The hook rewrites vout in place so the residual the
mixer sees is damped. The convergence gate reads the raw residual computed
BEFORE the hook fires (`scf/noncollinear.py:762`, gate at `:787`, hook at
`:808`), so the damping cannot fake convergence, and the probe records the
raw residual too, so every trace measures the true map.

Insertion point B, the step wrap (`damping2.py`). The Pulay update has two
parts, the extrapolated state sum_i c_i vin_i and the preconditioned step.
The hook only reaches the step. The DIIS coefficients are set by the dominant
channels (charge and longitudinal m, orders of magnitude above the transverse
noise), so the recombination amplifies transverse content it was never asked
to contract. The wrap monkeypatches `_build_nc_mixer` from the experiment
side and wraps `mixer.step`, damping the transverse part of the TOTAL update
u = mixed - vin. At a fixed point the update is zero, so the fixed point is
unchanged. No src file is edited by either variant.

## Systems and settings

The campaign's cells and tolerances exactly. fcc Ni (scalar PD_Ni_PBE for
SOC-free, Ni ONCV FR for SOC), 40 Ry, 4x4x4, gaussian 0.1 eV, etol 1e-6,
rhotol 1e-5, diago_tol 1e-9, max_iter 80, mixing_alpha 0.5, no symmetry, full
mesh, z-seed 0.6. The canted case is the 2-atom bcc Fe conventional cell
(ONCV FR or scalar), 35 Ry, 3x3x3, moments seeded 90 degrees apart (+z and
+x). "best" arms use the campaign's best recipe (johnson + quadratic schedule
+ mag_mixing_alpha 0.3). All runs on asus, 5 to 8 threads. Floors are the
mean raw residual over the last ten iterations in the driver's own
volume-scaled norm. The probe additionally splits each transverse component
into its G=0 and finite-G parts and, on the 2-atom cell, integrates per-atom
moments over a nearest-atom Voronoi partition to track the pair angle.

## Step 1. The brief's design does nothing

SOC-free Ni, the cleanest signal (m_x, m_y start at machine zero, exact
collinear symmetry). Baseline amplification measured at 4.1x per iteration
over iterations 8 to 18 (dm_x 5.2e-11 at 10, 5.5e-8 at 15, 9.4e-5 at 20).
The table lists the hook arms, moment frame unless noted.

| run | dm floor | dm_x floor | dm_z floor |
|---|---|---|---|
| nisf_baseline | 3.8e-04 | 1.0e-04 | 3.5e-04 |
| nisf_kerker_qlo (q0=2) | 6.9e-04 | 3.2e-04 | 5.6e-04 |
| nisf_kerker_qmid (q0=4) | 4.1e-04 | 1.3e-04 | 3.7e-04 |
| nisf_kerker_qhi (q0=8) | 6.1e-04 | 2.0e-04 | 5.7e-04 |
| nisf_kerker_qmid_lab | 3.8e-04 | 1.2e-04 | 3.5e-04 |
| nisf_flat_a03 | 2.5e-03 | 3.0e-04 | 2.3e-03 |
| nisf_flat_a01 | 6.5e-04 | 5.6e-05 | 6.5e-04 |

No arm improves on baseline and several hurt. The damning detail is in the
traces, the growth phase (1e-14 to 1e-4 by iteration ~20) is numerically
identical to baseline in every hook arm, damped or not, moment frame or lab.
The residual the mixer steps along is not the path the instability takes.

## Step 2. Where the mode actually lives

An instrumented baseline splits each transverse residual component into its
G=0 part and its finite-G remainder per iteration.

| iter | dm_x(G=0) | dm_x(G!=0) |
|---|---|---|
| 10 | 3.5e-11 | 3.9e-11 |
| 16 | 2.7e-07 | 1.8e-07 |
| 20 | 8.3e-05 | 4.4e-05 |
| 45 | 5.8e-05 | 8.2e-05 |

The two parts grow in lockstep at the same rate and land at comparable
floors. The unstable mode is not a finite-q magnon packet sitting in a low
shell, it is the rigid rotation of the entire magnetization density, whose
Cartesian-component signature has weight at every G where |m|(G) does,
predominantly but not exclusively at the bottom. Two consequences follow.
First, no |G|-selective kernel can isolate this mode from the physics.
Second, the brief's G=0 exemption splits the mode down the middle, the
uniform head stays free while the cloud is held, which distorts the rotation
rather than damping it and explains why several hook arms are worse than
baseline.

## Step 3. The recombination is the amplifier, and the frame must not move

The table lists the step-wrap arms on the same system, damping the total update.

| run | frame / form | dm floor | dm_x floor | dm_z floor |
|---|---|---|---|---|
| nisf2_qlo | moment, kerker q0=2 | 7.9e-04 | 2.4e-04 | 7.2e-04 |
| nisf2_qmid | moment, kerker q0=4 | 5.9e-04 | 2.8e-04 | 5.0e-04 |
| nisf2_qhi | moment, kerker q0=8 | 4.7e-04 | 1.3e-04 | 4.3e-04 |
| nisf2_qmid_lab | lab, kerker q0=4 | 5.0e-04 | 7.9e-06 | 4.9e-04 |
| nisf2_flat03 | moment, flat 0.3 | 4.1e-04 | 4.0e-05 | 3.9e-04 |
| nisf2_flat01 | moment, flat 0.1 | 6.3e-04 | 7.3e-05 | 6.2e-04 |
| nisf2_flat03_lab | lab, flat 0.3 | 8.2e-04 | 6.4e-05 | 8.1e-04 |

Three findings, in order of importance.

The insertion point decides everything. Same kernel, same lab frame, the
residual hook floors dm_x at 1.2e-4 (baseline level) and the step wrap floors
it at 7.9e-6, a 13x kill. The instability is fed by the DIIS recombination of
past states, which only the step wrap reaches. This is measured twice over,
since the hook-lab arm's failure and the wrap-lab arm's success differ only
in the insertion point.

The frame must hold still. Every moment-frame arm fails at both insertion
points. The frame vector tilts with the accumulated G=0 transverse noise, so
the damping chases the mode it is supposed to suppress and reclassifies part
of it as longitudinal each iteration. The lab frame, which for a z-seeded run
is the physically correct axis, does not chase.

Where the damping works, the reverse-Kerker structure earns its place. In
the lab frame at the working insertion point, D(G) with q0=4 beats the flat
null by 8x (7.9e-6 vs 6.4e-5). The q0 scan in the moment frame is
non-monotone and uniformly bad, so the scan only discriminates in the frame
that works.

## Step 4. The all-G brake, and why none of it moves the total

The rigid-rotation picture suggests damping the transverse residual at ALL G
including G=0, a moment-rotation brake at the hook insertion point.

| run | dm floor | dm_x floor | dm_z floor |
|---|---|---|---|
| nisf_allg_a03 (flat 0.3, all G) | 6.7e-04 | 9.9e-06 | 6.7e-04 |
| nisf_allg_a01 (flat 0.1, all G) | 8.7e-04 | 5.5e-11 | 8.7e-04 |
| nisfbest_allg_a03 (+ johnson best arm) | 5.5e-04 | 1.7e-05 | 5.4e-04 |

The brake works on its own terms, alpha 0.1 holds the transverse channel at
5.5e-11 for the whole run, a soft pin without the pin's fixed-point bias. But
the longitudinal floor worsens in every all-G arm, and that is the general
pattern. Across every socfree arm in the study, the total dm floor never goes
below the 3.5e-04 of the undamped johnson best arm, because the longitudinal
near-Stoner floor (the campaign's part b) binds as soon as the transverse
part is silenced, and the transverse treatments consistently push noise into
the longitudinal channel. The composite that was hoped for, transverse fix
stacked on the johnson best arm, does not materialize either,
nisfbest2_qmid_lab floors dm_x at 1.9e-04, the champion's 7.9e-6 does not
survive the switch from pulay to johnson (johnson builds an inverse-Jacobian
estimate from the raw pairs and the wrapped update interacts with it
differently, not traced further).

The bound is sharp and worth stating. With the transverse channel held at
5.5e-11, the run still floors at 8.7e-04 total. The remaining gap to rhotol
1e-5 is a factor 35 to 80 and belongs entirely to the longitudinal channel,
which this fix by construction cannot touch.

## Step 5. SOC

| run | dm floor | dm_x floor |
|---|---|---|
| nis_baseline | 2.0e-03 | 1.1e-03 |
| nis_kerker_qmid (hook) | 1.7e-03 | 7.8e-04 |
| nis_kerker_qlo (hook) | 1.6e-03 | 8.3e-04 |
| nis_flat_a03 (hook) | 1.6e-03 | 6.5e-04 |
| nis_allg_a03 (hook, all G) | 1.6e-03 | 6.2e-04 |
| nis2_qmid_lab (wrap) | 1.8e-03 | 8.6e-04 |
| nisbest_baseline | 1.7e-03 | 9.5e-04 |
| nisbest_kerker_qmid (hook) | 1.5e-03 | 7.5e-04 |
| nisbest_allg_a03 (hook, all G) | 1.8e-03 | 8.3e-04 |
| nisbest2_qmid_lab (wrap) | 1.7e-03 | 8.1e-04 |

The SOC floor sits at 1.5e-03 to 2.0e-03 under every treatment, a 20 to 25
percent trim at best, and no arm converges. Unlike the pin, the damping does
leave the SOC transverse physics able to settle, every SOC arm lands the
campaign's consensus fixed point (F within -4189.80228 to -4189.80231, moment
0.5790 to 0.5791 on the seed axis), so nothing is being zeroed that the
ground state needs. But the floor is seeded at 1e-3 by the first
diagonalization's spin mixing and saturates there regardless of what the
mixer does about it. The campaign's reading stands, under SOC the transverse
sector carries real physics at a scale the damping cannot and should not
remove.

## Step 6. The canted kill criterion, failed by every form

Two Fe moments seeded 90 degrees apart must still align. The baseline pair
angle closes 88.7, 83.3, 61.0, 51.0, 26.8, 19.1, 14.4, 0.6 degrees over the
first eight iterations (SOC-free and SOC identical to three digits).

| run | pair angle, iters 1-8 | late floor | final moment |
|---|---|---|---|
| fe2_baseline | 88.7 -> 0.6 monotone | --- | [2.84, 0, 2.90] |
| fe2_kerker_qmid (hook, moment) | 88.7 -> 171.8, anti | 0.1 | [3.78, 0, 1.70] |
| fe2_kerker_qlo (hook, moment) | 88.7 -> 109 -> 71, wander | ~70 at it 8 | [2.90, 0, 2.90] |
| fe2b_qmid_lab (wrap, lab) | 88.7 -> 113 -> 81, wander | 0.9 | [-2.60, 0, 3.31] |
| fe2_allg_a03 (hook, all G) | 88.7 -> 136.9 -> 93, opens | 0.37 | [2.98, 0, 2.93] |
| fe2soc_baseline | 88.7 -> 0.8 monotone | --- | [2.84, 0, 2.90] |
| fe2soc_kerker_qmid (hook, moment) | 88.7 -> 171.8, anti | 0.14 | [3.50, 0, 2.23] |
| fe2soc_allg_a03 (hook, all G) | 88.7 -> 137.2 -> 94, opens | 0.25 | [2.94, 0, 2.90] |
| fe2bsoc_qmid_lab (wrap, lab) | 88.7 -> 113 -> 81, wander | 2.54 | [-2.67, 0, 3.06] |

The moment-frame hook arms do not merely slow the alignment, they invert it,
the pair is driven to 171.8 degrees, near anti-alignment, by iteration 7
before a late recovery, and the final total moment lands well off the
baseline bisector. The mechanism is the split-mode problem from step 2 in
its physical form. Interatomic alignment is a STAGGERED rotation, it lives
at finite G (the first star of the 2-atom cell, |G| ~2.2 1/Ang, gets D =
0.23 at q0=4), so the G != 0 damping suppresses exactly the physical motion
while the free G=0 head keeps moving, and the state is steered onto a
different path. The lab-frame wrap arm wanders for tens of iterations and
lands its axis elsewhere too ([-2.60, 0, 3.31] against [2.84, 0, 2.90]).
Damped residual floors are also 2x to 6x worse than baseline on this cell in
every damped arm. The all-G brake fares no better than the G=0-free forms,
it opens the pair to 137 degrees before a late recovery, SOC-free and SOC
alike. The worst arm is the SOC lab-frame wrap, which never completes the
alignment (pair floor 2.5 degrees), floors the residual at 1.6e-1, and lands
8 meV above the consensus canted fixed point, a different basin. The G=0-free
design does not protect physical transverse evolution, because physical
transverse evolution is not confined to G=0.

## Fixed-point oracle

Every Ni arm in the study, damped or not, lands the campaign's consensus.
SOC-free, F = -4500.06270 within 6e-6 eV across all arms, moment 0.1154 to
0.1161 on z. SOC, F = -4189.80228 to -4189.80231, moment 0.5790 to 0.5791 on
z. The damping never bends the fixed point, exactly as a residual or update
transformation that vanishes at self-consistency should. The canted cell's
final states differ between arms because no arm converges there and the
damped trajectories genuinely visit different configurations, which is the
kill-criterion failure, not an oracle failure.

## Recommendation, no-go, and what would be needed instead

No production implementation of reverse-Kerker transverse damping. The
grounds, in decreasing order of weight.

1. The canted kill criterion fails for every parameterization tested,
   including the G=0-free designs the campaign hoped would protect physical
   rotation. Any workflow that starts moments off their final axis (canted
   seeds, spin spirals, unknown easy axis) is actively endangered.
2. The one real success, dm_x 7.9e-6 on z-seeded SOC-free Ni via lab-frame
   update damping, does not lower the total residual floor, the longitudinal
   near-Stoner channel binds at 2e-4 to 9e-4 everywhere and is measurably
   aggravated by most transverse treatments. The gap from the best floor to
   rhotol 1e-5 is a factor 35 and belongs to the longitudinal channel, which
   this fix by construction cannot touch.
3. It does not stack with the johnson best arm and it does not help SOC.

The campaign's recommendation 1 (gate magnetic spinor runs at rhotol about
1e-3 and document the floor) remains the operative fix. Its recommendation 4
should be amended by this study's mechanism, the transverse floor is a
rigid-rotation mode amplified by the DIIS state recombination, so any future
mixer-side cure must (a) damp the update, not the residual, (b) work in a
frame that does not follow the instantaneous moment (the seed axis is the
natural choice), and (c) either damp G=0 together with the cloud or leave
the whole mode alone. If a niche use case ever justifies the z-seeded win
(collinear-limit MAE workflows on a known axis), the src hook it needs is a
per-iteration transform of the mixed output, mixed' = vin + T(mixed - vin)
with T a per-G linear map on the m blocks, i.e. a `step_transform` callable
on the mixer (or a locally-rotated per-block `step_scale` generalization),
applied inside `PulayMixer.step` after extrapolation. The existing
`mixer_hook` cannot express it, and this study shows the residual-side hook
is the wrong insertion point for any recombination-driven instability.

## Flags

- Floors on these systems are noisy limit-cycle means, repeat runs vary by
  up to 2x (nisf_baseline 3.8e-4 was reproduced exactly under an identical
  rerun, but dm_z floors across arms scatter 2.1e-4 to 8.7e-4). Differences
  under ~2x in the tables should be read as noise, the 13x and 8x claims are
  well outside it.
- The johnson non-stacking result (nisfbest2_qmid_lab) is a single arm, the
  interaction between the wrapped update and johnson's inverse-Jacobian
  estimate was not traced further once the canted criterion had already
  failed the design.
- The step-wrap moment-frame arms were run before the frame mechanism was
  understood, they are retained in the tables as the measurement that
  exposed it.
- The canted fe2 arms never converge in either direction, so their final
  energies compare limit cycles, not fixed points. The 8 meV basin miss of
  fe2bsoc_qmid_lab is still meaningful, its trajectory demonstrably left the
  configuration every undamped run settles into.

## Files

- `damping.py` TransverseDampingHook, insertion point A (mixer_hook, residual)
- `damping2.py` StepWrapPatch, insertion point B (monkeypatched _build_nc_mixer, total update)
- `probe.py` NCConvergenceProbe + G=0/finite-G split + per-atom Voronoi moments
- `systems.py` campaign cells, fixture pseudos
- `run.py` hook-form matrix (groups ni_socfree / ni_soc / ni_soc_best / canted / allg)
- `run2.py` step-wrap matrix (groups nisf2 / nis2 / best2 / canted2 / lab2)
- `analyze.py` floor/growth/alignment tables
- `results/nixos_a..e/` summary.jsonl + per-run traces (raw residuals) + damp diagnostics
