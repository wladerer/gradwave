# USPP/PAW spin-channel mixing: why Ni PAW stagnates in the magnetization channel

Research note, branch `research/uspp-spin-channel`. Follows the ao-density-seed
study (archive branch `research/ao-density-seed`, ideas.md entry "Atomic-orbital
seeding, the density channel"), which localized the fcc Ni PAW stagnation to the
mixer's magnetization channel and left a mixer-level fix open.

The working hypothesis handed to this study was that the norm-conserving (NC)
path carries a Stoner spin preconditioner and the USPP/PAW path does not, and
that the asymmetry explains NC Ni converging in about 14 iterations while PAW Ni
stagnates at 120. The code says the opposite. This note reads the two paths
precisely, characterizes the stagnation with a magnetization-channel probe, and
measures the fix the code already contains.

## The hypothesis is backwards: USPP has the Stoner preconditioner, NC does not

The Stoner magnetization-channel preconditioner lives in
`src/gradwave/scf/spin_precond.py`. It builds the Newton model operator
`(I - χ₀^diag K_mm)⁻¹` on the m-channel, with χ₀^diag the rank-(Fermi bands)
occupation-response and K_mm the local XC spin kernel, inverted exactly by
Woodbury (`StonerSpinPrecond`, lines 45-66). Two builders assemble it, the
collinear `build_stoner_precond` (lines 69-157) and the non-collinear
`build_stoner_precond_nc` (lines 186-276).

Where each SCF driver wires it in main:

- `scf/uspp_loop.py` (collinear USPP/PAW). Takes `spin_precond: bool = False`
  (line 1207) and applies `build_stoner_precond` each iteration at lines
  1660-1688, preconditioning the residual m-channel `out[ng:2*ng]` before the
  mix. The USPP/PAW path HAS the preconditioner. It is off by default.
- `scf/noncollinear.py` (spinor/SOC). Applies `build_stoner_precond_nc` at
  lines 789-790. The non-collinear path HAS it.
- `scf/loop.py` (plain NC collinear, the `ni_nc` anchor). No `spin_precond` flag,
  no Stoner call. A grep for `spin_precond`/`stoner`/`extra_precond` returns
  nothing in the module body. The NC path does NOT have the preconditioner.

The perf branch `perf/magnetic-mixing` (commit df156ed, unmerged) was ADDING the
preconditioner to the NC path, mirroring the USPP wiring, not the other way
round. So NC Ni converges in about 14 iterations with no spin preconditioner at
all, and PAW Ni stagnates while the preconditioner it needs sits behind a
default-off flag.

## The real NC-vs-USPP asymmetry is the mixing scheme, not the preconditioner

Two other differences separate the paths, and both are documented in the source.

Default mixing scheme for nspin=2. The NC path defaults to johnson
(`_resolve_mixing_scheme`, `scf/loop.py:705-716`, "johnson for collinear-spin
(nspin==2), pulay otherwise", Fe FM 15→14 vs pulay). The USPP/PAW path defaults
to pulay (`_resolve_uspp_mixing_scheme`, `scf/uspp_loop.py:925-942`, "pulay if
nspin == 2 else johnson"). The USPP docstring names the tension directly, namely
johnson near the Stoner boundary is a win on fcc Ni (27→18) but a large loss on
robust bcc Fe (29→93) once it discards the becsum step-damping that pulay leans
on, so pulay is the safe magnetic default.

The composite mixing vector. The USPP/PAW mixer works on
`[ρ_tot, ρ_mag, becsum_up, becsum_dn]` (`scf/layout.py`, `MixLayout`), carrying
the on-site augmentation-charge (becsum) mode the NC vector has no analogue for.
The becsum mode is stiff even in a gapped insulator, which is why johnson beats
pulay across non-magnetic PAW but flips sign on magnetic PAW.

Kerker on the m-channel. The Kerker mask covers only the ρ_tot block
(`MixLayout`, `scf/layout.py:37-40`, ones over `ng`, zeros over the m-channel and
becsum). The magnetization channel is already excluded from Kerker screening.
Candidate (c) from the plan, excluding m from Kerker, is already in place.

So of the three fixes the plan proposed, two are already present. Porting the NC
Stoner preconditioner (a) is moot because USPP already has it. Excluding m from
Kerker (c) is already the layout. The live levers are the default-off
`spin_precond` flag, the pulay-vs-johnson scheme choice, and per-channel damping
(b).

## Stagnation characterization

The probe (`probe.py`) wraps the mixer's `step`, reads the raw residual
`ρ_out - ρ_in` in G-space on the density sphere, and bins the total and
magnetization blocks onto twelve linear |G|-shells, the same scheme the flight
recorder uses for the total density (`scf/recorder.py`). No src changes, the SCF
runs under `no_grad` regardless.

The characterization run is Ni PAW at start_mag 0.30 (the "comfortable" value
that the seeding study found stagnates like the rest), pulay with spin_precond
off, 120-iteration cap. The trace is in `results_probe_off.jsonl`.

The floor is in the magnetization channel, and it is stuck rather than decaying.
The total-density residual norm falls below 1e-4 by iteration 81 and settles near
3.5e-5. The magnetization residual norm does not clear 1e-5 at all. It floors at
a mean of 1.4e-4 over the last twenty iterations, oscillating between 7e-5 and
3e-4, a factor of four swing with no downward trend. At the floor the
magnetization residual norm is 3.9 times the total, so the composite residual
that rhotol 1e-5 gates on is carried by the magnetization channel.

The surviving mode is not long-wavelength. On iteration 1 the magnetization
residual sits almost entirely in the two lowest |G|-shells (low-shell fraction
0.88). By the floor it has migrated up, with the two lowest shells holding 0.003
and the shells above the sixth holding 0.23. The stuck magnetization mode is
mid-to-high |G|, which is consistent with Kerker (a long-wavelength charge
operator, and already masked off the m-channel) being irrelevant to it. The
charge channel converges normally the whole time.

Both known fixes act on this frozen mode. With spin_precond=on the same pulay run
at start_mag 0.02 converges in 56 iterations rather than stalling at the cap, so
the Stoner Newton step unfreezes the magnetization residual. Switching to johnson
converges every start_mag in 11 to 16 iterations. The measured trajectories are
in the results table below.

## Prototype results

Settings mirror the seeding study exactly, namely gaussian width 0.1 eV, etol
1e-6, rhotol 1e-5, mixing_alpha 0.3, max_iter 120, start_mag 0.02/0.05/0.10/0.30.
Systems are Ni PAW (`Ni.pbe-spn-kjpaw`, 50 Ry, 4×4×4, ecutrho 400 Ry), Fe PAW
(`Fe.pbe-spn-kjpaw`, 50 Ry, 6×6×6, the must-not-regress control), and Ni NC
(`PD_Ni_PBE`, 45 Ry, 6×6×6, the sanity anchor). The fixed-point oracle is the
converged F and moment matching the known FM branch.

Fixed-point references (converged FM branch, from the seeding study's
d-localized rescue): Ni PAW F = -5838.39832294044 eV, moment 0.59375 μB. Fe PAW
F = -4479.687818703 eV, moment 2.22222 μB. The Ni NM branch the default seed
falls onto at 0.10 sits +77 meV up (F = -5838.32137).

Baselines from the seeding study, default uniform-split SAD seed, which on the
USPP/PAW path is pulay with spin_precond off:
- Ni PAW default: 0 of 4 converged. All four hit the 120-iteration cap. At 0.02,
  0.05, 0.30 F sits within ~4e-5 eV of the FM branch but the residual never
  clears rhotol 1e-5. At 0.10 it lands on the NM branch, +77 meV.
- Ni PAW d-localized seed rescue: 3 of 4 converged.
- Fe PAW default: 4 of 4 converged (18 to 26 iterations). The d-localized seed
  regresses it (collapses 0.02 to NM, +660 meV), so Fe is the control any
  mixer-level change must not break.

Ni PAW, the target. n_iter, converged, and the branch reached:

| start_mag | pulay off (default) | pulay + spin_precond | johnson off | johnson + spin_precond |
|---|---|---|---|---|
| 0.02 | 120, no, FM floor | 56, yes, FM | 12, yes, FM | 13, yes, FM |
| 0.05 | 120, no, FM floor | 120, no, FM floor | 11, yes, FM | 11, yes, FM |
| 0.10 | 120, no, NM +77 meV | --- | 13, yes, FM | 15, yes, FM |
| 0.30 | 120, no, FM floor | --- | 16, yes, FM | 16, yes, FM |

The pulay-off column is the seeding study's default-seed baseline, 0 of 4.
pulay + spin_precond is not a reliable rescue, it converges 0.02 in 56 iterations
but stagnates 0.05 at the cap. johnson converges every start_mag in 11 to 16
iterations and selects the FM branch at 0.10 where the default lands on NM. The
spin_precond column on johnson is neutral, 11 to 15 iterations, same fixed point.
Every converged F matches the FM reference -5838.39832294044 eV to about 1e-8 eV
and the moment is 0.59375 μB, so the oracle holds.

Fe PAW, the must-not-regress control. The seeding study's default (pulay off)
converges 4 of 4 in 18 to 26 iterations. n_iter under johnson, all converged to
the FM reference:

| start_mag | pulay off (seeding) | johnson off | johnson + spin_precond |
|---|---|---|---|
| 0.02 | 26 | 18 | 18 |
| 0.05 | 20 | 14 | 14 |
| 0.10 | 18 | 13 | 14 |
| 0.30 | 21 | 16 | 16 |

johnson does not regress Fe. Every Fe johnson cell converges to the FM reference
-4479.687818703 eV, moment 2.22222 μB, in fewer iterations than the pulay
default at the same start_mag. This contradicts the current
`_resolve_uspp_mixing_scheme` docstring, which cites Fe 29→93 as the reason
nspin=2 PAW stays on pulay. At the seeding study's settings on current main that
regression does not appear.

Ni NC anchor, johnson default (nspin=2), no spin preconditioner. At start_mag
0.05 it converges to the FM branch in 13 iterations (moment 0.592 μB), the true
anchor value, and the NC path does not stall. At 0.02 it converges in 23
iterations but to the nonmagnetic branch (moment 3e-7 μB, moment-collapse
flagged), a seed-side branch-selection issue separate from the mixer stagnation
this note is about. The NC path converges fast where PAW pulay freezes.

## Go / no-go

Go, and the fix is not the one the plan expected. The honest bar was 4 of 4 from
the default seed. johnson mixing clears it, converging Ni PAW at every start_mag
in 11 to 16 iterations and selecting the FM branch at 0.10 where the default seed
lands on NM. The d-localized seed rescue reached 3 of 4. johnson reaches 4 of 4
with no change to the seed, and it holds the Fe control, converging every Fe cell
faster than the pulay default.

The Stoner spin_precond flag, the fix the plan expected to port or promote, is
not the answer. It already exists on the USPP/PAW path, and turning it on rescues
Ni 0.02 (56 iterations) but stagnates Ni 0.05 at the cap. The preconditioner
unfreezes the magnetization mode when it dominates cleanly and misses when the
becsum coupling carries part of the stall, which is why it is partial. johnson's
normalized multisecant update handles the same mode without the becsum
step-damping that pulay leans on and that freezes the mid-to-high-|G|
magnetization residual (the characterization above).

A production fix would change one line, namely `_resolve_uspp_mixing_scheme`
(`scf/uspp_loop.py:940-942`) returning "johnson" for nspin==2 rather than
"pulay". That is the whole change, the johnson mixer and the m-channel-excluded
Kerker mask are already in place.

The one caveat that keeps this a finding and not a merged default. The current
docstring justifies the pulay default with a measured Fe 29→93 johnson
regression. This study does not reproduce that regression at the seeding study's
settings on current main, where Fe johnson is strictly faster than pulay. Either
intervening changes (the trust-region mixer reset, becsum handling) removed it or
it was measured under different settings. Before flipping the global default the
owner should re-measure the Fe johnson regression that the docstring cites, on
the robust ferromagnets it was seen on, at the width and alpha it was seen at. If
it holds anywhere, ship johnson as the documented opt-in rescue
(`mixing_scheme: johnson`) for marginal-Stoner PAW rather than as the global
nspin=2 default. If it does not reproduce, the default flip is safe and the
docstring's caution is stale.

## Files

- `run.py`, the start_mag by precond by scheme matrix driver.
- `probe.py`, the magnetization-channel residual shell-decomposition monkeypatch.
- `probe_run.py`, one Ni PAW case with the probe installed, per-iteration dump.
- `results_johnson.jsonl`, the Ni and Fe PAW johnson matrix (off and on).
- `results_pulay_spinprecond.jsonl`, Ni PAW pulay with spin_precond on.
- `results_probe_off.jsonl`, the per-iteration stagnation trace (pulay off, 0.30).
- `results_anchor.jsonl`, the Ni NC anchor.
