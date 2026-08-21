# EFG magnitude round 2 — why gradwave's rutile TiO2 EFG sits at ~0.44x Elk

Read-only forensics + two numerical experiments, 2026-08-21. Elk run dir `asus:~/tio2_efg`
(STATE.OUT), Elk source `asus:~/github/elk-11.0.2/src`, gradwave states
`asus:~/tio2_states/*.pkl`, code under audit `src/gradwave/flapw/{efg,scf,lapw}.py`
(commit c76cb33). All probe scripts committed alongside this file. No src/ physics changed.

## VERDICT

**H-A is confirmed and pinned to a concrete bug; H-B is dead.**

The deficit lives in the **on-site l=2 valence density `rho_2M(r)`**, not the boundary/lattice
term. The cause is not a dropped product channel or Gaunt weight — every angular channel is
present. It is a **missing radial `1/r^2` weight**: `sphere_density_multipoles*` build `|psi|^2`
from the reduced radial functions `u = r*R` (from `_radial_u`) **without** the `/r^2` that
converts `|u|^2` to the true density `|R|^2`. So gradwave's `rho_2M(r) = r^2 * rho_true(r)`.
Every downstream consumer (`_valence_v`, `lx_sphere_poisson`, the Weinert L>0 moments in
`scf.py`, `nonspherical_potential`) treats it as the true density, so the on-site EFG integral
`(4πE2/5)∫ rho_2M / r dr` actually computes `∫ r·rho_true dr` instead of `∫ rho_true / r dr`.
That reweighting moves the integral's mass from the interior (where `1/r` peaks and the EFG is
built) to the surface, scaling **both sites down by ~⟨r^2⟩ ≈ 2-3x while preserving the angular
pattern** — exactly the observed signature (uniform ~0.44x, eta preserved, both sites, sign
correct, aug-lmax/fp-lmax/grid insensitive, nulls still zero because they are angular).

The contrast that isolates it (`_sphere_valence_density`, the SCF's *spherical* density, at
`scf.py:283`, already divides by `rr**2`): the multipole build is the one place the `/r^2` was
dropped.

## H-A EVIDENCE

### 1. Channel audit of `sphere_density_multipoles_bands` (efg.py:179) / `_multi` / `_bands`

`psi(r,Ω) = Σ_l Σ_radial rad(r)·(amp·Y_l)`, `rho_ang = Σ_n occ_n |psi|^2`, projected onto
`Y*_LM`. Because `psi` is the full sum, `|psi|^2` contains every product pair. Amplitudes come
from `_bands_amps` (efg-partner `scf.py:230`): APW `A_lm = (4π/√Ω) i^l Σ_G c_G a_l Y*_lm(k_G)`,
`B_lm` likewise for `u̇`, and an LO adds `cv·a` to the u-amp, `cv·b` to the u̇-amp, and `cv·cn`
as a third radial. Radials `us[l] = [u, u̇, (u2 for LO)]` from `_us_ext` → `_radial_u`.

| product channel entering \|ψ\|² | present? | note |
|---|---|---|
| u_l · u_l (same l) | YES | dominant O-2p→l=2 is the l=1×l=1 pair |
| u_l · u̇_l  and  u̇_l · u_l | YES | both radials carried per l |
| u̇_l · u̇_l | YES | |
| LO: u2·u, u2·u̇, u2·u2 | YES | LO enters u-amp, u̇-amp AND its own 3rd radial → all cross terms in \|ψ\|² |
| cross-l  l × l'  (s×d L=2, p×p, p×f, d×d…) | YES | `psi` sums all l ≤ aug-lmax; angular projection supplies the Gaunt coupling |
| angular / Gaunt weight | EXACT | Gauss-Legendre×uniform-φ grid exact for the band-limited integrand (matches the #337 contraction to 1e-16) |
| **radial `/r^2`  (\|u\|² → \|R\|² = true density)** | **MISSING** | `rad = u = r·R` used raw; no `/r^2`. ⇒ `rho_2M = r^2 · rho_true` |

Every angular/product channel is present. The only missing piece is the radial `r^2`.

### 2. Direct Elk-vs-gradwave on-site density comparison

Elk STATE.OUT reader (`elk_rho2m_reader.py`) validated end-to-end: reconstructing the EFG from
`vclmt`'s l=2 r²-coefficient reproduces `EFG.OUT` **exactly** on both sites (Ti1 Vzz 0.19903 a.u.
η 0.361; O1 tensor `[[-0.01283,-0.18383],[…],[+0.02566]]`) — so the reader, the real→complex
harmonic transform, and `_tensor_from_v` are all correct.

Elk's **on-site** EFG (interior l=2 Poisson of `rhomt` only, no lattice term), the object directly
comparable to gradwave's valence tensor:

| site | gradwave on-site (aug4, k222) | Elk on-site | Elk full | gradwave full |
|---|---|---|---|---|
| O  | **−3.13** eV/Å² (η 0.18) | **−10.27** (η 0.10) | −19.1 (η 0.74) | +8.97/−7.90 |
| Ti | +1.02 (η 0.27) | +20.73 (η 0.35) | +19.34 (η 0.36) | −0.79 |

gradwave's on-site O density EFG is **30% of Elk's** — the deficit is in `rho_2M` itself, before
any lattice term. It is **robust across aug-lmax** (on-site O: base −3.72, aug4 −3.13, aug5 −2.82)
and **across fp-lmax** (fp4 −3.13 ≈ fp6 −3.00) — so it is neither an angular-basis nor a
non-spherical-potential-order effect.

### 3. Radial ratio profile — the `1/r^2` signature

Rotation-invariant l=2 density power `P(r)=√Σ_M|rho_2M|²`, Elk/gradwave, aug4 (unit+mesh aligned):

| r/R | 0.23 | 0.35 | 0.60 | 0.74 |
|---|---|---|---|---|
| P_elk/P_gw | 29.4 | ~10 | ~3 | 1.56 |

Not flat — the deficit **grows toward the interior**, a `~1/r^2` shape. Small-r scaling
(measured): gradwave `rho_2M ~ r^~3-4`, Elk (true density) `~ r^2`. gradwave is one power of `r^2`
too steep — i.e. it carries `r^2 · rho_true`.

### 4. Decisive convention-independent check — the l=0 valence charge

The valence charge in the sphere is `√(4π)∫ rho_00 r² dr`. Using gradwave's `rho_00` (from the
same multipole build) as-is vs divided by `r^2`:

| site | as-is | `/r^2` | physical (Elk sphere valence) |
|---|---|---|---|
| O  | **1.46 e** | **5.10 e** | 5.67 e (Elk O sphere 7.67 − core 2) |
| Ti | 3.10 e | 7.65 e | — |

The `/r^2` version recovers the physical O valence charge; the as-is value is a factor ~4 too
small. This is independent of any EFG/harmonic convention and settles the direction: the multipole
density is `r^2 · rho_true`, and dividing by `r^2` restores the true density.

### 5. Effect of the fix (post-hoc `/r^2` on the captured density)

Dividing the captured `rho_2M` by `r^2` and recomputing the on-site EFG:

| site | current | `/r^2` | Elk on-site |
|---|---|---|---|
| Ti | +1.02 | **+18.41** (η 0.01) | +20.73 (η 0.35) |
| O  | −3.13 | **−15.68** (η 0.86) | −10.27 (η 0.10) |

Magnitudes jump ~3-6x into Elk's range (Ti nearly matches). The overshoot on O and the η shift are
expected artifacts of a *post-hoc* correction: the fullpot SCF that produced this density was
itself converged against the `r^2`-mis-scaled non-spherical potential (same buggy `rho_2M` feeds
`nonspherical_potential` and the Weinert L>0 moments), and the `∫·/r³`-weighted moment amplifies
the least-reliable deep-r region. The physically correct numbers require **re-converging the
fullpot SCF with the corrected build** — a follow-up. But the magnitude direction and scale are
unambiguous.

## H-B EVIDENCE — DEAD

The only grid-dependent part of the EFG is the interstitial boundary (lattice) term. Rebuilding
`v_grid`/`vbc_own` from the converged density on a band-limited-upsampled grid (`hb_grid_probe.py`,
scale 1.0/1.5/2.0×, i.e. nfft 28/42/56) and recomputing the **full** O EFG:

| grid scale | nfft | O full V_zz (eV/Å²) | η |
|---|---|---|---|
| 1.0 | 28 | +8.970 | 0.762 |
| 1.5 | 42 | +9.046 | 0.753 |
| 2.0 | 56 | +8.877 | 0.756 |

O moves **< 2%, non-monotonic (noise)**. Ti moves < 6%. Far below the ≥10% threshold. The
boundary/lattice term is grid-converged at the base grid; the 0.024 Å near-touch is **not**
under-resolved. H-B is not the cause; the fix is not a grid/Gmax policy.

(Consistency: the lattice term amplifies the on-site value by ~1.9x in gradwave — the same factor
as in Elk, −10.3→−19.1. The boundary is fine; only the on-site starting point is deficient.)

## PROPOSED FIX

In `efg.py`, build the multipole density from the true radial function `R = u/r`, not `u`:
in `sphere_density_multipoles_multi` / `_bands` (and the thin `sphere_density_multipoles`
wrapper), divide each radial in `us[l]` by `rr` before forming `psi` — equivalently, divide the
returned `rho_LM` by `rr**2`. This mirrors `_sphere_valence_density` (`scf.py:283`,
`rads[i]*rads[j]/rr**2`), which already does exactly this for the spherical density. Do it at the
radial level (divide `u` by `r` *before* squaring), not post-hoc on the moment, for numerical
cleanliness near the origin.

Because the same `rho_2M` dict feeds three consumers, the one fix propagates to all:
`_valence_v`/`efg_tensor*` (the EFG), the Weinert L>0 moments (`scf.py:742`), and
`nonspherical_potential` (`efg.py:59`). Consequences to re-validate after the fix:
1. The fullpot **SCF changes** (the non-spherical potential was mis-scaled), so re-converge the
   whole TiO2 matrix and re-check the r_nsph gate, kerker/damping tuning, and the "runaway fixed
   point" behaviour — those were characterised with the buggy density.
2. Nulls (cubic/Ne → 0) stay zero (angular, `r^2`-invariant) — necessary but not sufficient; they
   are exactly why this survived every gate.
3. **Add a valence-charge gate**: assert `√(4π)∫ rho_00 r² dr` equals the SCF sphere valence
   charge (O ≈ 5 e), which as-is reads 1.46 e — the single cheap test that would have caught this.
4. Expect both-site magnitudes to rise ~3x toward Elk (Ti into ~19-21, O on-site into ~-10);
   the exact converged O/Ti EFG and η are the follow-up's to report.

## Files
- `elk_state_probe.py` / `elk_rho2m_reader.py` — Elk STATE.OUT reader (validated vs EFG.OUT).
- `gradwave_rho2m_probe.py` — on-site tensors + Elk/gradwave power-ratio profile.
- `hb_grid_probe.py` — H-B grid A/B (dead).
- `r2_fix_test.py` — post-hoc `/r^2` on-site/full EFG.
- `l0_consistency.py` — the decisive l=0 valence-charge check.
