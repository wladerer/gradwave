# Design: ESM metal / capacitor sub-modes (constant-potential electrochemistry)

Status: **Level 1 implemented (NC); Level 2 deferred.** The vacuum/vacuum ESM
(`boundary: open_z`) is implemented and validated — potential, energy, forces, and
in-plane stress, for both the norm-conserving and USPP/PAW paths. **Level 1 below
(the metal/capacitor `metal_metal` boundary conditions at fixed charge, grounded +
applied bias) is now implemented for the norm-conserving path** as
`boundary: open_z_metal` + `esm_bias` (`core/energies/esm.py`
`hartree_potential_capacitor`, mode-aware `esm_energy`/`esm_potential`). Level 2
(constant-potential grand-canonical SCF) remains deferred — it is a separate,
larger feature (a floating electron count at fixed µ), not a boundary-condition
tweak.

## The three ESM boundary conditions (Otani–Sugino)

Per in-plane `G∥`, the 1D Poisson `(∂²_z − |G∥|²) v = −4πe² ρ` is solved on `[0, L]`
with:

| mode | z boundary | Green's function (screened, `g=|G∥|>0`) | use |
|---|---|---|---|
| `vacuum` (done) | `v → 0` at ±∞ | open: `e^{−g|z−z'|}/(2g)` | isolated slab |
| `vacuum/metal` | vacuum one side, `v=0` at a metal plane | half-space image | surface on an electrode |
| `metal/metal` | `v(0)=0`, `v(L)=V_bias` | finite-interval Dirichlet: `sinh(g z_<)·sinh(g(L−z_>)) / (g·sinh gL)` | biased capacitor / electrochemistry |

`G∥=0` mirrors this: vacuum → the open `|z−z'|` kernel (done); metal → the parabolic
finite-interval kernel `z_<(L−z_>)/L` plus, for a bias, the linear ramp
`v_bias(z) = V·(z−z_L)/(z_R−z_L)`.

## Two levels of scope

### Level 1 — metal electrostatics at fixed charge — **DONE (NC)**

Exposed as `boundary: open_z_metal` (+ `esm_bias` [V]): metal Dirichlet planes at
both z-box edges, grounded, with an optional applied bias. Unlocks a neutral slab
between grounded plates and the field-effect / Stark response to an applied bias.

**How the discretization subtlety was avoided.** The Dirichlet Green's function is
not translation-invariant, so the vacuum linear-vs-circular convolution trick does
not apply directly. Instead the metal potential is built from the *already-validated*
open potential plus a homogeneous image solution:

    v_cap(G∥, z) = v_open(G∥, z) − [v_open(G∥,0)·sinh(g(L−z)) + v_open(G∥,L)·sinh(gz)] / sinh(gL)

(linear image for `G∥=0`), computed with a numerically stable `sinh`-ratio (all
exponents ≤ 0, no overflow), plus the bias ramp `bias·z/L`. This inherits the
vacuum kernel's ion-width independence and needs no `Nz×Nz` kernel. The
charge-induced correction is the quadratic `½∫ρ ΔV` (grounded); the applied bias is
the linear `∫ρ·(bias·z/L)`, so `esm_potential = δΔE/δρ` and the force `δΔE/δR` stay
consistent for both (all FD-validated). Validated: `v=0`/`v=bias` at the planes to
1e-15, Poisson in the interior, uniform field `bias/L`, and an end-to-end NC SCF.

**Not yet:** capacitor **stress** (needs `esm_energy_strained`'s capacitor variant)
and the **USPP/PAW** capacitor path (`esm_bias` not threaded through `scf_uspp`;
`api` rejects `open_z_metal` there for now).

### Level 2 — constant-potential (grand-canonical) SCF (large — the real feature)

Electrochemistry means holding the electrode **potential** fixed while the slab
exchanges charge with the reservoir — a grand-canonical ensemble at fixed `μ`, with
a *floating* electron count `N(μ)`. That is not an electrostatics change; it is an
SCF change:
- the neutrality/`E_F` bisection becomes a `μ`-controlled charge equation (`N` is
  solved for, not fixed);
- the compensating counter-charge lives on the electrodes (the capacitor plates),
  not a uniform background;
- the energy becomes the grand potential `Ω = E − μN`, and forces/stress pick up
  the `−μ dN/dR` term.

This composes with Level 1's biased-capacitor electrostatics but is independently
sized (it touches `scf/loop.py`'s occupation/neutrality machinery and the energy
definition). It is the "constant-potential DFT" capability (cf. Bonnet–Marzari,
Sundararaman ESM-RISM) and deserves its own focused build.

## Recommendation

Build **Level 1** (metal/capacitor electrostatics) as the next ESM increment — it
is self-contained, reuses the vacuum autograd plumbing, and delivers grounded-
electrode and field-effect slabs. Gate **Level 2** (grand-canonical) on whether the
target science is genuinely constant-potential electrochemistry; it is a distinct
feature, not a boundary-condition tweak, and should not be smuggled into the
electrostatics increment.

## Validation ladder (when built)

1. Level 1: a charged sheet between grounded plates reproduces the analytic image
   series; the bias produces a uniform field `V/L` in vacuum (vs the analytic
   capacitor field), and the correction stays ion-width-independent.
2. Level 1 forces/stress vs finite difference (as for vacuum).
3. Level 2: `N(μ)` monotone and the grand potential stationary; a symmetric slab at
   zero bias reduces to the fixed-`N` neutral result.
