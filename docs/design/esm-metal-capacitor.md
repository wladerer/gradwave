# Design: ESM metal / capacitor sub-modes (constant-potential electrochemistry)

Status: **design note / deferred.** The vacuum/vacuum ESM (`boundary: open_z`) is
implemented and validated — potential, energy, forces, and in-plane stress, for
both the norm-conserving and USPP/PAW paths (`core/energies/esm.py`, PR that added
this file). This note scopes the remaining Phase-1 item from
`docs/design/dtn-3d-engine.md`: the **metal** and **capacitor** boundary
conditions for electrochemistry. It is deferred because — unlike the other Phase-1
increments, which were self-contained — the headline use case (constant-potential
electrochemistry) needs a grand-canonical SCF, which is a separate, larger feature.

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

### Level 1 — metal electrostatics at fixed charge (moderate)

Add `esm_mode: vacuum | metal | capacitor` and, for `capacitor`, an applied bias.
Mechanism: replace the open z Green's function with the Dirichlet one per `G∥`, and
add the bias ramp to the `G∥=0` channel. Unlocks: a neutral slab between grounded
plates, and field-effect / Stark response to an applied bias.

**The one real subtlety.** The current vacuum implementation gets its
discretization-matched, ion-width-independent ΔE from a *linear vs circular
convolution of the same translation-invariant kernel* (`esm_delta_potential`). The
Dirichlet Green's function is **not** translation-invariant (it depends on `z` and
`z'` separately, not `z−z'`), so that trick does not carry over directly. Options,
cheapest-first:
- Solve the Dirichlet BVP per `G∥` with a tridiagonal (finite-difference) `(∂²_z −
  g²)` operator — O(Nz) per mode — and form the open-minus-periodic correction
  against a **matching** tridiagonal periodic operator (circulant), so the
  short-range field still cancels. This keeps the "correction to periodic" framing
  and its ion-width independence.
- Or the analytic sinh/parabolic kernels applied via an O(Nz) forward/backward
  recurrence (the Thomas algorithm is exactly this for the tridiagonal form).

Avoid the dense `Nz×Nz` kernel per mode (the memory blow-up the vacuum recursion
was written to dodge). Forces/stress then follow for free by the same autograd
route already used for vacuum (the bias adds an explicit, differentiable term).

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
