# Design: ESM metal / capacitor sub-modes (constant-potential electrochemistry)

Status: **Level 1 implemented (NC + USPP/PAW); Level 2 (constant-µ SCF) working on
the NC path** — converges, N floats with µ, reduces to the canonical result at the
neutral µ; absolute-reference calibration is the remaining refinement. The
vacuum/vacuum ESM
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

### Level 2 — constant-potential (grand-canonical) SCF — **WORKING (NC)**

Exposed as `scf(target_mu=µ)` / `inputs.SCFParams.target_mu` (requires
`boundary="open_z_metal"` + a smearing). Validated on a free-electron metal (Na):
the SCF converges, the electron count **N floats monotonically with µ** (positive
differential capacitance / DOS>0), and **at the neutral µ it recovers the canonical
N exactly** (Na₂: `N(µ₀)=18.000`). What made it work, beyond the charged
electrostatics below:
- `common.constant_mu_occupations` (fixed µ → `N=Σw·g·f`, N floats; no bisection);
- the density mixer must **float the charge** — `check_g0` is disabled under
  `target_mu` so the G∥=0 (charge) residual is mixed rather than asserted zero;
- `SCFResult.n_electrons` carries the floating N; the grand potential is
  `Ω = free_energy − fermi·n_electrons`.

**Net-charge reference calibration — DONE.** The capacitor correction was built on
the vacuum ESM pieces, which are *invalid for a net-charged cell* (the open Poisson
of a charged sheet is ill-defined). So for a charged cell the effective potential
deviated from the plate-referenced `v_cap` by O(1 eV) in z-structure — a real error
(it persists for a smooth charged blob with no ions, so it is not the ion
self-energy). Fixed by referencing the **net** charge to the *spectral* periodic
(matching the KS energy) via a neutral/smooth-charge split: `ρ_tot = ρ_n (neutral)
+ ρ_q (net charge as an in-plane-uniform broad Gaussian)`; the neutral part keeps
the β-safe matched correction, the net-charge terms use `v_cap − v_periodic`
directly (valid for charge, grid-safe since ρ_q is smooth). The net-charge
reference now tracks `v_cap` to ~1e-3 (was ~3.4), `δE/δρ` stays consistent, and
neutral cells are unchanged. The remaining raw `v_eff − v_cap` (~0.08) is the ion
self-energy, correctly *excluded* (Ewald handles it — the calibrated energy differs
from the naive `½∫ρ v_cap` by exactly that self-energy).

**Still external:** mapping the absolute µ scale to SHE (the computational hydrogen
electrode convention) — as in every constant-potential DFT code; the natural
internal reference is the potential of zero charge (the neutral-cell Fermi level).

#### Original scoping notes

Electrochemistry means holding the electrode **potential** fixed while the slab
exchanges charge with the reservoir — a grand-canonical ensemble at fixed `μ`, with
a *floating* electron count `N(μ)`. The **electrostatics prerequisite is DONE**:
`hartree_potential_capacitor` (and the capacitor `esm_energy`/`esm_potential`) now
handle a **net-charged cell** — the `G∥=0` channel solves the Dirichlet BVP directly
with the parabolic Green's function `G_D0(z,z')=z_<(L−z_>)/L`, so the plates carry
the induced counter-charge (validated: analytic triangular potential, induced plate
charges sum to −σ, and `esm_potential=δΔE/δρ` holds for a +0.4 e cell → a fixed-µ SCF
on it is variational).

**Remaining (SCF-side) work — the invasive part.** It touches `scf/loop.py`'s
occupation/charge machinery:
- **Occupation.** Add a constant-µ path beside `shared_fermi_occupations`: given µ,
  `occ = g·f((ε−µ)/σ)` and `N = Σ w·occ` (N floats), instead of bisecting µ from a
  fixed N (`core/occupations.find_fermi`).
- **Floating charge.** `n_electrons` (today `charges.sum()`, neutral) becomes the
  floating `N(µ)`; the density integrates to `N`, the cell is charged (`q = ΣZ − N`),
  and the ESM must be `open_z_metal` (the plates hold `−q`; vacuum ESM can't source
  charge). The periodic `G=0` term is the usual charged-cell jellium background.
- **Grand potential.** Report/converge on `Ω = E − µN`; forces/stress pick up the
  `−µ dN/dR` term (zero at self-consistency by the same Hellmann–Feynman argument,
  since the ESM potential is already `δE/δρ`).
- **Config.** `esm_target_mu` (the electrode potential / applied U vs a reference).

This is "constant-potential DFT" (cf. Bonnet–Marzari, Sundararaman ESM-RISM). It
needs SCF-level validation (a constant-µ run converging to the right `N(µ)`, and a
`N`-vs-µ capacitance curve), so it is its own focused build on top of the now-done
charged electrostatics.

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
