# Differentiable Hubbard U

A GGA underbinds the localized $d$ and $f$ electrons of transition-metal
compounds. The DFT+U correction fixes this with a single parameter per manifold.
The parameter itself is not guessed. gradwave computes it from linear
response and exposes its exact energy derivative, so U is a determinable,
differentiable quantity rather than a fitted input.

## Theory

gradwave implements the rotationally-invariant Dudarev functional,[[19]](bibliography.md#dudarev) which
penalizes fractional on-site occupation,

$$ E_U = \sum_{I,\sigma} \frac{U_\text{eff}}{2} \operatorname{Tr}\!\left[ n^{I\sigma}\left(1 - n^{I\sigma}\right) \right], \qquad U_\text{eff} = U - J, $$

where $n^{I\sigma}_{mm'} = \sum_{kv} f_{kv\sigma} \langle \phi^I_m | \psi_{kv} \rangle
\langle \psi_{kv} | \phi^I_{m'} \rangle$ is the occupation matrix of the $(l)$
manifold on atom $I$, built from the pseudo-atomic orbitals $\phi^I_m$. The
potential $V_U$ is a nonlocal projector with a density-dependent D-matrix, so it
uses the same projector contraction as the Kleinman-Bylander nonlocal term. For USPP/PAW the overlaps carry the $S$-metric, $\langle \phi | S |
\psi \rangle$, matching QE's `U_projection_type='atomic'`.

**U from linear response.** Following Cococcioni and de Gironcoli,[[20]](bibliography.md#cococcioni) a
rigid probe $\alpha_J \sum_m |\phi^J_m\rangle\langle\phi^J_m|$ is added to one
correlated site and the on-site occupation response measured. The interacting
response $\chi_{IJ} = \mathrm{d}N_I / \mathrm{d}\alpha_J$ (density re-converged)
and the bare response $\chi^0_{IJ}$ (one non-self-consistent diagonalization at the
frozen potential) give

$$ U = \left( {\chi^0}^{-1} - \chi^{-1} \right)_{II}, $$

the $\chi^{-1}$ subtraction removing the delocalized rigid-shift part.

**Exact dE/dU.** At self-consistency the energy is stationary in the density, so
by the Hellmann-Feynman theorem the total derivative equals the partial,

$$ \frac{\mathrm{d}E}{\mathrm{d}U} = \sum_{I,\sigma} \frac{1}{2} \operatorname{Tr}\!\left[ n^{I\sigma}\left(1 - n^{I\sigma}\right) \right], $$

evaluated at the converged occupations with no finite differences and no SCF
re-run. This is the gradient a loop that *learns* U would backpropagate.

## Set up a +U calculation

DFT+U is switched on by passing a `HubbardManifold` (a species index, an
angular-momentum $l$, and the $U$, $J$ values in eV) to the SCF driver. It is
independent of `start_mag`, which only seeds the initial magnetization.

```python
from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.spin import SpinPBE
from gradwave.scf.loop import scf

res = scf(system, SpinPBE(), nspin=2, start_mag=[+0.5, -0.5, 0, 0],
          smearing="gaussian", width=0.05,
          hubbard=[HubbardManifold(species=0, l=2, u=5.0, j=0.0)])   # U=5 eV on Ni 3d

res.energies.hubbard      # E_U [eV]
res.hub_occ               # per-spin, per-site occupation matrices n^{Iσ}
```

The manifold applies to *every* atom of that species. For USPP/PAW use `scf_uspp`
and `from gradwave.scf.uspp_hubbard import HubbardManifold` (same fields, $S$-dressed
projectors).

## From an input file

A `hubbard` block turns +U on from a `gradwave` input, one manifold per
correlated species. It threads through `scf`, `relax`, `eos`, and `elastic` on
the norm-conserving path (forces and stress included), and the SCF occupation
correction also runs on USPP/PAW. `gradwave init hubbard` writes a starter.

```yaml
xc: pbe
nspin: 2
start_mag: {Ni: 0.5}
smearing: {type: gaussian, width: 0.05}
hubbard:
  - {species: Ni, l: 2, u: 5.0}      # U = 5 eV on the Ni 3d shell (l=2)
  # - {species: O, l: 1, u: 0.0, j: 0.0}   # a second manifold, per species
```

`species` is an element symbol (the correction hits every atom of it), `l` the
shell (2 = d, 3 = f), `u`/`j` the Hubbard and Hund parameters in eV. +U always
runs on the full spatial Brillouin zone: an IBZ-folded mesh under-counts the
occupation matrix, so `symmetry` is forced off whenever a `hubbard` block is
present (time reversal is kept, since $n$ at $-k$ is $n^*$ and $|n_{mm'}|^2$ is
invariant). Setting `u: 0` everywhere is inert, reproducing the plain-functional
run to the last bit.

+U also runs on the norm-conserving **noncollinear/spin-orbit** spinor path
(`noncollinear: true`, with or without `nonmagnetic: true` — fully-relativistic
pseudos included, since the +U term is orthogonal to the SOC nonlocal term).
There the per-orbital occupation "matrix" generalizes to a 2×2 spin block
$N^{I}_{(\sigma m),(\sigma' m')} = \sum_{kv} w_{kv} \langle\phi^I_m|\psi_{kv}^\sigma\rangle
\langle\psi_{kv}^{\sigma'}|\phi^I_{m'}\rangle$ (stack the composite $(\sigma, m)$
index into one $2n_\text{orb}$-dimensional matrix per site), and the Dudarev
trace $E_U = \sum_I \frac{U_\text{eff}}{2}\operatorname{Tr}[N(1-N)]$ carries over
unchanged — it reduces exactly to the plain collinear formula above when the
spinor is purely up- or down-polarized with no spin canting.
`core.hubbard.occupation_matrices_noncollinear`/`hubbard_dmatrix_noncollinear`
build $N$/$D$; `scf.noncollinear.scf_noncollinear` takes the same `hubbard=`
kwarg as the collinear `scf`. The +U force and stress are wired on this path
too (`postscf.forces.hubbard_force_noncollinear`,
`postscf.stress`'s fully-relativistic branch), the noncollinear generalization
of the collinear `hubbard_force`/Hubbard strain term below.

The noncollinear **USPP/PAW** SCF (`scf.uspp_noncollinear.scf_uspp_noncollinear`)
does not have +U wired at all yet (a `hubbard=` argument there raises
`NotImplementedError`); use the norm-conserving noncollinear path above, or the
collinear USPP/PAW path for a +U ultrasoft/PAW run.

Combining `hubbard` with a hybrid `xc` is rejected at load: the hybrid Fock SCF
has no +U hook.

## Large U on metallic systems

A large U on a manifold that sits at the Fermi level can stop the SCF from
converging. The +U potential uses the previous iteration's occupation matrix, so
a U of several eV shifts the correlated levels by about U/2, far above a metallic
smearing width. The occupations at the Fermi level then flip between full and
empty from one iteration to the next, and the density residual oscillates without
settling. The flight recorder records a nonzero band-reordering count in most
iterations, the signature of this flip-flop.

Two convergence aids handle this, and both leave the default runs bit-for-bit
unchanged. Occupation-matrix damping mixes the occupation matrix across
iterations, $n = (1-\beta)\,n_\text{prev} + \beta\,n_\text{new}$, in place of the
raw one-step-lagged matrix. A $\beta$ around 0.3 contracts the flip-flop toward
its fixed point. The U-ramp raises $U_\text{eff}$ linearly from a fraction to its
full value over the first $N$ iterations and then holds, so the correlated levels
never jump past the smearing window in a single step. Convergence is blocked
until the ramp completes, so the reported final energy is always at the full U.

Both are exposed on the input `hubbard` block written as a mapping, with
`occ_mix` ($\beta$ in $(0, 1]$, default 1.0) and `u_ramp_iters` (default 0, off).

```yaml
hubbard:
  manifolds:
    - {species: Pt, l: 2, u: 9.0}   # +U on the Pt 5d manifold
  occ_mix: 0.3
  u_ramp_iters: 15
```

The same knobs are `hub_occ_mix` and `hub_u_ramp_iters` on `scf` and `scf_uspp`,
and on the `GradWave` calculator. They apply on both collinear paths, the
norm-conserving `scf` and the USPP/PAW `scf_uspp`. The noncollinear spinor path
does not take them yet, so a large-U metallic SOC +U run has neither aid.

## Forces and stress with +U

The +U energy is a differentiable function of the positions and the cell through
the atomic-orbital projectors, so its force and stress contributions come from
the same autograd pass the plain terms use. The stress adds the Hubbard strain
term when the manifolds are supplied. The occupation matrices do not carry the
strain derivative on their own, so the same `manifolds` list passed to the SCF
must be passed to `stress`.

```python
from gradwave.postscf.stress import stress
sig = stress(res, SpinPBE(), manifolds=[HubbardManifold(species=0, l=2, u=5.0)])
```

A +U result handed to `stress` without its manifolds is rejected rather than
silently dropping the term. `stress` also carries the fully-relativistic
(spin-orbit) +U strain term, so the same call works on a `scf_noncollinear`
result with a fully-relativistic pseudopotential.

From an input file (or the `GradWave` ASE calculator) the +U force and stress are
folded in automatically: `relax`/`eos`/`elastic` with a `hubbard` block descend
on the +U-corrected forces and stress. This runs on both the norm-conserving and
the USPP/PAW path through the calculator — USPP/PAW +U forces (`forces_uspp`)
and stress (`stress_uspp`, the strained $S$-dressed occupation term) both exist,
so USPP/PAW +U relaxation, EOS, and elastic-constant runs work end to end, not
just single-point `task: scf`.

## Determine U, and its gradient

`linear_response_u` runs the finite-difference probe (one base plus two perturbed
SCFs for $\chi$, cheap one-shot solves for $\chi^0$). `linear_response_u_autodiff`
gets the same number from a single ground-state SCF using conduction-projected
Sternheimer response, with the Hartree-XC screening kernel taken as an autograd
Hessian-vector product of $E_\text{Hxc}$ (so any twice-differentiable, including
learnable, functional works with no explicitly written $f_\text{xc}$).

```python
from gradwave.postscf.hubbard_u import (
    linear_response_u, linear_response_u_autodiff, energy_derivative_u)

# U on the Ni 3d manifold (l=2, species 0), perturbing site 0
out = linear_response_u(system, SpinPBE(), l=2, species=0, site=0,
                        alpha=0.1, scf_kwargs=scf_kw)
print(out["U_eV"], out["chi0"], out["chi"])          # ~6.45 eV; chi0 < chi < 0

# exact dE/dU at a fixed +U point
print(energy_derivative_u(res, [HubbardManifold(species=0, l=2, u=5.0)]))
```

## Validation

- **NiO, U from response vs QE `hp.x`.** The DFPT reference U on Ni 3d is
  6.4308 eV. `linear_response_u` gives 6.4493 eV (0.3%) and the autodiff variant
  matches $\chi^0 = -0.2136$, $\chi = -0.0873$ from one SCF. Both localize
  correctly, $\chi^0 < \chi < 0$.
- **NiO, exact dE/dU.** The Hellmann-Feynman value matches a central difference of
  full SCF re-runs to $10^{-4}$.
- **Si (PAW), U = 2 eV on 3p.** $E_U$ agrees with QE to 0.008 meV, the total to
  0.31 meV/atom, forces to ~$10^{-5}$ eV/Å, and $U=0$ reproduces the plain PAW SCF
  to $10^{-10}$, so the machinery is inert when off.
- **Ni (PAW), U = 3 eV on 3d.** Spin-polarized $E_U$ to 0.004 meV, moment within
  0.02 $\mu_B$ of QE.

## Gotchas

- **The pseudopotential must carry the manifold's atomic orbital** (`PP_PSWFC`).
  PseudoDojo and psl (kjpaw/rrkjus) sets have it. SG15/ONCV generally do not, and
  give total DOS but no +U. For PAW the *raw* pseudo-orbital amplitudes are used
  (a PAW pseudo-orbital's plain norm is deliberately not one, since the $S$ overlap
  supplies the rest), and renormalizing them is a ~100 meV error.
- `linear_response_u_autodiff` is insulators-only (it projects onto the conduction
  space). The finite-difference `linear_response_u` handles metals.
- The magnetization channel of the response can have a screening eigenvalue well
  below $-1$ (NiO reaches $\approx -6$), so the interacting fixed point needs
  Anderson acceleration, not plain damping, handled internally.
- A constant total-energy offset from a pseudo's semicore/NLCC convention cancels
  in $\Delta E(U)$ and in response, so compare differences, not absolute totals.

!!! note "Learning U vs learning the functional"
    U here is a *determinable and differentiable* input, with an exact
    $\mathrm{d}E/\mathrm{d}U$, the substrate a learning loop would use. The
    parameters gradwave trains today are the exchange-correlation functional
    (κ, μ) and the hybrid exchange mixing (α, ω). See
    [Learning XC by AD](learning-xc.md), whose adjoint carries a +U
    occupation-response channel so it trains correctly through a +U ground state,
    and [Hybrid functionals](hybrid-functionals.md).

## Next

Continue to [Non-collinear magnetism and spin-orbit coupling](noncollinear-soc.md).
