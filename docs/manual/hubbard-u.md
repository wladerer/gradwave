# Differentiable Hubbard U

A GGA underbinds the localized $d$ and $f$ electrons of transition-metal
compounds. The DFT+U correction fixes this with a single parameter per manifold,
and the parameter itself is not a knob to guess: gradwave computes it from linear
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
plugs into the same projector contraction the Kleinman-Bylander nonlocal term
already uses. For USPP/PAW the overlaps carry the $S$-metric, $\langle \phi | S |
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

## Determine U, and its gradient

`linear_response_u` runs the finite-difference probe (one base plus two perturbed
SCFs for $\chi$, cheap one-shot solves for $\chi^0$). `linear_response_u_autodiff`
gets the same number from a single ground-state SCF using conduction-projected
Sternheimer response, with the Hartree-XC screening kernel taken as an autograd
Hessian-vector product of $E_\text{Hxc}$ (so any twice-differentiable, including
learnable, functional works with no hand-coded $f_\text{xc}$).

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
    parameter gradwave actually trains today is the exchange-correlation
    functional. See [Learning XC by AD](learning-xc.md), whose adjoint carries a
    +U occupation-response channel so it trains correctly through a +U ground
    state.

## Next

Continue to [Non-collinear magnetism and spin-orbit coupling](noncollinear-soc.md).
