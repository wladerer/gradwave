# Properties (post-SCF)

These functions take a converged SCF result and compute a derived quantity.
Forces and stress are single autograd passes through the total energy at the
converged point. The USPP/PAW variants carry the augmentation and one-center
terms. Each function documents which formalisms and spin settings it supports.

## Forces and stress

::: gradwave.postscf.forces.forces

::: gradwave.postscf.forces.hubbard_force

::: gradwave.postscf.stress.stress

::: gradwave.postscf.stress.stress_kbar

::: gradwave.postscf.stress.symmetrize_stress

### Ultrasoft / PAW

::: gradwave.postscf.paw_forces.forces_uspp

::: gradwave.postscf.paw_stress.stress_uspp

## Band structure

`band_structure` solves the fixed converged potential at arbitrary k-points;
`bands_along_ase_path` follows an ASE band path.

::: gradwave.postscf.bands.band_structure

::: gradwave.postscf.bands.bands_along_ase_path

::: gradwave.postscf.bands.BandStructure

## Phonons and the Hessian

`gamma_phonons` runs off central finite differences of the analytic forces;
`gamma_hessian` builds the analytic Γ Hessian by symmetry-irreducible columns.

::: gradwave.postscf.hessian.force_constants_gamma

::: gradwave.postscf.hessian.gamma_phonons

::: gradwave.postscf.phonons.gamma_hessian

::: gradwave.postscf.phonons.gamma_frequencies

::: gradwave.postscf.phonons.HessianSymmetry

## Density of states

::: gradwave.postscf.dos.kpm_dos

## Dielectric response

ε∞ and Born effective charges from E-field density-functional perturbation
theory.

::: gradwave.postscf.dielectric.dielectric_born

## Band symmetry

Irrep (Mulliken) labels for the bands at a k-point, from the little group of
the converged potential.

::: gradwave.postscf.irreps.band_irreps

::: gradwave.postscf.irreps.little_group

::: gradwave.postscf.irreps.KPointIrreps

::: gradwave.postscf.irreps.IrrepCluster

## Hubbard U as an observable

The Hubbard U is a first-class differentiable output, not only an input.
`energy_derivative_u` gives the exact dE/dU by stationarity;
`linear_response_u` and its autodiff variant compute the Cococcioni
linear-response U.

::: gradwave.postscf.hubbard_u.energy_derivative_u

::: gradwave.postscf.hubbard_u.linear_response_u

::: gradwave.postscf.hubbard_u.linear_response_u_autodiff
