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

## Projected density of states

Atom-, orbital-, spin-, and (fully-relativistic) j-resolved projections; see
the `projections` input block in [Inputs and outputs](../io.md).

::: gradwave.postscf.pdos.projected_dos

::: gradwave.postscf.pdos.projected_dos_noncollinear

::: gradwave.postscf.pdos.projected_dos_soc

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

## Basis-set error estimate

The plane-wave (Ecut) discretization error from one converged run; see the
[Basis-set error estimation](../error-estimation.md) tutorial.

::: gradwave.postscf.discretization_error.estimate_density_error

::: gradwave.postscf.discretization_error.estimate_force_error

::: gradwave.postscf.discretization_error.DiscretizationError

## Magnetic configuration (constrained)

Constrained non-collinear DFT: hold atomic moments toward target directions and
descend the torque to the ground-state configuration; see the
[Non-collinear magnetism and SOC](../noncollinear-soc.md) tutorial.

::: gradwave.postscf.moment_config.atomic_weights

::: gradwave.postscf.moment_config.reference_moment_magnitudes

::: gradwave.postscf.moment_config.constrained_moment_scf

::: gradwave.postscf.moment_config.relax_moment_directions

The `"perp"` (direction-only) and `"vector"` (magnitude-robust) penalties, and
the autograd that produces both the SCF constraining field and the config-search
torque from one scalar, live in `gradwave.scf.moment_penalty`.

::: gradwave.scf.moment_penalty.penalty_energy

::: gradwave.scf.moment_penalty.field_coeff

::: gradwave.scf.moment_penalty.direction_gradient

## Spin-Hamiltonian parameters (J, D, K)

The exchange tensor `𝒥ᵢⱼ = ∂Tᵢ/∂êⱼ` is the site-to-site derivative of the
constrained-moment torque; `decompose` splits it into the isotropic Heisenberg J,
the DMI vector component, and the symmetric-traceless anisotropic exchange. See the
bcc Fe benchmark against Pajda 2001 in `examples/fe_exchange.py`.

::: gradwave.postscf.spin_exchange.exchange_from_atom

::: gradwave.postscf.spin_exchange.decompose

::: gradwave.postscf.spin_exchange.heisenberg_couplings

## Magnetic characterization (one call)

`characterize_magnetism` is the high-level entry point: it runs a non-collinear
reference SCF for the atomic moments, extracts the exchange couplings, and returns a
`MagneticReport` with the moments, magnetic ordering, Heisenberg J, DMI, and a
mean-field Curie temperature — the important quantities and qualities in one call.

::: gradwave.postscf.magnetism.characterize_magnetism

::: gradwave.postscf.magnetism.MagneticReport

## Volumetric export

Real-space fields written to `.cube`/`.xsf` for VESTA and Ovito: the density
(CHGCAR analog), the band-decomposed density (PARCHG analog), the electron
localization function, and the noncollinear magnetization density. The `write_*`
helpers take a converged result and a path, choosing the format from the extension.
The `output.volumetric` input block calls these after a run; see
[Inputs and outputs](../io.md#volumetric-export) for the coverage by result type.

::: gradwave.postscf.volumetric.write_density

::: gradwave.postscf.volumetric.write_band_density

::: gradwave.postscf.volumetric.write_elf

::: gradwave.postscf.volumetric.write_magnetization

::: gradwave.postscf.volumetric.density

::: gradwave.postscf.volumetric.band_density

::: gradwave.postscf.volumetric.elf

::: gradwave.postscf.volumetric.magnetization
