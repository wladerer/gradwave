# gradwave

gradwave is a differentiable plane-wave density functional theory code written
in PyTorch. It solves the Kohn-Sham equations[[1]](bibliography.md#ks) in a
plane-wave basis with norm-conserving or ultrasoft/PAW pseudopotentials. Every
energy term is differentiable, so one reverse-mode pass through the total energy
gives the forces and a second the Hessian. An implicit-differentiation wrapper
returns the response of the self-consistent density to any parameter of the
functional.

This manual is organized as a wiki. Start with [Installation](installation.md),
then work through the tutorials. They relax a geometry with exact autograd
forces, train an exchange-correlation functional back to PBE[[10]](bibliography.md#pbe)
through the self-consistent density, reduce the Brillouin zone by symmetry,
estimate the plane-wave basis error, determine the Hubbard U from linear
response, add spin-orbit coupling, and read spin Hamiltonians out of the magnetic
ground state. Each tutorial opens with the theory it rests on, then runs a shipped
example. For the shortest path to a result without the theory, the
[Cookbook](cookbook.md) has task recipes. The [Reference](reference.md) page
collects the CLI, the output files, and the entry points, and the
[Bibliography](bibliography.md) lists every citation.

## What it computes

- Total energies and free energies, with Fermi-Dirac, Gaussian, Methfessel-Paxton,
  and cold smearing for metals.
- Exact Hellmann-Feynman forces and the stress tensor from autograd, geometry and
  variable-cell relaxation through any ASE optimizer.
- Band structures with point-group irrep labels, total and projected (l, m, j)
  density of states.
- Brillouin-zone reduction by symmetry, with density and becsum symmetrization.
- Basis-set (Ecut) error estimation from a single calculation, no cutoff sweep.
- Exchange-correlation functionals trained by gradient descent through the SCF
  density, each gradient one adjoint solve.
- DFT+U with the Hubbard U determined from linear response and an exact dE/dU.
- Collinear spin, non-collinear magnetism, and spin-orbit coupling from
  fully-relativistic pseudopotentials.
- Constrained non-collinear magnetism, ground-state moment configurations, spin
  spirals, magnetocrystalline anisotropy, and the exchange constants (J, DMI) of a
  Heisenberg model.
- Γ-point phonons from the analytic self-consistent position response.
- Norm-conserving (ONCV) and ultrasoft/PAW pseudopotentials, detected from the UPF
  file, on CPU and GPU in float64/complex128.

## Validation vs Quantum ESPRESSO

Every capability is checked against Quantum ESPRESSO `pw.x`[[5]](bibliography.md#qe)
at identical cutoff, k-mesh, and pseudopotential. A representative set:

| quantity | agreement |
|---|---|
| Si total energy (LDA and PBE, 30 Ry, 4×4×4) | ≤ 0.001 meV/atom |
| Al free energy (PBE, semicore, Gaussian, 40 Ry) | < 2 meV/atom |
| Si forces (displaced, vs `tprnfor`) | < 5 meV/Å |
| Si band structure L–Γ–X–U–Γ (occupied) | < 10 meV |
| bcc Fe magnetic moment (spin-PBE, 60 Ry) | 2.2244 vs 2.22 μB |
| learnable XC, dE/dθ and dL(ρ)/dθ vs SCF finite difference | 1e-5 / 2e-4 rel |
| NiO Hubbard U vs `hp.x` DFPT | 6.449 vs 6.431 eV (0.3%) |
| Si Γ phonon (PAW) vs `ph.x` | 0.003% |
| GaAs spin-orbit split-off Δ₀ vs fully-relativistic QE | 0.336 eV, 2e-3 eV |

The [Performance](performance.md) page reports wall times (a Si SCF runs in 1.4 s on
an RTX 3050) and works through where the small-system gap against a mature Fortran
code comes from.

## Design

Every energy term in gradwave is a pure tensor function of the atomic positions,
the cell, the density, and the exchange-correlation parameters. The
self-consistent field (SCF) loop and the eigensolver run under `no_grad` and stay
invisible to autograd, following the plane-wave pseudopotential formulation of
Payne et al.[[2]](bibliography.md#payne) The converged quantities are then detached
and fed back into the pure energy, so a single backward pass differentiates it.

Two capabilities follow from that design.

- Geometry optimization uses exact Hellmann-Feynman
  forces[[3]](bibliography.md#feynman) from autograd, driven by any ASE
  optimizer.[[4]](bibliography.md#ase)
- Functionals are trained by gradient descent through the SCF fixed point, where
  each gradient is one adjoint solve rather than finite differences.

The code is validated against Quantum ESPRESSO
`pw.x`[[5]](bibliography.md#qe) at identical cutoff, k-mesh, and pseudopotential.
The formalism is detected from the UPF file, so the same input schema drives the
norm-conserving (ONCV[[6]](bibliography.md#oncv)) and
USPP/PAW[[7]](bibliography.md#paw) paths. Base units are eV and Å, and every
tensor is float64 or complex128.

## Pages

| page | what it covers |
|---|---|
| [Installation](installation.md) | set up the environment and run the first SCF |
| [Cookbook](cookbook.md) | task recipes, the shortest path to each quantity |
| [Geometry optimization](geometry-optimization.md) | relax a structure with autograd forces |
| [Learning XC by AD](learning-xc.md) | train a functional through the SCF density |
| [Symmetry reduction](symmetry.md) | IBZ k-reduction and density symmetrization |
| [Basis-set error estimation](error-estimation.md) | estimate the plane-wave (Ecut) error |
| [Differentiable Hubbard U](hubbard-u.md) | DFT+U and U from linear response |
| [Non-collinear magnetism and SOC](noncollinear-soc.md) | spinor SCF and spin-orbit coupling |
| [Magnetic structure and spin Hamiltonians](magnetism.md) | moment configs, spin spirals, J and DMI |
| [Inputs and outputs](io.md) | input schema, output files, checkpoints, analysis |
| [Reference](reference.md) | CLI and entry points |
| [Performance](performance.md) | where time goes, what helps, the GPU precision story |
| [Wisdom](wisdom.md) | non-obvious do and do-not rules learned the hard way |
| [Bibliography](bibliography.md) | numbered references |
