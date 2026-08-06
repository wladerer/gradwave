# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-06

### Added

- Distributed k-point-sharded SCF across `torchrun` ranks (Gloo), composed with
  IBZ symmetry reduction. Covers NC, USPP/PAW, and DFT+U, and routes `relax` and
  `eos` tasks through sharding.
- Equation-of-state (Birch-Murnaghan) and elastic-constant workflows, including
  relaxed-ion elastic constants and directional Poisson / auxetic scans.
- DFPT Gamma-point phonons with a full-metric-coupling Hvp path, plus dielectric
  response and Born effective charges, including the fully-relativistic (SOC) case.
- USPP/PAW support and DFT+U (Dudarev): end-to-end input surface, forces and
  stress, nspin=2, linear-response U, and noncollinear/SOC support.
- SCF flight recorder for per-iteration diagnostics.
- Noncollinear/SOC post-SCF analysis: Hellmann-Feynman forces, ELF, and the
  dielectric/Born path.
- Phase-diagram toolchain: quasi-harmonic thermodynamics, convex-hull ground
  states, configurational Monte Carlo, and common-tangent T-x construction.
- Differentiable chemical composition (alchemical) path for local, charge, and
  nonlocal terms.
- Harmonic thermodynamics from the phonon DOS.
- Gradual typing rollout (`ty`, jaxtyping, beartype) across the package, with an
  `@override` convention and a growing error-tier file list.
- MIT LICENSE file and CONTRIBUTING guide.

### Changed

- Default USPP/PAW nspin=2 mixing to Johnson; default `scf.mixing.scheme` to
  `auto` so the resolvers govern.
- Exposed the local Thomas-Fermi preconditioner as `scf.mixing.precond`.
- Seed Davidson from the previous ionic step's eigenvectors and remap warm-start
  density across FFT-grid changes.
- CPU-offload the batched-Davidson subspace solve and QR on CUDA, gated on
  measured fp64 capability.
- Opt-in energy-metric SCF convergence gate.
- Ran the feature/gate inventory sweep to parity and refreshed README, manual,
  and output-schema docs.

### Fixed

- Pulay stress correction for variable-cell relax, and Pulay pressure estimator
  accuracy via annulus extrapolation.
- Deadlock-free distributed result gather via CPU-staged raw-tensor all_gather.
- Rebuild the IBZ when positions break symmetry ops.
- Input validation for `ecut` and pseudo paths; allow `task: relax | eos` with
  `distributed: true`.
- Stoner spin-preconditioner fix for the FM-Ni noncollinear+SOC limit cycle.

## [0.1.0]

Initial baseline: differentiable plane-wave DFT for periodic solids in PyTorch.
Norm-conserving SCF loop, Davidson eigensolver, k-points and symmetry, forces
and stress, bands/DOS/PDOS, pseudopotential parsing, CLI, and public API.

[Unreleased]: https://github.com/wladerer/gradwave/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/wladerer/gradwave/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/wladerer/gradwave/releases/tag/v0.1.0
