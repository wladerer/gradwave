# Open tasks

Parked work from the overnight sessions, carried out of the ephemeral task
list so it survives. The open backlog of larger features lives in
[ideas.md](ideas.md). This file holds the smaller, already-scoped items that
were in flight or deferred.

## Deferred behind the performance sprint

### Pt(111) slab and CO adsorption

Two postponed application tasks, both waiting on the acceleration work to land
first.

- Build a minimal Pt(111) slab from the optimized bulk fcc Pt (the EOS lattice
  constant is already fit).
- Compute the CO adsorption binding energy on that slab.

### 0.23 eV total-energy offset vs QE (fcc Pt PAW)

gradwave gives −10167.53 eV against QE's −10167.30 eV for 1-atom fcc Pt
(a = 3.97 Å, 40/400 Ry, 12×12×12, gaussian 0.2 eV, identical psl kjpaw
pseudopotential). 0.23 eV is large for a converged metal. The likely cause is a
smearing free-energy (−TS) convention or a k-weight detail. It mostly cancels
in binding-energy differences but should still be pinned down by comparing the
energy-term breakdown against QE's.

## Estimator and post-processing

### Coarse-space Dyson refinement of δρ (task, not yet validated)

The self-consistent coarse-space correction dressing the first-order δρ. A
first cut is implemented as (1 − χ0 K)^−1 δρ⊥, reusing `apply_chi0` and
`apply_k_hxc`, opt-in and default off. It is not yet validated. On diamond
250→600 eV it is neutral-to-slightly-negative (corr 0.928→0.907), so the exact
Schur coupling must be pinned against Cancès JCP 2016 (arXiv 2111.01470) or the
DFTK source, then validated on a case where the coarse-space error is larger.
Do not enable by default until validated.

### nscf eigenvectors and broadening options

Return eigenvectors from `bands`/`bands_uspp` at a dense k-mesh on the frozen
potential. Add Methfessel-Paxton and tetrahedron broadening to the DOS binner.

## Acceleration, remaining tier

### Tier 3: real-space augmentation and fp32-deep GPU draft schedule

Two independent experiments, keep only what wins on measurement.

- Real-space (localized Q) augmentation as a cheaper alternative to
  reciprocal-space augmentation on the full dense grid.
- An fp32-dominant Davidson draft schedule that reserves fp64 for a final
  polish, targeting the GPU fp64 tax. This is hardware-specific to the consumer
  RTX 3050 and may not transfer.
