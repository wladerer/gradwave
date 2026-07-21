# Open tasks

Parked work from the overnight sessions, carried out of the ephemeral task
list so it survives. The open backlog of larger features lives in
[ideas.md](ideas.md). This file holds the smaller, already-scoped items that
were in flight or deferred.

## Application tasks

### Pt(111) slab and CO adsorption

Two application tasks that were parked behind the acceleration work. That work
has since concluded (dual FFT grid, local-TF preconditioner, and compiled XC
landed; CheFSI and RMM-DIIS closed as measured negatives), so these are no
longer blocked.

- Build a minimal Pt(111) slab from the optimized bulk fcc Pt (the EOS lattice
  constant is already fit).
- Compute the CO adsorption binding energy on that slab.

## Estimator and post-processing

### Coarse-space Dyson refinement of δρ (task, not yet validated)

The self-consistent coarse-space correction dressing the first-order δρ. A
first cut is implemented as (1 − χ0 K)^−1 δρ⊥, reusing `apply_chi0` and
`apply_k_hxc`, opt-in and default off. It is not yet validated. On diamond
250→600 eV it is neutral-to-slightly-negative (corr 0.928→0.907), so the exact
Schur coupling must be pinned against Cancès JCP 2016 (arXiv 2111.01470) or the
DFTK source, then validated on a case where the coarse-space error is larger.
Do not enable by default until validated.

This is the same Dyson machinery, and the same missing term, as the *response*
diagnostic in the SCF convergence-error estimator: `estimate_scf_error` in
`postscf/convergence_error.py` reports its headline from the energy trajectory
(robust, validated), but its `denergy_response` form still omits the `chi0^-1`
kinetic-response term and is kept as a labelled diagnostic only. Pinning the
exact Schur coupling here should validate the Dyson dressing and promote that
diagnostic to a real second-order estimate.

### nscf eigenvectors and broadening options

Return eigenvectors from `bands`/`bands_uspp` at a dense k-mesh on the frozen
potential. Add Methfessel-Paxton and tetrahedron broadening to the DOS binner.

## Acceleration, remaining tier

### Tier 3: real-space augmentation

Real-space (localized Q) augmentation as a cheaper alternative to
reciprocal-space augmentation on the full dense grid. An independent experiment,
keep only if it wins on measurement.

The companion fp32-dominant Davidson draft schedule (fp64 reserved for a final
polish, targeting the GPU fp64 tax) is deferred as effectively pre-answered: the
acceleration-frontier sweep in [ideas.md](ideas.md) and the CheFSI/RMM-DIIS
negatives all point to the consumer-RTX-3050 fp64 tax being the wall, with
fp32-deep schedules hardware-specific and unlikely to transfer. Revisit only on
a real-fp64 card (A100/H100), where the fp32 gain has room to dominate.
