# Differentiable chemical composition (non-VCA)

Design note for making chemical composition a differentiable parameter, so that
`dE/dcomposition` and `d(property)/dcomposition` are exact autograd derivatives
and composition can drive gradient-based inverse design.

## Why not the virtual crystal approximation

Composition is discrete. A site is Ga or Al, not 0.7 Ga. The virtual crystal
approximation makes it continuous by averaging the pseudopotential per site and
running the SCF on that averaged crystal, then reading off its properties. That
averaged crystal is fictitious. It loses local chemistry, charge transfer, and
local relaxation, and it is systematically wrong for size-mismatched or
chemically distinct species.

This design never treats an averaged-potential crystal as a physical material.
Two mechanisms deliver differentiable composition without that step, and they
compose.

### Alchemical composition derivatives (the sensitivity engine)

At a real integer-composition structure, the exact response of the energy to
transmuting site i, `dE/dλ_i`, is a Hellmann-Feynman expectation of the
pseudopotential derivative. It is free at convergence for the total energy, and
the self-consistent part for non-energy losses comes from the SCF adjoint the
code already has (`scf/implicit.py`, `postscf/uspp_implicit.py`). The derivative
is that of a real material, not an averaged one. Between two real compositions A
and B, thermodynamic integration of `dE/dλ` over the path recovers the exact
energy difference. The endpoints are real and the λ-path is an integration
device, not a claimed material.

### Differentiable occupation surrogate (the optimization object)

A cluster expansion writes the energy as a polynomial in the site-occupation
variables, relaxed to continuous values, and it is differentiable in those
variables with corners that are exact DFT. It is fit to real integer-composition
energies and their alchemical gradients. It never averages a potential.
Composition is optimized on the smooth surrogate, snapped to integer, and
verified with DFT.

The synthesis is the point. Because the whole stack is autograd, the composition
channel composes with every property derivative already present, so
`dE_gap/dcomp`, `dB_modulus/dcomp`, `dBarrier/dcomp`, and `dC_V/dcomp` all follow
from one backward pass with the geometry response folded in through the Hessian.

## Mechanism in the code

A per-atom composition weight `λ_i` in [0, 1] blends two endpoint
pseudopotentials, threaded as a differentiable input.

- Charge. `charges_i(λ) = (1 − λ_i) Z_A + λ_i Z_B` at `scf/loop.py:186`. This
  flows into `ewald_energy(charges=...)`, `n_electrons`, and the G=0 local term.
- Local potential. `setup_common.build_vloc_tables` moves from per-species to
  per-atom blended form factors `v_i(G, λ) = (1 − λ_i) v_A(G) + λ_i v_B(G)`. The
  structure factor already sums per atom, so this extends the existing assembly.
- Nonlocal projectors. `pseudo.kb.beta_form_factors` and the D_ij channel
  energies blend per atom. This constraint sets the scope. Endpoint pseudos must
  share angular-momentum channels on a common grid, so norm-conserving isovalent
  pairs come first.

With `λ` a `requires_grad` tensor, autograd gives `dE/dλ`, and the implicit-diff
adjoint supplies the self-consistent term for non-energy losses.

## Phases

The first target is isovalent (Z conserved, Ewald neutral), with the alchemical
gradient engine landing before the surrogate.

1. Alchemical channel and endpoint exactness. Per-atom blended local potential
   and charge, λ threaded to autograd. The oracle has two parts. At λ=1 the
   system reproduces the real B-substituted cell in energy and forces, and
   `dE/dλ` matches a full-SCF finite difference over λ at a relaxed reference.
   The first alloy is Si-Ge. **Landed** (`scf.alchemical.setup_alchemical_system`,
   `blend_local_table`, `alchemical_charges`; oracles `test_alchemical_scf_endpoints_match_pure`,
   `test_local_potential_gradient_vs_fd`, `test_charge_channel_ewald_gradient_vs_fd` in
   `tests/unit/test_alchemical_composition.py`).
2. Nonlocal projector blending. Extend to the KB projectors for channel-
   compatible norm-conserving pairs. The oracle checks that the Si-to-Ge
   transmutation gradient matches finite difference, and that thermodynamic
   integration from Si to Ge recovers the real energy difference within tolerance.
   **Landed** (`scf.alchemical.blend_projector_data` and the nonlocal term in
   `alchemical_energy_gradient`; oracles `test_nonlocal_blend_gradient_vs_fd`,
   `test_alchemical_scf_gradient_hellmann_feynman`).
3. Geometry and property response. Fold in `dR/dλ` by solving
   `K (dR/dλ) = −d(force)/dλ` with the existing Hessian, so property derivatives
   are correct through relaxation. The oracle checks `d(lattice constant)/dλ`
   and `dB_modulus/dλ` against finite difference. **Not landed.** `scf/alchemical.py`
   folds no geometry response; `alchemical_energy_gradient` is a fixed-geometry
   Hellmann-Feynman energy gradient, with no `K (dR/dλ)` solve against the Hessian.
4. Heterovalent extension. Charge-changing substitutions on the same channel,
   with a compensating background and Ewald neutrality. Higher physics risk, so
   it follows the validated isovalent machinery. **Landed.** `alchemical_energy_gradient`
   adds the Janak chemical-potential term `μ · dN/dλ` (`scf/alchemical.py:268`,
   `float(res.fermi) * charges.sum()`) and the NLCC core-correction term for a
   charge-changing endpoint; oracles `test_per_site_alchemical_heterovalent_endpoint`,
   `test_heterovalent_gradient_core_and_janak`.
5. Surrogate and inverse-design loop. A cluster expansion or small equivariant
   surrogate fit to real endpoints and alchemical gradients, then composition
   optimization toward a target property with snap-and-verify. Demonstrated on
   one alloy, for example a Si_xGe_{1-x} band-gap or modulus target. **Partially
   landed.** The surrogate and optimizer exist (`postscf.composition_design.fit_surrogate`,
   `optimize_composition`, `sample_alchemical`); the end-to-end snap-and-verify
   inverse-design demonstration on an alloy is not yet in the repo.

## Scope limits

- Intermediate λ is a modeling device, not a physical material. Only the
  endpoints and the derivatives and integrals between them are claimed.
- Nonlocal blending needs channel-compatible pseudos. Heterovalent cases need a
  compensating background and careful Ewald neutrality.
- The self-consistent response term inherits the implicit-diff backend scope,
  norm-conserving nspin=1 insulators for `scf/implicit.py`, with the USPP/PAW
  backend lifting the spin restriction. The Hellmann-Feynman term is general.

## Payoff

Composition becomes a first-class differentiable input alongside positions,
strain, and the functional parameters. Gradient-based inverse design can then
target any differentiable property, a band gap, an elastic modulus, a diffusion
barrier, or a heat capacity, with the geometry response included, on the same
autograd stack that already produces those properties.

The "with the geometry response included" part of this payoff, and the synthesis
claim above that `dB_modulus/dcomp`, `dBarrier/dcomp`, and `dC_V/dcomp` follow
from one backward pass, both depend on Phase 3, which has not landed. Today
`alchemical_energy_gradient` gives `dE/dλ` and property derivatives at fixed
geometry. Any derivative that runs through relaxation, `dB_modulus/dλ` and
`d(lattice constant)/dλ` among them, waits on the `K (dR/dλ) = −d(force)/dλ`
solve of Phase 3.
