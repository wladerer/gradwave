# Joint geometry + electronic optimization (prototype, #122)

Status: prototype, norm-conserving insulators only. Code: `gradwave/opt/joint.py`;
tests: `tests/unit/test_joint_opt.py` (fast + standard),
`tests/integration/test_joint_vs_bfgs.py` (slow, the head-to-head).

## Problem

The standard relaxation nests a full SCF inside every BFGS geometry step:
~10 geometry steps × ~10 SCF iterations × a Davidson solve each. But gradwave's
total energy is an explicit differentiable function of the cell, the positions,
and the orbital coefficients, so nothing forces the nesting — one optimizer can
descend on all three at once and let the electronic state converge *alongside*
the geometry instead of *inside* it.

## Formulation

Minimize the KS total energy over three blocks of variables:

- **strain** ε (3×3, symmetrized): `a(ε) = a₀(1+ε)ᵀ`, fixed-basis
  (Nielsen–Martin) convention — integer Miller labels frozen, `G = m·B(ε)`,
  the exact parametrization `postscf/stress.py` differentiates. Same Pulay
  (basis-incompleteness) caveat as any fixed-basis variable-cell step.
- **positions** as fractional coordinates `s` (they ride the strained cell:
  `τ = s·a(ε)`).
- **orbitals**: unconstrained complex variables `Z_k` per k-point,
  orthonormalized *inside* the energy by Löwdin, `C_k = (Z_k Z_kᴴ)^{-1/2} Z_k`,
  after a diagonal Teter-style preconditioner `p(G) = 1/(1+T_G/T_ref)`
  (`C_k = Löwdin(Z_k · p)`). Löwdin is smooth (no QR sign/pivot kinks) and the
  preconditioner turns L-BFGS's identity metric on Z into a TPA-preconditioned
  metric on C — without it the coefficient block's conditioning is ~ecut/gap
  and first-order descent crawls.

Only the `N_e/2` occupied bands are carried, with fixed occupations f = 2.
The occupied-subspace energy is invariant under band rotations, so occupied
near-degeneracies are harmless; the *gap* is what protects the subspace.
This is deliberately insulator-only — see "Metals" below.

The energy assembly (`joint_energy`) mirrors `postscf.stress._energy_strained`
term by term (strained density sphere from Miller labels, differentiable SBT
form factors, strained projector columns, `ewald_strained`), with two changes:
ρ is rebuilt from the live coefficients on the graph, and positions enter as
live fractional leaves (`_strain.strain_cell` detaches positions, so the module
assembles the strained cell itself). At ε = 0 with converged SCF orbitals the
functional reproduces the SCF total energy to ~1e-9 eV and its ε-gradient
reproduces `stress()` exactly (both asserted in the fast tests).

**Cell-dependent basis.** The plane-wave count changes with the cell, so a
single fixed G-sphere would pin the answer to the starting cell's basis. The
driver (`joint_relax`) runs an outer rebuild loop: L-BFGS to convergence in the
frozen basis, re-setup at the relaxed cell, transfer orbitals by Miller-index
matching (zero-fill the new columns, Löwdin repairs orthonormality), repeat
until the per-cycle strain falls below `rebuild_tol`. In practice cycle 2+
converges in a handful of closures. The nested reference (ASE +
`FrechetCellFilter`) rebuilds its basis every cell step and its stress is the
same fixed-basis expression, so both methods land on the same fixed-basis
minimum at the final cell.

**Optimizer.** One `torch.optim.LBFGS` (strong-Wolfe line search, history 120)
over all leaves. Convergence gates: max Cartesian force < `fmax`, max stress
component < `smax`, max coefficient gradient < `ctol`.

**Seeding.** Cycle 0 seeds the orbitals with 3 loose SCF iterations
(diago_tol 1e-4); those Davidson H-applies are counted exactly and charged to
the joint method. Seeding from random/plane-wave-identity orbitals works but
wastes ~2× more closures than the loose SCF costs.

## H-application accounting

One "H-apply" = Ĥ acting on one band vector ≈ 2 sphere↔grid FFTs + one
projector contraction.

- **Nested reference**: counted exactly by patching
  `BatchedHamiltonian.apply` (`count_h_applies`), summing nk·nb of every
  applied block across all Davidson solves of all SCFs of the relaxation.
- **Joint**: one closure (energy + backward) costs one ψ FFT per (band, k) for
  the density plus its adjoint in the backward pass — ≈ 1 H-apply per (band, k)
  per closure; the Hartree/XC/local terms are band-independent dense-box FFTs
  (a handful, amortized across bands, same as the reference's own density
  builds, which the exact counter also does not charge). Reported
  `h_equiv = h_seed + n_closures · nk · n_occ`.

The comparison is FFT-work-based and slightly generous to the *reference*
(its density builds, mixing, and stress/force autograd passes are not
charged; the joint side's equivalent overheads are the same dense-box terms).

## Results (Si, 2-atom primitive cell, LDA, 15 Ry, 2×2×2, no symmetry)

Start: atom 1 displaced 0.08–0.05–0.03 Å, cell strained ~1.5% (normal + shear).
Both relaxed to fmax = 0.005 eV/Å (reference: BFGS + FrechetCellFilter).
Final energies compared via a fresh fully-converged SCF at each method's final
geometry (removes either method's own energy-accounting bias).

| quantity | nested BFGS+SCF | joint | agreement |
|---|---|---|---|
| bond length [Å] | TBD | TBD | TBD |
| a₁ length [Å] | TBD | TBD | TBD |
| E(SCF @ final) [eV] | TBD | TBD | TBD Ha |
| H-applies | TBD | TBD (seed TBD + closures TBD) | ratio TBD× |

(Fill from `pytest tests/integration/test_joint_vs_bfgs.py -m slow -n0 -s`.)

## Failure modes observed / avoided

- **Coefficient conditioning** — dominant issue. Unpreconditioned L-BFGS on Z
  needs many × more closures; the diagonal Teter scaling recovers most of it.
- **Level crossings** — avoided by construction here (insulator, occupied
  subspace only, rotation-invariant energy). Attempting smeared occupations
  from current Ritz values would make E(Z) non-smooth exactly at crossings of
  the Aufbau frontier; that is the metal extension's real problem, not a bug
  of this prototype.
- **Cell-dependent basis** — handled by the rebuild loop; a single frozen
  sphere biases the lattice constant at 15 Ry by more than the 1e-3 Å bar.
- **Löwdin rank** — if a line-search step makes Z nearly rank-deficient the
  S^{-1/2} clamps (1e-14); never triggered in the Si runs.

## Metals (not attempted, by design)

Smeared Aufbau occupations from the current Ritz values make the objective
only piecewise-smooth in Z (occupation weights reshuffle at crossings), and
L-BFGS assumes smoothness. The clean formulations are free-energy functionals
over an *extended* set of bands with occupations as variables (Marzari–
Vanderbilt ensemble DFT) — a different prototype.

## Next steps

- Metals via ensemble-DFT occupations (see above) on the Al fixture.
- Momentum/trust-region alternatives to L-BFGS for the coefficient block
  (OMM/exponential-map with explicit Riemannian gradient).
- Share the seed with the calculator's warm-start machinery so MD-style
  trajectories amortize the seed entirely.
- PAW/USPP: needs the S-metric in the Löwdin step and the augmentation
  channels on the strain graph (`paw_stress` already has the latter).
