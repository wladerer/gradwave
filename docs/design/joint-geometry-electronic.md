# Joint geometry + electronic optimization (prototype, #122)

Status: prototype, norm-conserving. Insulators (fixed occupations) and metals
(variational occupations, free-energy descent) are both supported, with LDA or
PBE. Code: `gradwave/opt/joint.py`, `gradwave/opt/_metals.py`;
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
  orthonormalized *inside* the energy by Cholesky, `C_k = L⁻¹(Z_k·p)` with
  `Z Zᴴ = L Lᴴ`, after a diagonal Teter-style preconditioner
  `p(G) = 1/(1+T_G/T_ref)`. The preconditioner turns L-BFGS's identity metric
  on Z into a TPA-preconditioned metric on C — without it the coefficient
  block's conditioning is ~ecut/gap and first-order descent crawls.

For an insulator only the `N_e/2` occupied bands are carried, with fixed
occupations f = 2. The occupied-subspace energy is invariant under band
rotations, so occupied near-degeneracies are harmless; the *gap* is what
protects the subspace. For a metal the occupations become variational and the
object minimised is the Mermin free energy — see "Metals" below.

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

## Results

### Fixed cell (Si2, LDA, 12 Ry, 2×2×2, atom displaced 0.1 Å)

Joint descent (positions + orbitals, `fix_cell=True`): converged to
fmax < 0.005 eV/Å in **80 closures**, bond 2.35124 Å vs ideal 2.35126 Å
(2e-5 Å). Cost: h_seed 1432 (3 loose SCF iterations) + 80·8k·4b = 2560,
**h_equiv 3992 — less than ONE fully converged SCF at the same settings**
(a single cold SCF here is ~4000–4500 H-applies). Every nested-relaxation
scheme pays ≥ n_steps such SCFs.

### Variable cell head-to-head (Si, 2-atom primitive cell, LDA, 15 Ry, 2×2×2, no symmetry)

Start: atom 1 displaced 0.08–0.05–0.03 Å, cell strained ~1.5% (normal + shear).
Both relaxed to fmax = 0.005 eV/Å (reference: BFGS + FrechetCellFilter).
Final energies compared via a fresh fully-converged SCF at each method's final
geometry (removes either method's own energy-accounting bias).

| quantity | nested BFGS+SCF | joint | agreement |
|---|---|---|---|
| bond length [Å] | 2.41596 | 2.41604 | 8e-5 Å |
| a₁ length [Å] | 3.94585 | 3.94535 | 5e-4 Å |
| E(SCF @ final) [eV] | −211.460329 | −211.460332 | 1.2e-7 Ha |
| H-applies | 74 920 | 9 632 (seed 1 664 + 249 closures) | **7.8× fewer** |

Joint used 4 basis-rebuild cycles and 249 total closures. Success bar
(energy within 1e-5 Ha, geometry within 1e-3 Å, fewer H-applications): met.
Run on asus via `pytest tests/integration/test_joint_vs_bfgs.py -n0 -s -m ""`.

## Failure modes hit (attempt log)

1. **Attempt 1 — symmetric Löwdin (eigh) orthonormalization: FAILED.**
   `eigh`'s backward carries 1/(λᵢ−λⱼ) factors; symmetry-degenerate Si bands
   at the 2×2×2 zone-boundary k-points make Z Zᴴ have *exactly* repeated
   eigenvalues, so the very first backward pass returned NaN grads, the first
   L-BFGS update wrote NaN into every leaf, and the next Cholesky/eigh call
   crashed. Γ-only runs (accidentally non-degenerate) hid the bug. The energy
   is invariant to which orthonormal frame is returned, so the fix is free:
2. **Attempt 2 — Cholesky orthonormalization (C = L⁻¹Z): WORKS.**
   Cholesky's backward is spectrum-independent; a trace-scaled jitter
   (1e-13·tr S/n) keeps S SPD when a line-search trial approaches rank
   deficiency. All results below are attempt 2.
   (A 3rd attempt — Adam warmup or exponential-map/OMM — was not needed; kept
   in "next steps" as optimizer work, not correctness work.)

Other modes watched for:

- **Coefficient conditioning** — the diagonal Teter scaling is what makes
  L-BFGS viable; the un-preconditioned metric has condition ~ecut/gap.
- **Volume collapse under line search** — strong-Wolfe trial steps along ε
  are unbounded; det(1+ε) → 0 overflows the 1/Ω energy terms into NaN. The
  closure short-circuits det(1+ε) ∉ [0.2, 5] to a finite quadratic penalty
  and the line search backtracks off it.
- **Level crossings** — for insulators avoided by construction (occupied
  subspace only, rotation-invariant energy). For metals the smearing entropy
  makes F smooth through Aufbau-frontier crossings (occupation weights change
  continuously), which is exactly why the free-energy functional — not the bare
  energy — is the object to minimise; see "Metals" below.
- **Cell-dependent basis** — handled by the rebuild loop; a single frozen
  sphere biases the lattice constant at 15 Ry by more than the 1e-3 Å bar.

## GGA / PBE

The joint functional is agnostic to the semilocal functional: `joint_energy`
already rebuilds `σ = |∇ρ|²` on the strained grid whenever `xc.needs_gradient`,
so passing a `PBE()` (or any GGA) `xc` to `joint_energy` / `joint_relax` "just
works". At ε = 0 with the converged PBE orbitals the functional reproduces the
PBE SCF total to ~1e-5 eV and its ε-gradient reproduces `stress()` with the PBE
sigma term — both asserted in the fast tests
(`test_joint_energy_pbe_*`). meta-GGA (`needs_tau`) is still refused (the τ
rebuild on the coefficient graph is not wired).

## Metals — free-energy descent with variational occupations

`gradwave/opt/_metals.py` + `joint_free_energy`. When `joint_relax` is given a
`smearing` scheme it carries `n_bands ≥ N_e/2` orbitals and minimises the
**Mermin free energy** `F = E − σS` instead of `E`. Occupations are made a
function of the orbitals by reusing the SCF's own machinery, not by adding
optimizer variables:

1. Build the **subspace Hamiltonian** `H_sub[k] = C_k^† Ĥ[ρ] C_k`
   (`n_bands × n_bands`) term by term from the SAME ingredients `joint_energy`
   already assembles — per-band kinetic on `(k+G)²`, the real-space ψ grids for
   the local `V_eff` matrix elements, the projector overlaps for the nonlocal
   block. It adds **no** sphere↔grid FFTs beyond the ψ grids the density build
   already pays for, so the H-apply accounting stays ≈ 1 per (band, k) per
   closure (now with `n_bands` instead of `N_e/2` bands).
2. Diagonalise `H_sub` → Ritz values `λ_i`; feed them to the shared Fermi
   search / entropy (`core.occupations`, exactly the SCF path) → occupations
   `f_i` and `−σS`.
3. Rebuild ρ (and the energy) from the occupation-weighted **Ritz** orbitals
   (`C' = Rᵀ C`), i.e. the density matrix `γ = R diag(f) Rᵀ` in the coefficient
   basis.

The circularity ρ → V_eff → H_sub → λ → f → ρ is closed by a short **detached**
fixed-point at the base cell (orbitals held fixed, only occupations/potential
iterate — 5-6 iterations reach the SCF occupations). Because the whole
occupation solve is detached, the notorious `torch.linalg.eigh` backward — whose
`1/(λ_i−λ_j)` factors blow up at the *maximally* degenerate Fermi surface —
**never runs**: the live gradient reaches the free energy only through the
diagonal weights `f` (detached) and the rotation `R` (detached), plus the
orbitals and geometry at fixed occupations. This is the block-coordinate
(alternating orbital / occupation) reading of ensemble DFT; the occupation block
is re-solved every closure so both blocks reach stationarity together. It is
exact at the SCF fixed point — there `C` already diagonalises `H_sub` (`R = I`)
and `f` is the self-consistent Fermi filling — so:

- **F(ε = 0) = SCF free energy** to ~1e-7 eV (LDA and PBE), and
- **dF/dε = Ω·stress()** to ~1e-4 eV/Å³ with occupations detached (fixed
  occupations have no explicit ε-dependence, matching `postscf/stress.py`).

Both are asserted in `test_joint_free_energy_*` / `test_metal_stress_*` on a
small smeared Al cell.

### The bug that hides behind every energy test

`H_sub` must be assembled with a **consistent bra/ket conjugation** across all
three blocks: `H[a,b] = Σ c_a*(G) … c_b(G)` (first index conjugated). Writing
the per-band Rayleigh quotient naively — conjugating the *ket* for kinetic and
nonlocal but the *bra* for the local term — leaves the (real) diagonal
untouched, so it passes every total-energy and stress test, yet
transpose-conjugates the off-diagonal and corrupts **every** eigenvalue by
electron-volts (on Al a spurious ~1 eV Fermi-level shift and a metal that looks
insulating, `S ≈ 0`). Caught only by diffing `eigh(H_sub)` against
`BatchedHamiltonian`; the fixed convention is asserted implicitly by the
free-energy anchor.

### Descent — what works, and the honest limitation

The metal *relaxation* (`joint_relax(smearing=…)`) is where the theory meets the
optimizer, and three schemes were tried before one held:

1. **Full subspace rotation, occupations recomputed every closure — stalls.**
   Recomputing (rotation `R`, occupations `f`) inside each line-search trial
   hands L-BFGS a gradient inconsistent with the moving objective; strong-Wolfe
   makes no progress (Al: E frozen, `‖∇‖` constant across dozens of closures).
2. **Marzari–Vanderbilt occupation *variables* (η as extra L-BFGS leaves) —
   stalls.** The occupation block (eV-scale) and the preconditioned coefficient
   block (O(1)) live on wildly different scales; the single L-BFGS metric is so
   ill-conditioned the line search dies with `|∂F/∂η| ≈ 3` unresolved.
3. **Block-coordinate DIAGONAL ensemble — converges (with two fixes).** Each
   orbital is occupied by its own Rayleigh quotient `e_i = ⟨ψ_i|Ĥ|ψ_i⟩`
   (`_metals.diagonal_occupations`, no rotation, so nothing drifts), FROZEN per
   L-BFGS chunk (so the objective and gradient are consistent), and — the second
   fix — the L-BFGS **memory is reset each chunk** (curvature pairs from the
   previous chunk's different-occupation objective otherwise drive a persistent
   oscillation). The orbital-gradient gate is relaxed to ~2.5e-3 (the per-chunk
   occupation refresh perturbs the orbital block at that level).

Head-to-head (positions-only, 2-atom bcc-like Al, LDA, 13 Ry, 3×3×3, gaussian
σ=0.3), joint vs nested BFGS+SCF, both to fmax = 0.02 eV/Å:

| quantity | nested BFGS+SCF | joint | agreement |
|---|---|---|---|
| E(SCF @ final) [eV] | −3362.5694 | −3362.5372 | 1.2e-3 Ha |
| Al–Al pair sep [Å] | 3.5104 | 3.5753 | 0.065 Å |
| H-applies | 43 442 | 13 930 | **3.1× fewer** |

So the free-energy *functional* is exact (the ε=0 anchors), and the descent is
real and cheaper in Hamiltonian work, but it is a **partial** result: the
frozen-occupation (diagonal-ensemble) force carries a bias that floors the
attainable fmax at ~0.02 eV/Å — tightening to 0.008 does not converge — so on
this soft mode the geometry agrees only to ~0.07 Å and the energy to ~1.5e-3 Ha,
not the insulator's 1e-3 Å / 1e-7 Ha. The bias vanishes only at exact
self-consistency (where the diagonal quotients are the eigenvalues and the
Hellmann–Feynman force is complete); closing it needs the occupation *response*
force, i.e. the eigenvector-sensitivity the detached scheme deliberately drops.

## Next steps

- **Close the metal force bias.** The clean route is a degeneracy-robust
  `eigh` backward (custom `autograd.Function`, Lorentzian-broadened
  `1/(λ_i−λ_j)`) so the rotation/occupation response can stay LIVE without NaNs
  at the Fermi surface — then the full subspace scheme (attempt 1) becomes a
  single consistent objective and the frozen-occupation bias disappears.
- **Precondition the MV occupation block** (attempt 2) so occupation and
  coefficient variables share a metric — the other way to a consistent joint
  descent.
- **Warm-start the occupation solve across closures** (carry ρ / f) to drop the
  per-closure inner fixed-point from ~5 iterations to 1-2.
- Momentum/trust-region alternatives to L-BFGS for the coefficient block
  (OMM/exponential-map with explicit Riemannian gradient).

- **Warm-start the occupation solve across closures** (carry ρ / f) to drop the
  per-closure inner fixed-point from ~6 iterations to 1-2.
- **Marzari–Vanderbilt occupation *variables*** as an alternative to the
  subspace route if line-search behaviour degrades on harder metals (occupation
  levels as extra L-BFGS leaves; smearing keeps F smooth, no `eigh` at all).
- Momentum/trust-region alternatives to L-BFGS for the coefficient block
  (OMM/exponential-map with explicit Riemannian gradient).
- Share the seed with the calculator's warm-start machinery so MD-style
  trajectories amortize the seed entirely.
- PAW/USPP: needs the S-metric in the Löwdin step and the augmentation
  channels on the strain graph (`paw_stress` already has the latter).
