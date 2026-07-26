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

The circularity ρ → V_eff → H_sub → λ → f → ρ is closed by a **detached**,
damped fixed-point at the base cell (orbitals held fixed, only
occupations/potential iterate). Since #129 the solve is then made **LIVE** in a
single final pass through the degeneracy-robust `RobustEigh` (see "The live
occupation solve" below): the rotation `R` and the Fermi weights `f` carry
gradient, and the objective takes the grand-potential form
`F = E − σS − μ(N − N_e)`. It is exact at the SCF fixed point — there `C`
already diagonalises `H_sub` (`R = I`) and `f` is the self-consistent Fermi
filling — so:

- **F(ε = 0) = SCF free energy** to ~1e-7 eV (LDA and PBE),
- **dF/dε = Ω·stress()** to ~1e-4 eV/Å³ (fixed occupations have no explicit
  ε-dependence, matching `postscf/stress.py`), and
- **−dF/dτ = postscf.forces()** to ~4e-7 eV/Å (live occupation/rotation
  response — the #129 anchor).

All are asserted in `test_joint_free_energy_*` / `test_metal_stress_*` on
small smeared Al cells.

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

The original draft reported a head-to-head for scheme 3 (frozen occupations,
both methods to fmax = 0.02) claiming "same basin, 0.065 Å apart, 3.1× fewer
H-applies". **That table was an artifact** — see the correction below.

### The live occupation solve (#129) — degeneracy-robust eigh, exact force

`opt/_metals.RobustEigh` is a custom `autograd.Function`: forward is the exact
Hermitian `eigh`; backward replaces the eigenvector-response denominators
`1/(λ_i−λ_j)` with the Lorentzian-broadened
`F_ij = (λ_j−λ_i)/((λ_i−λ_j)² + ε²)` — finite at exact degeneracy, exact when
`|λ_i−λ_j| ≫ ε`. Verified against `torch.linalg.eigh`'s own backward, against
`gradcheck`, and NaN-free at exact degeneracy where the bare backward is not
(`tests/unit/test_metals_eigh.py`).

`robust_subspace_occupations` uses it to keep the subspace rotation and the
Fermi occupations LIVE on the joint graph: a detached, damped (β = 0.5 — the
raw occupation/potential iteration charge-sloshes on near-degenerate Fermi
shells) inner loop converges (ρ, μ, f); then ONE live pass rebuilds `H_sub` on
the graph and re-derives (rotation, f, −σS) through `robust_eigh`. Getting the
gradient right required three consistency conditions, each found via a
concrete failure:

1. **Grand-potential form.** With `f` live in λ, `E − σS` alone carries
   `∂F/∂f_i = μ ≠ 0` and the live electron count is unconstrained — a spurious
   particle-number force `μ·∂N/∂(τ, C)` swamps the real one (observed:
   reported fmax 4e-2 while the true force was 0.35 eV/Å). The Legendre term
   `−μ(N − N_e)` is zero in value at the Fermi solution and exactly cancels
   the drift.
2. **The entire `H_sub` potential must be LIVE.** Every response channel of
   the live gradient carries the prefactor `(λ_i − ε_i^true)`, `ε_i^true` the
   band derivative of the live `joint_energy`. With a frozen inner-loop
   potential that prefactor vanishes only AT the inner fixed point and is
   first-order in the orbital error away from it — the trajectory force is
   then noise (spurious near-zeros at ~1e-3 reported fmax with 0.35 eV/Å of
   true force). The live pass therefore rebuilds `v_H + v_xc` from the live
   density (XC functional derivative taken with `create_graph`, so the kernel
   response rides the graph) and `v_loc`/projectors from the live positions.
3. **The inner loop must actually converge.** The force error is a direct
   readout of the leftover `(λ−ε)` residual: on a charge-sloshing 2×2×2 Al
   shell, 60 damped sweeps leave 4e-4 eV/Å, 200 sweeps reach ~1e-7.

With all three: at electronic self-consistency the live functional reproduces
the SCF free energy to ~5e-10 eV and `−dF/dτ` equals `postscf.forces` to
**~4e-7 eV/Å** (`test_joint_free_energy_position_gradient_matches_forces`) —
the frozen-occupation force bias is closed at the functional level.
`dF/dε = Ω·stress` is unchanged (`H_sub` lives on the base-cell reciprocal
lattice; fixed occupations have no explicit ε term).

### Descent — corrected head-to-head, honest status

**Correction.** The frozen-occupation head-to-head reported `joint 3.5753 Å vs
ref 3.5104 Å` — but the STARTING pair separation is 3.5754 Å: the old metal
descent **never moved the atoms**. Its "converged, fmax < 0.02" was a spurious
zero of the biased frozen-occupation force field at the start geometry (the
true force there is 0.35 eV/Å), and the reference itself was only converged to
fmax = 0.02. The tight (fmax = 0.005) reference relaxes to 3.5078 Å,
E(SCF) = −3362.5695 eV. The rewritten slow test asserts the displacement
actually happens.

With the live scheme the descent is real: monotone free-energy decrease, atoms
move 3.5754 → 3.528 Å (ε = σ/10, 609 closures), E(SCF@final) within 1.2e-4 Ha
of the tight nested minimum. Two optimizer pathologies and their mitigations:

- **Stall recovery.** Metal curvature turns over as the Fermi ensemble shifts;
  stale L-BFGS pairs freeze strong-Wolfe at zero step. On a stall (< 1e-9 eV
  per chunk) the driver re-canonicalizes the orbital leaves (`Z ← C/p`, fresh
  Teter reference — Z drifts in Löwdin's null directions, whose flat curvature
  poisons the memory) and restarts the optimizer; two stalls in a row end the
  cycle.
- **Broadening bias along the trajectory.** At ε = σ = 0.3 eV the Lorentzian
  suppresses the Fermi-surface rotation response by `δ²/(δ²+ε²)` (97 % of a
  δ = 0.05 eV pair), creating fake force zeros: that run "converges" at
  fmax = 1.2e-3 but 0.031 Å from the true minimum. ε = σ/10 shrinks the miss
  to 0.02 Å; the fixed-point anchors are ε-independent because the rotation
  cotangent is diagonal there.

**Negative result against the #129 success bar.** Insulator-grade agreement
(fmax < 0.005 at ~1e-3 Å) is NOT reached: away from self-consistency the
detached inner solve and the broadened backward still leave a trajectory-level
force bias of order the orbital residual, the stiff ensemble modes (curvature
~ f'/σ, untouched by the Teter preconditioner) stretch L-BFGS to 450-600+
closures, and the H-apply advantage inverts (h_equiv ≈ 133 k vs tight-ref
h_ref ≈ 50 k — **~0.4×**, vs the insulator's 7.8×). The live scheme is kept —
it is exact where the old one was silently wrong, and the descent genuinely
descends — the slow test asserts what is true, and the PR stays a draft.

## Next steps

- **Preconditioned Marzari–Vanderbilt occupation block** (the remaining scoped
  route in #129): occupation levels as explicit leaves with a metric matched
  to the coefficient block — no `eigh` on the graph, no inner fixed point, and
  the ensemble stiffness lands in a block that can be preconditioned
  analytically (`∂²F/∂η² ≈ 2w·δ̃/σ`).
- **Warm-start the inner occupation solve across closures** (carry ρ / f / μ);
  the cold uniform start pays 10-60 damped sweeps per closure.
- Curvature-aware treatment of the near-Fermi rotation modes (their stiffness
  ~ f'/σ is what stretches the closure count).
- Momentum/trust-region alternatives to L-BFGS for the coefficient block
  (OMM/exponential-map with explicit Riemannian gradient).
- Share the seed with the calculator's warm-start machinery so MD-style
  trajectories amortize the seed entirely.
- PAW/USPP: needs the S-metric in the Löwdin step and the augmentation
  channels on the strain graph (`paw_stress` already has the latter).
