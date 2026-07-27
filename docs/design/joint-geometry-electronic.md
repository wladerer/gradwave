# Joint geometry + electronic optimization (#122, productionized #123)

Status: **opt-in production for norm-conserving insulators**; every other system
falls back to the nested engine. Code: `gradwave/opt/joint.py`; entry point
`api.run_relax(inp)` with `relax.method="joint"` (dispatch + fallback in
`api._relax_joint`); tests: `tests/unit/test_joint_opt.py` (fast + standard),
`tests/integration/test_joint_vs_bfgs.py` (slow, the head-to-head).

## Production status & coverage

Joint descent is exposed through the normal relax path as an **opt-in engine**,
never a silent default. `relax.method` defaults to `"nested"` (a full SCF inside
every BFGS/FIRE step — the robust, all-formalisms route). Setting
`relax.method="joint"` routes norm-conserving insulators through `joint_relax`
and **transparently falls back to nested** on any unsupported system or on
non-convergence, so a user can always call it and trust the result.

| system | joint engine | notes |
|---|---|---|
| NC insulator, fixed cell (positions) | **works** | positions + orbitals; `fix_cell=True` |
| NC insulator, variable cell (stress) | **works** | strain + positions + orbitals; outer basis-rebuild loop |
| NC insulator, external pressure | falls back | joint minimizes E, not the enthalpy E+pV |
| USPP / PAW | falls back | generalized S-overlap not yet on the energy graph (see below) |
| spin-polarized / noncollinear | falls back | prototype is nspin=1 |
| metal / any smearing ≠ `none` | falls back | smeared Aufbau occupations reintroduce level-crossing discontinuities |
| odd electron count | falls back | not an insulator at fixed occupations |

The applicability guard is `api._joint_supported`; the fallback is logged (and
printed under `verbose`). The joint relax reports `method`, `h_applies`,
`h_seed`, `n_closures`, and basis-rebuild `n_steps` in the relax block.

**Convergence criteria match the calculator.** The force gate is `relax.fmax`
(eV/Å); under variable cell the stress gate is `fmax/Ω` [eV/Å³], because ASE's
`FrechetCellFilter` treats σ·Ω as a generalized cell force gated by the same
scalar `fmax`. The final energy/forces/stress are recomputed with one calculator
SCF at the relaxed geometry, so the reported numbers are ASE-consistent (not the
joint functional's fixed-basis value) and `last_result` is populated for
downstream error estimates. On non-convergence within `40·max_steps` closures
the engine returns to nested rather than shipping a half-relaxed geometry.

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
- **Level crossings** — avoided by construction (insulator, occupied subspace
  only, rotation-invariant energy). Smeared occupations from current Ritz
  values would make E(Z) non-smooth exactly at Aufbau-frontier crossings;
  that is the metal extension's real problem, not a bug of this prototype.
- **Cell-dependent basis** — handled by the rebuild loop; a single frozen
  sphere biases the lattice constant at 15 Ry by more than the 1e-3 Å bar.

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

## USPP/PAW gap (precise, deferred — follow-up issue)

Generalizing the joint functional to USPP/PAW is a **documented gap**, not a
half-shipped feature: `relax.method="joint"` detects a PAW/USPP pseudo and falls
back to nested. What is and is not done:

- **Done — S-orthonormalization.** `lowdin(z, zs)` takes the generalized overlap
  applied to the raw rows (`zs = Z·S`, where for USPP `S = 1 + Σ_ij q_ij
  |βᵢ⟩⟨βⱼ|`), Choleskys the Gram `Z S Zᴴ = L Lᴴ`, and returns `C = L⁻¹Z` with
  `C S Cᴴ = I` — the S-metric the generalized functional needs, spectrum-safe
  backward (unit-tested). This is the piece the prototype was missing in its
  Löwdin step.
- **Remaining (genuinely large, deferred).** `joint_energy` still assembles the
  *norm-conserving* density (`ρ = Σ |ψ|²`) and the bare kinetic+nonlocal terms.
  The generalized functional additionally needs, all live on the strain graph:
  (1) the **augmentation charge** `Σ_ij Q_ij(r) ⟨ψ|βᵢ⟩⟨βⱼ|ψ⟩` added to ρ (the
  `becsum` density, whose form factors `paw_stress` already differentiates
  w.r.t. strain — reuse that), (2) the **bare vs screened `D_ij`** split (the
  screened `D` depends on the live potential, so the nonlocal energy is no longer
  a fixed quadratic form), and (3) for PAW, the **one-center** `E_onecenter`
  contribution on the graph. That is a term-by-term re-derivation of the USPP SCF
  energy assembly (`scf/uspp_loop.py` + `postscf/paw_stress.py`) as one autograd
  closure, plus the `Z·S` projector contraction wired into `_coeffs_from_z`.
  Scoped as a follow-up (track under a `joint-uspp` issue): the NC path ships
  now; USPP joint is not blocked by orthonormalization but by the augmented
  energy assembly.
