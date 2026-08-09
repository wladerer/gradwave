# Plan: the response-kernel program (four moonshots on one substrate)

## Status

Proposed, not started (2026-08-09). Four speed/scaling reformulations that came out
of the round-5 moonshot brainstorm — AdjointNet, ContinuationSCF, ToeplitzEmbed,
ChaosField — turn out to be four different exploitations of a single in-repo object:
the linear-response / implicit-diff kernel in `scf/implicit.py` and
`postscf/_response.py`. This note frames them as one program with shared
infrastructure and a sequence, and states the first experiment and the kill-criterion
for each rung. It is a roadmap, not a commitment to build all four.

## The unifying observation

gradwave already assembles the exact Kohn-Sham response operator and solves against it
for its gradients. `scf/implicit.py` exposes the pieces as differentiable code:

- `apply_chi0` (`implicit.py:341`) — independent-particle response χ₀, with an
  insulator Sternheimer channel and a metal-window channel `_chi0_channel_metal`
  (`implicit.py:243`) that carries fractional Fermi-surface occupations and the δμ
  number-conservation term.
- `apply_k_hxc` (`implicit.py:375`) — the Hartree + f_xc kernel as an autograd HVP of
  E_xc, so learned XC flows through automatically.
- `solve_adjoint` (`implicit.py:411`) — already Anderson-solves `(I − K_Hxc·χ₀)u = v̄`,
  i.e. the self-consistent response linear system.

The per-band shifted linear solve underneath is `cg_sternheimer`
(`postscf/_response.py:488`), which today preconditions with a hardwired Teter filter
(`teter_b`, lines 510/521). The import contract already permits the cross-module calls
this program leans on (`gradwave.scf.implicit -> gradwave.postscf._response`,
`pyproject.toml:572`).

Each of the four ideas sharpens or reuses this kernel:

| idea | what it does to the response kernel | wall | risk |
|---|---|---|---|
| AdjointNet | replaces the Teter preconditioner in `cg_sternheimer` with a fitted M_θ | W4 (gradient cost) | low — residual-gated, SPD in the dominant regimes |
| ContinuationSCF | uses `I − λ·χ₀·K` as the continuation Jacobian; `solve_adjoint` is the corrector | W4 (SCF iters) | medium — bifurcation branch-jump on magnetic metals |
| ChaosField | keeps the IFT adjoint (schedule-independent) while making the forward async | W4 (wall-clock) | high — async amplifies metal sloshing |
| ToeplitzEmbed | reuses the self-consistent response substrate for a host↔defect Dyson update | W2 (defect size) | high — plane waves destroy the low-rank structure |

The walls: W1 fp64 tax, W2 the ~110–128-atom size cliff, W3 accuracy-as-a-chore,
W4 mixing-limited metal iterations (~2.3× QE today). None of the four touch W1 or W3.

## Why this is a program and not four tickets

The shared substrate means infrastructure built for the safe rungs is inherited by the
speculative ones:

- A good preconditioner for the shifted response operator (AdjointNet, rung 1) is
  *the same hard kernel* ContinuationSCF's Newton-Krylov corrector needs near λ→1.
- Soft-mode deflation (`scf/soft_mode.py`, PR #263 — `soft_subspace` at
  `soft_mode.py:231`, `dominant_screening_eigenvalue` at `:137`) is the near-null-mode
  machinery both ContinuationSCF's corrector and any honest ChaosField stabilizer
  require.
- The exact-Woodbury pattern in `scf/spin_precond.py` (`build_stoner_precond`,
  `.apply` at `spin_precond.py:62`) is ToeplitzEmbed's low-rank-update template and a
  candidate structured preconditioner for AdjointNet.

So the sequence below front-loads the shared, low-risk kernel work.

## Relationship to existing plans

This is disjoint from `docs/plans/fitted-preconditioners.md`, and the two should not
be conflated. That plan fits the *density-mixing* preconditioner (the outer SCF
`precond_op` hook in `scf/mixing.py:56`, i.e. an approximation to `(1 − v_c·χ₀)⁻¹` on
the density). AdjointNet fits the *inner response-solve* preconditioner inside
`cg_sternheimer` — a different operator (shifted, per-band, conduction-projected),
solving a different linear system. The moonshot report's DielKernelNet idea is the
learned-mixing preconditioner and belongs to `fitted-preconditioners.md`, not here.

## Sequence

Recommended order **3 → 4 → (7, 8)**: AdjointNet is the wedge (safe, most reuse, and
it de-risks ContinuationSCF's corrector); ContinuationSCF is the cleanest single
mechanism; ToeplitzEmbed and ChaosField are the two high-variance swings that both sit
on a hardened response kernel. Each rung is a gate — a measured negative is a result,
and the substrate it built stays for the next rung.

---

### Rung 1 — AdjointNet: fitted preconditioner for the response linear solves

**Change.** Swap the hardwired `teter_b` preconditioner in `cg_sternheimer`
(`_response.py:510/521`) for a pluggable `M_θ` — first an analytic structured form
(shifted-Teter / RPA-diagonal), then a fitted FNO-over-G if the analytic form leaves
iterations on the table. Train by unrolling the CG solve and backpropagating the
residual norm, exactly the pattern `scf/learned_precond.py` (`MultipoleKerkerPrecond`,
`learned_precond.py:65`) already uses for the mixing preconditioner.

**Why it is safe.** The solve stays exact at convergence regardless of M_θ — a bad
preconditioner costs iterations, never a wrong gradient. The two dominant regimes
(insulator; metal above the window) are SPD, so a Cholesky-factor parametrization keeps
M_θ SPD by construction.

**First experiment.** Instrument `cg_sternheimer` to log iteration counts per (band, k)
on a DFPT phonon or dielectric run for one insulator (Si) and one metal (Al or Cu).
Establish the Teter baseline. Then drop in a one-parameter shifted-Teter M(λ_shift) and
grid-search λ_shift — if even the analytic shift cuts iterations materially, the fitted
version has headroom; if not, the operator is already well-conditioned and the rung
stops here.

**Kill-criterion.** Fitted M_θ fails to beat tuned analytic shifted-Teter by ≥1.5× on
iteration count across both regimes after a fair training budget, OR the near-resonance
regime (small ε_c − ε_v) makes M_θ amplify the near-null direction and *increase*
iterations. Either is a publishable negative about the conditioning of the response
operator.

**Honest scope.** Accelerates *within* W4 (the gradient-cost side). It does not remove a
wall; it lowers the cost of every exact gradient (DFPT, inverse-design density losses).

---

### Rung 2 — ContinuationSCF: pseudo-arclength homotopy from a solvable limit

**Change.** Add an outer predictor-corrector driver in a coupling/temperature parameter
λ (Hartree+XC scaled 0→1, or smearing cooled). The corrector solves the Newton system
whose Jacobian is `I − λ·apply_chi0·apply_k_hxc` — `solve_adjoint` (`implicit.py:411`)
already Anderson-solves exactly this at λ=1, so the corrector is largely in place. Missing
pieces: the arclength/turning-point reparametrization and branch-monitoring.

**Why it might win.** Metal SCF iteration count decouples from the mixer spectral radius;
correctors are a handful of Newton-Krylov steps governed by path smoothness, not charge
sloshing. Exact at the endpoint. The converged response falls out as a byproduct — SCF and
linear-response become one object.

**First experiment.** Nonmagnetic metal with no bifurcation on the path (bulk Al). Compare
H-applies to convergence: continuation vs the current mixer, and — the real control — vs
plain Newton-Krylov on `I − χ₀·K` seeded from a SAD guess (if a good guess already lands in
the Newton basin, continuation buys nothing there). Use `dominant_screening_eigenvalue`
(`soft_mode.py:137`) to confirm the path stays away from the λ→1 near-singularity for this
case.

**Kill-criterion.** On the bifurcation-free case, continuation fails to beat the *better* of
(mixer, plain Newton-Krylov + a response preconditioner from rung 1) on H-applies. Separately,
the honest blocker — at a symmetry-breaking pitchfork (Stoner/CDW), continuation tracks the
connected higher-energy branch and converges to a non-ground-state — is a *scope* limit, not a
kill: seed a broken-symmetry configuration and continue at fixed order parameter. Document that
the method is a solver for bifurcation-free paths, not a black-box ground-state finder (this is
what the one prior electronic-structure homotopy paper, arXiv:2507.00290, also concludes).

**Honest scope.** Nonmagnetic metals and insulators, plus fixed-symmetry broken-symmetry SCF.
Not a guaranteed ground-state solver across a magnetic transition.

---

### Rung 3 — ToeplitzEmbed: host resolvent + Woodbury defect update

**Change.** Solve the primitive-cell host once (a reusable spectral resolvent), then treat a
defect/adsorbate as a localized Woodbury update on top, with the self-consistent host↔defect
Hartree coupling living in the `implicit.py` response substrate. `scf/spin_precond.py`'s exact
Woodbury inversion (`spin_precond.py:62`) is the direct template.

**The blocker, stated up front.** In a plane-wave basis ΔV = V_defect − V_host is a local
potential whose Woodbury rank equals the number of fine-grid points covering the defect region
(10³–10⁴ for a multi-atom defect), so a naive O(rank³) update is catastrophic. KKR wins only
because its muffin-tin/ℓ-truncated basis makes the impurity t-matrix genuinely low-rank; plane
waves have no such compression. The redeemable version compresses `G_host·ΔV` to genuine low
rank by randomized SVD in the Fermi/gap window or a Wannier active space (QDET's move), which
demotes it from exact-by-construction to approximate embedding needing a PW corrector.

**First experiment.** Before any solver work, *measure the rank*: for a single-vacancy Si
supercell, form `G_host·ΔV` in the gap window and compute its singular-value spectrum
(randomized SVD). If the effective rank at 1e-3 relative tolerance is a few-hundred or less,
the compressed scheme is viable and worth building; if it is thousands, the idea is
insulator-shallow-defect-only and the write-up says so.

**Kill-criterion.** Effective rank does not compress below ~O(few hundred) for a screened
insulator defect, OR the required PW corrector to reach sub-meV costs a full-cell solve anyway.
Metals are expected to stay the hard case (all Fermi-surface states within the screening
length enter the rank) and are out of scope for v1.

**Honest scope.** Screened defects/adsorbates in insulators and semiconductors; the payoff is a
*differentiable* plane-wave defect embedder with exact autograd forces on the defect region,
which no KKR/QDET code offers.

---

### Rung 4 — ChaosField: asynchronous, barrier-free forward with the exact adjoint

**Change.** Delete the SCF-index and all-k reduction barriers in the forward solve; run
bands/tiles/k-points as an asynchronous field whose fixed point is schedule-independent. The one
airtight pillar: the IFT adjoint (`solve_adjoint`) evaluates only at the converged fixed point,
so backward is schedule-invariant no matter how chaotic the forward was — the learned-XC /
inverse-design gradient stack rides along unchanged.

**The blocker, stated up front.** Async convergence needs ρ(|M|) < 1, but the metallic
K_Hxc·χ₀ sloshing eigenvalue grows with cell size and exceeds 1, and the only local stabilizer
(Kerker) is spatially nonlocal — exactly the term a real-space actor field cannot apply locally.
The global Fermi level μ (`find_fermi` over all k) is a second irreducible barrier.

**First experiment.** Cheapest possible falsification: on a load-imbalanced metal k-mesh, measure
(a) the async-critical spectral radius via `max_real_screening_eigenvalue` (`soft_mode.py:160`)
as a function of cell size — does it actually exceed 1 where we care? — and (b) the straggler
distribution across k-points (how much wall-clock is lost to barrier-synchronization today). If
(b) is small, the wall-clock prize is small and the rung is not worth its risk regardless of (a).

**Kill-criterion.** Straggler-induced barrier waste is < ~20% of SCF wall-clock on realistic
metal meshes (so there is little to reclaim), OR the async spectral radius exceeds 1 and the only
fix is a globally-synced coarse space of smallest-G Kerker modes — which reintroduces a barrier
and weakens the by-construction claim to async-fine + sync-coarse (still possibly worth it, but a
different, more modest idea).

**Honest scope.** Load-imbalanced metals with many k-points on heterogeneous nodes; the realistic
end state is async-fine + a tiny sync-coarse space, not fully barrier-free.

**Measured result (2026-08-09) — blocker CONFIRMED, part (a).** Ran the spectral-radius half on
asus: `dominant_screening_eigenvalue` (spectral radius of the raw, un-Kerkered fixed-point
Jacobian M = K_Hxc·χ₀) on Al fcc at two box sizes, PBE, Fermi-Dirac smearing.

| cell | box L | ρ(M) | λ_max^real (soft-mode margin) |
|---|---|---|---|
| Al, 1 atom | 2.86 Å | 0.42 | +0.001 |
| Al, 8 atoms | 5.73 Å | **2.27** | +0.001 |
| Si, 2 atoms | 3.84 Å | 1.06 | +0.002 |

ρ(M) grows steeply with box length (0.42 → 2.27 for a 2× cell, a 5.4× jump) and is already
2.3× over the async-convergence threshold ρ(|M|) < 1 at just 8 atoms — and ρ(|M|) ≥ ρ(M), so
the true async bound is worse still. The sub-1 value at 1 atom is the degenerate small-box limit
(large G_min ⇒ weak Hartree charge mode), not a counterexample; it pins the crossing. The
dominant mode is the negative Hartree charge-sloshing mode (hence the sign), exactly the term an
unpreconditioned local async field cannot tame — Kerker, its only stabilizer, is nonlocal. So an
async chaotic-relaxation SCF diverges on any physically-relevant metal cell and worsens with size,
by construction. The soft-mode margin (λ_max^real ≈ +0.001, i.e. 1 − λ ≈ 1) confirms no
CDW/magnetic instability is involved — this is the plain screening wall, not a special case.
Part (b) (straggler/barrier-waste) was not run and is now moot: even with large reclaimable waste,
the async formulation is non-convergent on metals without the nonlocal coarse-space fix. Net: the
rung survives only in the demoted async-fine + sync-coarse form, not as a by-construction async
cure. Bench: `benchmarks/chaosfield_spectral_radius/`.

---

## What this program explicitly does not claim

No single rung breaks all four walls. Three of the four (4, 8, and the metal side of 7) attack
W4/metals, and every one of their blockers concentrates on the same physics — the near-singular
metal dielectric — so the honest expectation is that the metal-iteration rungs survive as
scoped solvers or seed/corrector hybrids, not by-construction cures. Rung 1 is the safe,
compounding investment; it should be built first whether or not the others proceed. W1 (fp64) and
W3 (accuracy) are untouched here and belong to the representation/substrate moonshots
(DGDFT — already benchmarked in PR #264 — and WaveletMADNESS), not this program.

## Related

- `docs/plans/fitted-preconditioners.md` — the density-*mixing* preconditioner (disjoint operator;
  DielKernelNet lives there, not here).
- `scf/soft_mode.py` (PR #263) — near-null-mode deflation shared by rungs 2 and 4.
- `scf/spin_precond.py` — exact-Woodbury template for rung 3.
- `scf/learned_precond.py` — train-through-the-solver precedent for rung 1.
- Moonshot round-5 report (session artifact) — the full 13-idea ranking this program was curated from.
