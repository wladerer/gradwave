# DG-ALB build plan

Staged implementation plan for a differentiable discontinuous-Galerkin
adaptive-local-basis (DGDFT) engine in gradwave. Sequenced to front-load the one
deep uncertainty — does the DG interior-penalty assembly converge to the
plane-wave energy? — into a cheap spike before any 3D infrastructure is built.

## What the three benchmark studies already settled

- **Speed** (`bench_dgalb.py`): the ALB build is O(N) (~3.5 s/element), plane
  waves are ~N^2.4 and OOM past ~216 atoms; crossover ~150-200 atoms.
- **Global solver** (`bench_dgalb_solver.py`): dense `eigh` beats the O(N)
  density-matrix solvers until it OOMs (~1000-1500 atoms); then purification
  (not FOE, unless metallic) purely for memory. ALB build is the real bottleneck.
- **Accuracy** (`bench_dgalb_accuracy.py`): DG-ALB is a genuine (not lossy)
  representation — buffer + ~10-12 ALBs/core-atom → tens of meV and dropping; a
  no-buffer element is BC-limited at ~1 eV; M scales with the CORE.

So the basis is fast, representable, and the solver path is clear. The ONE thing
no benchmark can settle without building it is the DG interior-penalty global
assembly — that is what this plan is for.

## Architecture: reuse vs new

REUSE (gradwave already has these):
- `grids.build_fft_grid`, `grids.build_gsphere` — per-element sub-box grids/spheres
- `core.fftbox.{r_to_g, g_to_r}` — per-element kinetic in G-space
- `core.hamiltonian.HamiltonianK`, `core.batch.BatchedHamiltonian`, `build_batched`,
  `projectors_b`, `becp` — local element Hamiltonians + nonlocal PP contractions
- `solvers.davidson_batched` — local eigensolves, batched over (independent) elements
- `scf.loop.setup_system`, `effective_potentials` — per-element setup + v_eff assembly
- `scf.mixing` — density mixing for the outer loop
- `scf.implicit` — IFT fixed-point adjoint (the escape hatch for differentiability)
- `postscf.forces`/`stress`, `constants`, `dtypes`

NEW (the "zero non-volumetric machinery" gap):
- element mesh + periodic face topology
- v_eff restriction to sub-boxes; core-restriction + per-element ALB orthogonalization
- the DG face assembly (surface quadrature: jump / average-flux / penalty)
- block-sparse global H container + solve; density assembly from ALB coefficients
- the moving-(adaptive-)basis SCF driver; Pulay forces for a position-dependent basis

## De-risking spike  [DONE — CLEARED, see dgalb_spike_1d.py / FINDINGS.md]

**1D / two-element SIPG validation.** 1D periodic box, 2 elements, free electrons
(or a known potential). Build toy ALBs, assemble the SIPG kinetic + penalty form,
solve, compare eigenvalues to the analytic/plane-wave spectrum, TUNE eta and find
the coercivity threshold. Kill point 1: if SIPG can't reproduce the 1D spectrum,
stop here for the cost of a script.

## Phases

### Phase 0 — Domain decomposition & face topology  [risk: low]
Build an `ElementMesh`: partition the cell into a regular element grid; define
core boxes, extended (buffer) boxes, periodic face-adjacency (each interior face
shared by exactly two elements).
Gate: mesh tiles the cell; every interior face has exactly two owners; periodic
wraparound correct.

### Phase 1 — Extended-element solves -> ALB construction  [risk: medium]
Per-element sub-box grid/sphere; restrict global v_eff to each extended box;
local KS solve for M lowest states (batched over elements); core-restrict +
per-element orthogonalize -> ALBs.
Reuse: build_fft_grid, BatchedHamiltonian, davidson_batched, setup_system.
Gate: ALREADY PROTOTYPED — `bench_dgalb_accuracy.py` is this completeness check.
Note: the one hard blocker (per-element eigh-backward at the M-truncation) lives
here; skip autograd for the first build, add in Phase 5.

### Phase 2 — DG operator assembly  [risk: HIGH — the crux]
Diagonal blocks: per-element `int grad b_i . grad b_j` (kinetic, G-space) +
`int b_i V_eff b_j`. Off-diagonal blocks: face surface-quadrature assembling the
SIPG consistency, symmetry, and penalty terms (jumps [b], average fluxes {grad b}).
Plus the nonlocal-PP (KB) term in the ALB basis. Output: block-sparse global H
(standard eigenproblem — ALBs orthonormal by the non-overlapping core partition,
no overlap S).
New: face-topology iteration, surface quadrature, jump/average/penalty assembly.
Gate (THE project correctness gate): DG total energy & eigenvalues converge to
the plane-wave SCF reference to sub-meV as M and buffer grow, on small Si cells;
check Hermiticity, coercivity, eta sensitivity. Kill point 2.

### Phase 3 — Global solve + self-consistent loop  [risk: medium]
Solve the block-sparse global eigenproblem (dense eigh first, per the solver
study); assemble rho from ALB coefficients; close the SCF loop — new rho -> new
v_eff -> REBUILD ALBs (potential-dependent) -> reassemble -> resolve.
Reuse: scf.mixing. Subtlety: the basis moves each iteration (adaptive).
Gate: self-consistent DG-ALB energy matches plane-wave SCF to sub-meV on 8-64-atom Si.

### Phase 4 — Forces & stress  [risk: medium]
Hellmann-Feynman + Pulay (basis-motion) terms; the ALB basis moves with atoms.
Gate: forces vs finite-difference and vs plane-wave.

### Phase 5 — Differentiability  [risk: medium — the blocker, two known routes]
Make the stack autograd-clean. The DG assembly (Phase 2) is already smooth — no
new blocker. Remaining: the per-element eigh-backward (subspace-consistent /
Lorentzian-broadened), OR wrap the fixed point in `scf.implicit`'s adjoint so you
never backprop through the inner solves.
Gate: gradcheck of energy w.r.t. positions / learned-XC params vs finite-diff.

### Phase 6 — Scale-out  [risk: medium]
Swap dense global eigh -> purification (per the solver study); block-sparse
kernels; distribute element solves.
Gate: 10^3-10^4-atom runs; O(N) scaling curves.

## Critical path & kill points

- Kill point 1 (cheap): the spike. SIPG must reproduce the 1D spectrum. **CLEARED** — sub-meV at ~8-12 ALBs/atom, Hermitian to 1e-15, coercivity threshold sigma~M^2 confirmed.
- Kill point 2 (the real one): Phase 2 gate. 3D DG energy must converge to the
  plane-wave reference to sub-meV at a tractable M/buffer.
- Phases 0-2 are the WHOLE risk and are fast to reach. Phases 3-6 are known
  engineering conditional on Phase 2 passing.

## Open decisions

- Element size / buffer / M: accuracy study gives starting values (one-element
  buffer halo, ~10-40 ALBs/core-atom).
- ALB representation for face integrals (real-space values + gradients on face
  quadrature points).
- Nonlocal PP in the ALB basis — may warrant its own sub-phase.
- eta schedule (element-size-dependent; from the spike + DGDFT literature).
