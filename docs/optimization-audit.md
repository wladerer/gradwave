# Optimization audit: what is left, and whether the architecture needs to change

An audit of gradwave against its own `docs/manual/performance.md` and
`docs/manual/wisdom.md`, looking for levers not yet pulled. The
performance corpus is thorough, so most of the cheap optimizations are already
implemented or already specced. What remains splits into one designed-but-unbuilt
lever, one eigensolver the codebase has never tried, and a few profile-visible
micro-costs. A full architecture change is not warranted, and the reasoning for that
is at the end.

Every profile in the corpus agrees on the bottleneck: the wavefunction FFT inside the
Davidson Hamiltonian apply, then the small batched linear algebra around it, then on
a consumer GPU the fp64 arithmetic tax on both. Any real lever removes FFT work,
moves it to cheaper precision, or avoids redoing it. The items below are ordered by
value.

## 1. Ship the dual grid (designed, verified, not built)

This is the highest-value work item that already exists on paper. `wisdom.md` specs
it in full and reports it verified exact to 6e-16 relative error, and it is not in
the code.

A hard USPP/PAW pseudo runs the Davidson H-apply on the dense `ecutrho` box when the
wavefunctions only need the smooth `ecutwfc` box that holds their products. For fcc
Pt at 40/400 Ry that is a 35^3 transform where 21^3 suffices, 4.6x too many points on
the single most-called kernel. The FFT is 34% of that SCF, and the local term
`<psi_i|V|psi_j>` provably needs only the smooth part of V because any two
wavefunction G-vectors differ by at most `2*Gmax(ecutwfc)`. Running the wavefunction
transforms on the smooth box, filtering `v_eff` to it through a precomputed Miller
map, and embedding the smooth density into the dense box before adding augmentation
is the exact recipe QE uses. Expected about 1.3x on hard PAW, and it is norm-conserving-neutral
because there `ecutrho = 4*ecutwfc` is already the smooth box.

Nothing about this is speculative. It is a known, exact, specced transform waiting to
be implemented. Build it first.

## 2. The eigensolver the code has never tried: Chebyshev-filtered subspace iteration

This is the one genuinely missing idea, and it is the answer to a question the
performance page asks and leaves open.

The GPU section concludes that the small-system gap is fp64 arithmetic, that three
structural fixes each failed to move it, and that "what would actually move it is an
fp32-dominant solver schedule that drafts far deeper and reserves fp64 for a final
polish." That conclusion is correct, and it also points at the solver itself. Every
fix that was tried, sync-free copies, CUDA-graph capture, batch-width, was a patch to
block Davidson. Davidson was never itself questioned.

Block Davidson cannot be the deep-fp32 vehicle, and the reason is structural. Every
expansion round does a Rayleigh-Ritz `eigh` and a QR, and for the generalized USPP
problem the subspace reduction must stay in fp64 because an fp32 Cholesky of the
near-singular overlap produces garbage rotations (`wisdom.md`, eigensolvers). So the
fp32 window is capped at the H-apply while the diagonalizer tolerance is above 1e-5,
which the page itself notes is "too short to matter at 9-iteration solves." Davidson
interleaves a fp64 reduction between every batch of fp32 applies by construction. It
can never draft deep.

Chebyshev-filtered subspace iteration removes exactly that interleaving. One SCF
iteration becomes: estimate the spectral bounds cheaply, apply a degree-k Chebyshev
polynomial of H to the whole band block through the three-term recurrence, then do a
single Rayleigh-Ritz at the end. The filter is nothing but H-applies, no
orthogonalization and no eigh inside it, so the bulk of the FFT work moves into a
region that is precision-robust by design. Chebyshev filtering only amplifies the
wanted low-energy subspace; round-off in the amplification is removed by the final
fp64 Rayleigh-Ritz that fixes the rotation. That is the deep fp32 draft the page
wants, realized by an algorithm built for deep fp32 rather than retrofitted onto
Davidson. It captures the measured 12x c64 FFT and 10x c64 eigh advantages that mixed
precision on Davidson cannot reach.

It is also the sync-light structure the sync-free experiment was reaching for. One
Rayleigh-Ritz per SCF iteration instead of one per Davidson round means one host
readback per iteration instead of the dozen the profile counts, without the delayed
convergence bookkeeping that made the sync-free Davidson slower. This is not a novel
idea in the field. It is the standard reason GPU-first plane-wave and real-space DFT
codes (CheFSI as introduced by Zhou, Saad, and Chelikowsky, and the solvers in DFT-FE
and PARSEC) chose it, and the reason is precisely the profile gradwave has.

The caveats, since the win is regime-specific:

- In pure fp64 on the CPU, CheFSI is likely neutral to slightly worse than a
  well-tuned Davidson, because it does more total H-applies for the same accuracy. Its
  entire value is the fp32-deep-plus-GPU regime. Frame it and benchmark it as a GPU
  medium-system solver, not a general Davidson replacement, or it gets dismissed on a
  2-atom CPU run the way earlier levers were.
- It needs spectral bounds. The upper bound is a few Lanczos steps or a Gershgorin
  estimate, cheap and standard; the lower bound of the wanted window is the current
  eigenvalue estimate the code already carries.
- The generalized USPP/PAW metric is the real work. The filter has to apply S^{-1}H
  or run in the L^{-1}H L^{-dagger} basis, and the indefinite-S-at-low-cutoff trap
  still governs the final Rayleigh-Ritz. Norm-conserving is the clean first target,
  where S is the identity and the filter is bare H-applies. Prove the fp32-deep GPU
  win on NC, then extend to USPP with the S-metric.
- Degree selection replaces Davidson's per-band adaptivity. A fixed degree per SCF
  iteration, raised as the density converges, is the usual recipe and pairs naturally
  with the existing quadratic diagonalizer-tolerance schedule.

This is the item most likely to move the number the whole performance story is about.
It composes with the dual grid (fewer, cheaper filter applies) and with a fp32-deep
schedule (the filter is where the fp32 lives).

### Implementation

A working, tested solver is in `src/gradwave/solvers/chebyshev.py`, written to the
same signature as `davidson_batched` so it is a drop-in. It is wired into the NC
batched SCF as an opt-in, `scf(..., eigensolver="chebyshev")`, with Davidson the
default. On Si (LDA, 2x2x2, 15 Ry) the two solvers converge to the same free energy
bit-for-bit in the same iteration count, so the wiring is correct. The GPU fp32-deep
benchmark is also done, and it came back a no-go on the RTX 3050 (CheFSI 2.5 to 5x
slower than Davidson at every grid size that fits in 6 GB). The default stays
Davidson. See the benchmark result under "What is left to touch" below.

The module has three pieces.

- `_lanczos_bounds(h_apply, mask, steps=6)` runs a short per-k Lanczos from a random
  start and returns a rigorous spectrum bracket, `hi = max(θ(T)) + ‖f‖` and
  `lo = min(θ(T)) − ‖f‖`. Six steps suffice because only the extremes matter; the
  upper bound is essentially the highest kinetic energy on the sphere. The unit test
  confirms the bracket contains the true spectrum and is not absurdly loose.
- `_cheby_filter(h_apply, x, degree, a, b, a0)` is the scaled three-term recurrence
  from Zhou, Saad, Tiago, and Chelikowsky (J. Comput. Phys. 219, 2006). With the damp
  interval `[a, b]` mapped to `[-1, 1]` by `c = (a+b)/2`, `e = (b-a)/2`, and `a0` the
  amplification scale point: `σ1 = e/(a0−c)`, `Y = (HX − cX)(σ1/e)`, then per step
  `σ2 = 1/(2/σ1 − σ)` and `Y_new = (HY − cY)(2σ2/e) − (σσ2)X`. Per-k scalars broadcast
  over the `(nb, npw)` block. Everything below `a` is amplified, everything in `[a, b]`
  is damped.
- `chebyshev_filtered_batched(...)` is the subspace iteration: orthonormalize, then
  each round filter the Ritz block, QR, one H-apply for the Rayleigh-Ritz, one small
  `eigh`, residual check. It converges a fixed H to `tol`, exactly Davidson's contract.
  `chebyshev_filtered_batched_ms(...)` is the fp32-deep wrapper: the whole subspace
  iteration drafts in complex64 to a crossover, then polishes in fp64 from the warm
  start. Unlike the Davidson mixed-precision path, no per-round fp64 reduction caps the
  low-precision window; the filter H-applies stay fp32 throughout the draft. This is
  the composition the performance page asks for.

One implementation detail that is a real CheFSI failure mode, not a tuning knob, and
cost a test iteration to rediscover: the block must carry buffer bands. If the damp
interval starts exactly at the highest requested band, that band sits on the
amplify-damp boundary and stalls while every band below it reaches machine precision.
The solver carries a small internal buffer (two bands by default) above the nb
returned, gates convergence on the returned bands only, and lets the buffer ride the
edge loosely. The real SCF already keeps empty bands, so a caller can pass
`n_buffer=0` and widen `x0` instead.

Validation in `tests/unit/test_chebyshev.py`, all passing, on a synthetic batched
Hermitian operator with a plane-wave-like spectrum: eigenvalues match a dense `eigh`
to better than 1e-7, eigenvectors are orthonormal and satisfy the eigenequation, the
result matches `davidson_batched`, and the fp32-deep polish removes the draft error.
These isolate the eigensolver with no SCF or pseudopotential in the loop.

What is left to touch, in order.

- Wiring. Done for the NC collinear path (`scf(..., eigensolver="chebyshev")`),
  Davidson stays the default, and `tests/integration/test_scf_vs_qe.py` pins the two
  to the same energy. The noncollinear spinor twin was tried and is a clean drop-in,
  since the SpinorHamiltonian is still a standard Hermitian problem, but on a captured
  Al spinor H CheFSI ran the full 100-iteration cap without reaching 1e-8 while
  Davidson converged in 18, landing 3.8e-4 eV off. CheFSI converges slowly on the
  dense metal spinor spectrum, so with the no-go benchmark below the noncollinear path
  stays on Davidson and is not wired. The fp32-deep `chebyshev_filtered_batched_ms`
  composition is likewise not wired.
- The GPU benchmark that decides it, DONE, and it is a no-go on the RTX 3050. A fixed
  converged Hamiltonian was extracted and all four solver variants timed on the GPU.
  On a small many-k system (Si, 4x4x4 k, grid 24^3) CheFSI ran 5x slower than
  Davidson. On the larger grid it was built for (conventional Si8, 2x2x2 k, grid 35^3)
  a direct FFT probe measured the fp32 batch at 2.81 ms against fp64 at 9.44 ms, a
  3.4x fp32 FFT advantage, not the 12x the premise assumed. The fp32-deep CheFSI still
  came in 2.5x slower than fp64 Davidson (best CheFSI 14.7 s vs best Davidson 5.9 s),
  because the degree-12 filter does 2 to 3x more H-applies and the 3.4x FFT gain does
  not cover that. Both grids are below the size where the fp32 FFT advantage would
  reach 12x, and 6 GB caps how far the grid can grow, so the verdict on this card is
  that CheFSI does not beat Davidson. It stays opt-in and off by default. Revisit on a
  larger card where the grid, and with it the fp32 FFT gain, can grow.
- Degree schedule. A fixed degree of 8 to 12 is a fine start; raising it as the density
  converges pairs with the existing quadratic diagonalizer-tolerance schedule. Only
  relevant if a bigger card reopens the benchmark.
- USPP/PAW, the larger follow-on. The filter needs the S-metric, either `S^{-1}H`
  applies or the `L^{-1} H L^{-dagger}` basis, and the indefinite-S-at-low-cutoff trap
  still governs the final Rayleigh-Ritz. Do this only after the NC win is proven. The
  `chebyshev.py` docstring marks it out of scope on purpose.

## 3. Profile-visible micro-costs

Smaller, lower-risk, and each removes a named line from a committed profile.

- **DONE. Dropped `linalg.cond` from the Davidson conditioning guard.** The batched
  USPP solver computed a full condition number of the subspace overlap every round
  (`scf/uspp_batch.py`), an SVD read back to the host to branch, on top of the
  `cholesky_ex` it already ran. Probing a low-ecut Si PAW SCF (8, 10, 12 Ry, where the
  truncated-sphere S goes indefinite) showed the overlap tips into non-PD, which
  `cholesky_ex` flags with info>0, long before its condition number nears the 1e14
  trip. Over 700 cond calls, the SVD branch never fired independently and the max
  condition number observed was ~9e7. The factorization catch is the whole guard, as
  the per-k path already assumed. Removed the cond SVD and its host sync per round.
  The batched-vs-per-k equality tests (identical eigenpairs) and USPP/PAW-vs-QE
  regression still pass. This was part of the 5% "Davidson diagnostics" line in the
  PAW profile.
- **Make the one-center ddd cheaper.** The PAW profile spends 5% on the one-center
  D-matrix via an autograd `run_backward` every iteration, and notes QE does it
  analytically. The response HVP already got the `hvp_factory` treatment; the
  per-iteration SCF ddd did not. Two routes: an analytic derivative of the quadrature,
  or, now that the XC energy-density path compiles, compiling the one-center
  `energy_density` calls, which are made per-lm in an angular loop and are exactly the
  small-real-tensor case where fusion removes dispatch overhead.
- **Revisit the metal preconditioner, not the mixer.** The Pt metal takes 16
  iterations to QE's 7. Both codes start from the same superposition-of-atomic-densities
  guess, so this is not a starting-density difference in kind; the page attributes it
  to "starting-density and preconditioner quality." Johnson mixing already closed the
  scheme axis. The remaining lever is the density preconditioner itself: gradwave uses
  Kerker, QE's default couples Kerker with a local Thomas-Fermi term that damps the
  charge-sloshing modes a bare Kerker leaves. The Stoner preconditioner that was built
  targets the magnetic instability and proved too expensive per iteration; a cheap
  Kerker-plus-local-TF for ordinary non-magnetic metals is a different and lighter
  object and is the untried half of the "preconditioner quality" gap.

  The gap is metal-specific, not a PAW or spin-polarization problem. On the O2 triplet
  nspin=2 PAW reference (`o2_paw_spin_ci`), gradwave converges in 21 iterations to QE's
  20 on the identical input (gradwave Johnson mixer with the etol-and-rhotol dual
  criterion, QE plain Broyden beta=0.3 with conv_thr=1e-10 Ry, and the energies agree to
  0.6 meV with moment 2.000 muB in both). A gapped magnetic system has no charge-sloshing
  bottleneck,
  so both codes converge at the same rate. Only the metal (Pt) opens the 16-vs-7 gap, so
  the preconditioner is the right lever and spin is orthogonal to it.

## 4. Does the architecture need to change? No, and here is the test

The tempting radical answer is to leave plane waves for a real-space finite-element or
finite-difference discretization, the DFT-FE and PARSEC route, on the theory that
local stencils parallelize on GPUs better than a global FFT. That is the wrong trade
for this codebase, and the reason is what gradwave is for.

The plane-wave basis is not incidental here. It is what makes the differentiability
and validation story work: one systematic convergence parameter, exact stress from
straining the G-vectors, the entire per-milli-eV agreement with QE that the project is
built to demonstrate, and a pseudopotential and augmentation machinery that is validated
against a reference code cell by cell. A real-space rewrite discards all of that
validated surface to chase a GPU-parallelism win that CheFSI, the dual grid, and a
fp32-deep schedule already capture most of inside the plane-wave framework. The one
genuine floor those do not remove is the consumer-GPU fp64 tax, and that is a hardware
procurement decision, a datacenter-class card, not an architecture decision.

The test to apply before ever entertaining a rewrite: build CheFSI on the
norm-conserving path, run it fp32-deep on the RTX 3050 medium-cell benchmark, and see
whether it converts the measured 12x c64 kernel advantage into an end-to-end win. If it
does, the architecture is fine and the plane-wave solver simply needed the GPU-era
eigensolver. If a well-built fp32-deep CheFSI still leaves a large gap that is not the
fp64 tax, that is the first real evidence the basis itself is the constraint, and only
then is a different discretization worth costing. On the evidence in the corpus today,
it is not.

## Priority

1. Dual grid. Designed, exact, unbuilt, about 1.3x on hard PAW.
2. CheFSI on the norm-conserving path, fp32-deep, as a GPU medium-system solver. The
   item most likely to move the central GPU number. The solver is written and unit-tested
   (`solvers/chebyshev.py`, `tests/unit/test_chebyshev.py`); what remains is the SCF
   wiring at `loop.py:555` and the RTX 3050 benchmark that is also the section 4 go/no-go.
3. The `linalg.cond` guard (DONE, see section 3) and the one-center ddd. Cheap, low
   risk, each removes a named profile line.
4. Kerker-plus-local-TF metal preconditioner. The untried half of the metal
   iteration-count gap.
