# Performance

This page explains where a gradwave run spends its time, which levers move it,
and which plausible-sounding optimizations do nothing. The numbers come from
committed benchmarks on an 8-core laptop and an RTX 3050, at identical cutoff,
k-mesh, and pseudopotential to Quantum ESPRESSO. Read the [Wisdom](wisdom.md) page
for the shorter list of do and do-not rules that these measurements produced.

The one-line summary is that the small-system gap against a mature code is kernel
maturity and, on a consumer GPU, fp64 throughput. It is not an architectural
defect, and no structural rewrite of the solver moves it.

## Where the time goes

Profile before optimizing. A representative molecular SCF, triplet O₂ in a vacuum
box at a 35/280 Ry cutoff pair, spends its 53 seconds like this.

| stage | share |
|---|---|
| Davidson Hamiltonian applies | 22 s, of which 13.5 s is FFTs |
| XC potential assembly | 5.8 s |
| one-center ddd | 4 s |
| mixing | 4 s |
| density build, occupations, rest | remainder |

The FFT and the small batched linear algebra inside the eigensolver dominate.
Every lever below either removes work from those two, moves it to better hardware,
or avoids redoing it.

## What actually helps

### IBZ symmetry

Reducing the k-mesh to the irreducible wedge with G-space density symmetrization is
the largest single lever, worth 5 to 14 times depending on the point group. It is
on by default (`use_symmetry=True`) and gated by tests that check the reduced and
full-mesh energies agree. Reach for this first.

### Warm-start SCF

The ASE calculator reuses the previous step's density and orbitals as the next SCF
start. Same-position restarts drop from about 9 iterations to 2, which is what
checkpoint restarts and parameter scans want. An ionic move still costs about 8
iterations from any seed, so warm-starting helps repeated calculations at fixed or
near-fixed geometry more than it helps a single relaxation.

Warm-starting the SCF density across EOS volumes is a clear win, since the fixed
point barely moves and branch selection stays stable. Warm-starting band-path
chunks from a single previous point is the opposite. Near-degenerate seeded
subspaces stall the adaptive Davidson and the run gets 2.5 times slower, so band
paths solve cold.

### Mixed precision

Opt-in `mixed_precision=True` runs fp32 draft solves while the adaptive diagonalizer
tolerance is above 1e-5, with the subspace reduction and the S-normalization always
in fp64. The generalized subspace reduction must stay fp64 because an fp32 Cholesky
of the near-singular USPP overlap produces garbage rotations.

This is not a general win. It helps moderate-grid, many-k, smeared or spin-orbit
cases by up to about 1.45 times. It regresses fixed-occupation insulators, where
the fp32 drafts inflate the iteration count. On a consumer GPU whose fp64 runs at a
fraction of fp32 rate the option is nearly neutral on small systems, for the reason
in the GPU section below. Measure it on your workload rather than assuming it helps.

### Irreducible phonon displacements

For a Γ Hessian, `HessianSymmetry` computes only the displacement columns whose
group orbit spans every atom, then reconstructs the full Hessian. Diamond Si needs
one column of six, zincblende needs two. The reconstruction also removes the
column-to-column numerical spread, so degeneracies and the acoustic zeros come out
exact rather than approximately equal.

### Kerker preconditioning for vacuum adjoints

The density-loss adjoint takes an optional `kerker_q0` that filters the outer
residual by $q^2/(q^2 + q_0^2)$. On small dense-metal problems it slows the solve,
because Anderson mixing with history is already near-exact on so small a linear
system. On a vacuum cell it converts a long-standing stagnation floor into genuine
convergence. Triplet O₂ moves from a floored 1.4e-4 residual to a converged 1.4e-5.
Set it for vacuum systems and leave it off for validation work, where the strict
behavior is what you want.

### Reusing the response HVP

The adjoint evaluates the same Hessian-vector product at the same frozen converged
becsum on every outer iteration. Building the first-order graph once per atom and
retaining it, rather than rebuilding it per call, cuts a spin PAW HVP from 870 to
524 milliseconds with bit-identical results. The density-loss adjoint, the position
response, and the Newton step all inherit it.

## What does not help

These were built or measured and did not pay. They are here so no one spends the
time again.

- **A structural GPU rewrite for small systems.** The small-system GPU gap is fp64
  precision, not launch latency or eager-mode overhead. See the GPU section.
- **Sync-free Davidson.** Removing per-round host syncs (the convergence scalar and
  the expansion tally) with pinned async copies and event queries measures slower
  than the synchronous path at every size tested, and the delayed expansion count
  does extra work. The code path stays in the solver, default off, because a future
  fp32-deep redesign would want it, but on its own it is not a win.
- **CUDA graphs.** Capturing the real batched Hamiltonian apply replays
  bit-identically at 1.0 to 1.1 times eager speed. The kernels are already
  back-to-back, so there is no launch gap to remove.
- **torch.compile.** Inductor does not codegen complex operations, and the
  real-decomposed slice that would compile is too small next to the FFTs. It was
  tried and removed.
- **fp32 drafting on a CPU insulator.** The cast overhead beats pocketfft's fp32
  gain, so the draft is slower for that case. The mixed-precision wins are on GPU
  many-k and smeared workloads, not here.
- **Γ-point real wavefunctions.** Half-basis real algebra at Γ can at best halve the
  Hamiltonian-apply share, which caps the end-to-end gain at roughly 1.3 to 1.5
  times, for the most invasive change in the stack. Mixed precision already banks
  1.2 times on the same system at a fraction of the risk. This is deferred, not
  rejected, and worth revisiting only if Γ-only molecular workloads dominate.

## Case study, geometry relaxation vs QE

Relaxing displaced diamond with an identical pseudo, cutoff, and k-mesh in both
codes on the same cores, both landing the same minimum to 1e-4 Å, first looked like
a large deficit and turned out to be two small things plus kernel maturity.

| run | ionic steps | wall |
|---|---|---|
| QE, BFGS | 5 | 14.0 s |
| gradwave, BFGS | 3 | 47.6 s |
| gradwave, FIRE | 25 | 405 s |

Two separate factors made the original run slow.

- The optimizer default was FIRE, which took 25 steps where BFGS takes 3, an 8.5
  times penalty. The default is now `bfgs`.
- One real defect remained. The norm-conserving batched Davidson fed unnormalized
  preconditioned residual rows into orthonormalization, whose rank threshold then
  replaced near-converged rows with random jitter and wasted most expansion rounds.
  The USPP solver already had the fix. Back-porting it halved the per-iteration cost
  with an identical trajectory and closed the gap from 3.4 times to 1.9 times.

What did not turn out to be the problem is as useful as what was. QE keeps no mixer
state between ionic steps, so an early guess that it did was wrong. Forces cost 0.07
seconds per step, so the theory that the autograd backward was expensive was wrong.
QE's smaller band count for a fixed-occupation insulator is worth about 10 percent
and is a policy choice, not a defect. The remaining 1.9 times is FFT and small
batched linear algebra against decades-tuned FFTW and LAPACK, and it shrinks on GPU
and with system size.

## The GPU story is precision, not structure

The kernel-level claim verifies emphatically. On the exact hot shapes from the
laptop profile, a consumer GPU is much faster than the laptop CPU, and faster still
in single precision.

| kernel | laptop CPU c128 | 3050 c128 | 3050 c64 |
|---|---|---|---|
| batched FFT | 13.1 ms | 6.5 ms | 1.1 ms (12×) |
| batched Hermitian eigh | 4.9 ms | 2.4 ms | 0.5 ms (10×) |

Yet the same small SCF gains only 11 to 15 percent end to end on that GPU. Scaling
the k-mesh nearly 5 times did not widen the edge, which rules out a pure launch-
latency story. Three structural fixes were built and each failed to move it, listed
above.

The reason is arithmetic. A double-precision Hamiltonian apply is two c128 FFTs plus
fp64 einsums, running on a GeForce card whose fp64 executes at 1/64 the fp32 rate.
The single-precision twin kernels are 6 to 12 times faster on the same device, so
the gap is precision, not structure. The fp32 draft window, the first few SCF
iterations above diagonalizer tolerance 1e-5, is too short to matter at 9-iteration
solves, and everything after runs in crippled fp64.

What would actually move it is an fp32-dominant solver schedule that drafts far
deeper and reserves fp64 for a final polish, or a datacenter-class fp64 GPU. Larger
grids and heavier bands amortize the fp64 handicap on their own, which is why the
larger norm-conserving and USPP benchmarks see real GPU wins while one-atom cells do
not.

## Measuring performance without fooling yourself

- **Iteration counts are more trustworthy than wall time for solver-logic
  questions.** Back-to-back wall-clock deltas on a laptop are dominated by thermal
  throttling. Compare iteration counts when the question is about solver quality, and
  reserve wall time for kernel microbenchmarks run in isolation.
- **Compare the quantity the other code prints.** QE's convergence criterion is an
  energy criterion, where the error scales as the residual squared. Demanding a
  density threshold 100 to 1000 times tighter explains most of an apparent iteration
  gap before any mixer-quality difference. For smeared metals the density residual
  floors at occupation noise while the free energy is long settled, so gate on the
  energy tail.
- **Screen mixers on a linearized rig, confirm on the real SCF.** A real SCF costs
  15 to 50 minutes per mixer data point. Arnoldi on finite-difference applies of the
  true one-iteration map reduces that to milliseconds and measures the actual gain
  spectrum. The rig sees local convergence only, never basin selection, so confirm
  the winner once on a real SCF.
