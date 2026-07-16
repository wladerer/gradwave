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

The clearest single predictor is system size, not the metal-versus-insulator axis.
On the RTX 3050 the end-to-end speedup grows monotonically with the cell, from 1.14
times on 2-atom Si through 1.28 times on 16 atoms to 1.39 times on 54, because the
fp32 draft only pays once the dense subspace eigensolve and the big-sphere
Hamiltonian applies dominate the per-iteration cost. A 2-atom cell is the wrong
regime to judge it. There the eigensolve is negligible and the density-build FFTs,
which stay fp64 for charge conservation, set the floor. The draft costs nothing in
accuracy at any size. On a frozen geometry the mixed and fp64 free energies agree to
1e-9 meV at every convergence threshold from 1e-7 to 1e-10, with identical iteration
counts, so the fp64 polish removes the draft error whether or not the run stops
early.

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

### Compiled XC layer (opt-in)

torch.compile is dead on the complex FFT-bound Hamiltonian apply, the entry below
still stands for that. The exchange-correlation functional is the opposite case. It
is real-valued and runs a chain of roughly thirty elementwise transcendental
operations that Inductor fuses well. Passing `compile_xc=True` to the `GradWave`
calculator, or calling `xc.enable_compile()`, routes `XCFunctional.energy_density`
through a cached compiled callable with an eager fallback.

Measured on `PBE.energy_density`, a 64³ grid, float64, 8 CPU threads. The ratio is
the reliable figure since eager and compiled are timed back to back in one process.

| path | eager | compiled | speedup |
|---|---|---|---|
| forward energy density | 1173 ms | 61 ms | 19x |
| forward + v_xc backward | 2693 ms | 171 ms | 16x |

The v_xc result is bit-accurate to eager at 3e-16, and the forward value is exact.
The win concentrates where the XC transcendental chain runs many times per SCF
iteration and is not FFT-bound, namely the PAW one-center quadrature
(`scf/paw_onsite.py`) and learned-XC training. On a plain ground-state SCF the
end-to-end gain is a few percent, because XC is a minority of runtime and its
FFT-based gradient assembly in `core/density.py` is outside the compiled kernel.

Two limits set the scope. First, the first compile traces for about a minute
whether or not it succeeds, so it pays back only over a long SCF or a training run,
never a one-shot, which is why the gate test is in the slow tier. Second,
torch.compile with aot_autograd cannot double-backward, and the `f_xc` response
kernel (dielectric, Newton, Stoner, learned-U) is exactly a double backward through
`E_xc`. Those call sites wrap their `xc.energy()` in the `xc_eager()` context
manager (`core/xc/base.py`), which forces the eager path, so response and HVP code
stays correct with `compile_xc` on, it just does not
accelerate there. Correcting the earlier report, HVP-based learned-XC training does
not benefit for the same reason, only the forward and `v_xc` legs do.

On NixOS the compiled path needs `openssl` on `PATH` for Inductor cache hashing and
`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` on GPU. When either is absent the first
call latches to eager and returns the identical result, so the flag is always safe
to leave on.

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
- **torch.compile on the Hamiltonian apply.** Inductor does not codegen complex
  operations, and the real-decomposed slice that would compile is too small next
  to the FFTs. It was tried and removed for the complex apply. The real-valued XC
  layer is a separate live win and is not covered by this line, see "Compiled XC
  layer" below.
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

## Case study, a hard PAW metal vs QE

The diamond number above is a favorable case, a norm-conserving insulator with BFGS
parity. A hard PAW metal is the other end. One-atom fcc Pt (psl kjpaw, PBE, 40/400
Ry, 12×12×12 giving 72 irreducible k, gaussian 0.2 eV) on the same asus box, QE
`pw.x` on 8 MPI ranks with k-pools against gradwave on the CPU and on the RTX 3050.

| run | hardware | iters | wall | s/iter |
|---|---|---|---|---|
| QE `pw.x` | 8 CPU cores | 7 | 3.2 s | 0.46 |
| gradwave | 8 CPU threads | 16 | 67 s | 4.2 |
| gradwave | RTX 3050 | 16 | 903 s | 56 |

The gradwave rows are on AC power; QE is the reference run. An earlier set read 118 s
on the CPU and 976 s on the GPU on battery, which had throttled the CPU (its turbo is
capped unplugged) but not the fp64-bound GPU, flattering the GPU by shrinking the CPU
baseline — see the AC-power caveat under "Measuring performance" below.

The energies agree to sub-meV: QE and gradwave both give −10167.53 eV, matching to
0.25 meV with every term within 3 meV, re-verified fresh at 6×6×6 and 12×12×12. An
earlier −10167.30 QE figure recorded here was a bad reference, not a real offset. So
this is a clean speed gap. It factors into three independent terms that multiply to
the 283 times CPU-to-GPU-vs-QE spread.

- 13.5 times, the same gradwave code on the RTX 3050 versus the CPU (903/67). Pure
  consumer-GPU fp64 tax: the card is far slower than the CPU it ships with for a
  one-atom cell that never fills it, running at 100 percent utilization while drawing
  only 25 W of its 60 W budget at a full 1942 MHz — the fp64 units saturated while the
  rest of the die idles. The GPU actively hurts here, and AC power widens this gap
  rather than closing it, because it unthrottles the CPU and cannot feed the
  arithmetic-bound GPU.
- 9 times, gradwave-CPU versus QE per iteration (4.2 / 0.46). PyTorch dispatch and a
  less-tuned 400 Ry augmentation against decades of Fortran.
- 2.3 times, gradwave takes 16 iterations to QE's 7. Its default mixing converges a
  metal slower than QE's.

So the honest per-regime picture is 1.9 times for an NC insulator relax and about 21
times for a hard PAW metal on the CPU, and the laptop GPU makes the metal case worse,
not better, until the cell grows past the fp64 crossover. Threading did not help this
small problem past 8 cores, 16 threads came out marginally slower. Run small PAW-metal
campaigns on the CPU.

### Where the PAW-metal time goes

Profiling one fcc Pt SCF (`benchmarks/pt_uspp_bench.py --profile`, 6x6x6, 67.5 s / 17
iterations on 8 CPU threads) splits the per-iteration cost as follows.

| cost | share | note |
|---|---|---|
| FFT (`ifftn`+`fftn`, mostly the wavefunction H-apply) | 34% | dense grid, dual-grid target |
| einsum (projectors + one-center) | 9% | |
| PAW one-center D-matrix via autograd `run_backward` | 5% | QE does it analytically |
| subspace `eigh` | 5% | |
| Davidson diagnostics (`linalg.cond` + `abs`) | 5% | conditioning guard |
| misc Davidson (qr, solve_triangular, norm, cat) | 7% | |

The 21 times factors as roughly 9 times per iteration and 2.3 times iteration count.
For the iteration count, the mixing scheme is the lever and the smearing kernel is not.
Sweeping fcc Pt, `johnson` converges in 13 iterations against `pulay` 17 and `broyden`
20, and gaussian, cold, and mp1 sit within one iteration at fixed scheme. The converged
free energy is bit-identical, so johnson is a free 1.3 times on a smeared metal (now the
metal-campaign default). It does not reach QE's 7 iterations, which is a starting-density
and preconditioner-quality gap. For the per-iteration 16 times, the largest single lever
is the dense-grid wavefunction FFT (34 percent). The dual grid now runs the batched
H-apply local term on the smooth `ecutwfc` box instead of the dense `ecutrho` box, exact
by the bandwidth argument (see the wisdom notes) and verified two ways, the batched path
matches the dense per-k reference to 2e-13 eV and the Pt free energy is unchanged. It
halves the FFT time on Pt for about 1.2 times, and the win grows with `ecutrho/ecutwfc`.
The density-build FFT is still dense, a further increment.

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

Direct evidence the bound is arithmetic and not clocks or power: on AC an fcc-Pt SCF
ran the RTX 3050 at 100 percent utilization while drawing only 25 W of its 60 W budget
at a full 1942 MHz, and unplugging barely changed the wall time (903 vs 976 s). A
clock- or power-limited kernel draws its whole budget; a card starved of fp64 units
saturates the few it has and idles the rest of the die, which is this trace exactly.

What would actually move it is an fp32-dominant solver schedule that drafts far
deeper and reserves fp64 for a final polish, or a datacenter-class fp64 GPU. Larger
grids and heavier bands amortize the fp64 handicap on their own, which is why the
larger norm-conserving and USPP benchmarks see real GPU wins while one-atom cells do
not. On the RTX 3050 a 16-atom Si cell already runs 1.69 times faster than the
8-core CPU at fp64, so the card is worth using once the cell reaches production size,
even though the 2-atom toy loses to the CPU on kernel-launch and transfer overhead.
The earlier impression that this GPU was hopeless came from a magnetic PAW metal that
ran 250 iterations over 27 k-points and two spins with the one-center work on the
CPU, which is iteration count and host round-trips, not a regime where the GPU
stretches its legs.

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
- **Freeze the geometry when comparing precisions or codes.** A benchmark that
  rattles the structure with a fresh random draw on each run compares different
  systems, not different methods. A per-call rattle once showed a 200 meV
  mixed-versus-fp64 energy gap that was entirely the structural difference between two
  rattles, and it vanished to 1e-9 meV the moment the perturbed geometry was built
  once and reused. Draw the structure before the loop, not inside it.
- **Benchmark on AC power.** A laptop on battery caps CPU turbo, so a CPU-vs-GPU or
  cross-code comparison taken unplugged flatters the GPU by handicapping the CPU. The
  fcc-Pt CPU point moved 118 → 67 s plugged in while the fp64-bound GPU held near
  900 s, turning an apparent 8.3× GPU deficit into the true 13.5×.
