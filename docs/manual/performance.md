# Performance

This page explains where a gradwave calculation spends its time, what reduces it,
and which plausible-sounding optimizations do nothing. The numbers come from
committed benchmarks on an 8-core laptop and an RTX 3050, at identical cutoff,
k-mesh, and pseudopotential to Quantum ESPRESSO. Read the [Wisdom](wisdom.md) page
for the shorter list of do and do-not rules that these measurements produced.

The small-system gap against a mature code is kernel
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
Every optimization below either removes work from those two, moves it to better hardware,
or avoids redoing it.

## What helps

### A sane CPU thread default

gradwave caps intra-op CPU threads at `min(cores, 8)` on import instead of
inheriting the caller's `nproc`. The SCF is latency-bound small linear algebra
and FFTs, so fork-join BLAS across every core stalls fast cores behind slow ones
and the per-op sync cost dominates. A profile on a 22-logical-core hybrid CPU
(Core Ultra 7 155H) measured all-core execution 2 to 3 times slower than about 8
threads, with 22 threads slower than a single thread. Peak was near 8. This is a
zero-effort gain applied to every CPU run.

It is a default, applied only when you have expressed no preference, and is
overridable three ways:

- `GRADWAVE_NUM_THREADS=<n>` in the environment sets the count directly and wins
  over everything below.
- Setting `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, or `OPENBLAS_NUM_THREADS`
  yourself tells gradwave to keep its hands off; it leaves torch at whatever the
  stack configured and does not clobber your choice.
- `gradwave.set_num_threads(n)` retunes at runtime from Python.

Otherwise the auto-default stands. Reach for a higher count only if a benchmark
on your own hardware shows the crossover sits above 8 for your systems.

### IBZ symmetry

Reducing the k-mesh to the irreducible wedge with G-space density symmetrization is
the largest single source of speedup, giving 5 to 14 times depending on the point group. It is
on by default (`use_symmetry=True`) and gated by tests that check the reduced and
full-mesh energies agree. Reach for this first.

### Norm-conserving many-k: the GPU wins even on tiny cells

The "small cell runs faster on the CPU" rule below is specific to low-k PAW, where
the dense-grid FFT and the one-center work on the CPU dominate. A norm-conserving
Hamiltonian is the opposite regime: pure FFT plus batched `eigh`, batched over the
whole irreducible wedge. On the delta_gauge EOS campaign (1–2 atom primitives, PBE,
16³ meshes → 145 irreducible k-points) the RTX 3050 beat the 22-core CPU by 2.6–5.6×
on Al/Cu/Si with bit-identical energies, because the many batched k-points fill the
otherwise-idle fp64 units. So the device crossover is about *batch width*
(`n_k · n_band · grid`), not atom count. An 8-core laptop CPU was ~16× slower still
(Ge 435 s/vol vs ~8 s on the GPU) — offload to a slow CPU only for a job the GPU
cannot fit, never for throughput.

The fit is the 6 GB VRAM ceiling. The batched apply holds `n_k^IBZ · n_band · ∏nᵢ`
complex128, and the real-space grid scales as `nᵢ ∝ |aᵢ|·√E_cut`, so a big-cell
low-Z element overflows before a compact high-Z one: K (a ≈ 5.3 Å, 54³) OOMs while
Cu (a ≈ 3.6 Å, 36³) fits, despite Cu's higher cutoff. The fix is to scale the
k-mesh to cell size rather than fix a uniform N³ — K and Sr used 12³ against 16³ for
the ~3.6 Å cells, the same reciprocal-space density, which is both more correct and
fits the card. The still-missing code path is k-chunking the batched apply
(`_band_chunk` handles bands, not k); it would let the GPU take every big-grid
system at full mesh instead of falling back to the CPU.

### CUDA batched-QR CPU-offload

Found while chasing whether CUDA graphs could remove per-round kernel-launch
overhead in `davidson_batched` (see "What does not help" below, and the
`_QR_CPU_MAX_COLS` comment in `solvers/davidson.py`): they could not, because
isolating every op in a round showed the kernels were already back-to-back --
except one. `torch.linalg.qr` on the (nk, npw, cols) tall-skinny shape that
`_orthonormalize_b` uses to orthonormalize new expansion directions measured
**~3.9 ms on an RTX 3050 for a diamond-C, 50 Ry, nk=8, npw=465, cols=8 round --
bigger than the two-FFT Hamiltonian apply next to it (~2.3 ms)**. A plain
D2H + CPU LAPACK QR + H2D round trip runs the identical shape in ~0.3 ms, a
>10x win, for the same reason issue #133 CPU-offloaded the subspace `eigh`:
cuSOLVER's batched `geqrf`/`orgqr` pays a fixed per-call tax a genuinely tiny
problem cannot amortize. A sweep (nk 8-112, npw 465-2500, cols 8-64) found the
offload a clear-to-break-even win at cols<=16 (worst case 0.98x, typically
2-12x) and mixed above that, so `_qr_offload` gates on cols<=16 -- comfortably
above any nb in this repo's benchmark battery, so it fires on every ordinary
Davidson round; wider blocks (LOBPCG's stacked [X,W,P], CheFSI's buffered
block) fall through to the GPU unchanged.

`_orthonormalize_b` is shared by every batched solver (Davidson, CheFSI,
LOBPCG, the USPP/PAW generalized Davidson), so the fix lands everywhere at
once with no new flag. End to end on the same diamond-C, 50 Ry benchmark the
sync-free Davidson docstring already used as its reference point (RTX 3050):

| k-mesh | before | after | speedup |
|---|---|---|---|
| 4³ | 0.152 s/it | 0.098 s/it | 1.55x |
| 6³ | 0.234 s/it | 0.140 s/it | 1.67x |

Converged total energy is bit-identical before/after at both sizes (QR spans
the identical subspace regardless of which LAPACK implementation computes it;
Rayleigh-Ritz only depends on the span, not the basis' phase convention), and
inert on CPU (`_qr_offload` only takes the CPU branch when the input is
already on CUDA).

The offload is now gated on measured fp64 capability. The win above assumes a
card whose fp64 units are slow enough that the D2H/H2D round trip beats them,
which holds on the RTX 3050 (fp64 at about 1/64 of fp32) but breaks on a
datacenter fp64 GPU. On the 4x H100 session (issue #206,
`benchmarks/results/h100-session`) the same on/off A/B measured the offload as a
13% penalty, with Cr2O3 eskolaite converging in 16 iterations either way to an
energy identical at 1.8e-10 meV/atom, at 37.1 s with the offload against 32.7 s
without. `_qr_offload` therefore times a small fp64 GEMM against an fp32 GEMM
once per device and offloads only when the ratio exceeds 8. The RTX 3050
measures 38.3 and keeps today's behavior, while datacenter fp64
hardware measures a ratio near 1 and keeps the QR on-GPU. The environment
variable `GRADWAVE_QR_OFFLOAD` in `{on, off, auto}` forces the offload either
way or restores the microbenchmark default, so a benchmark toggles the path
without monkeypatching the module global.

### Warm-start SCF

The ASE calculator reuses the previous step's density and orbitals as the next SCF
start. Same-position restarts drop from about 9 iterations to 2, which is what
checkpoint restarts and parameter scans want. An ionic move still costs about 8
iterations from any seed, so warm-starting helps repeated calculations at fixed or
near-fixed geometry more than it helps a single relaxation.

Warm-starting the SCF density across EOS volumes clearly helps, since the fixed
point barely moves and branch selection stays stable. Warm-starting band-path
chunks from a single previous point is the opposite. Near-degenerate seeded
subspaces stall the adaptive Davidson and the calculation gets 2.5 times slower, so band
paths solve cold.

### Local-TF preconditioner for inhomogeneous cells (opt-in)

The default Kerker filter screens long-wavelength charge sloshing with a single
screening length, the right operator for a bulk metal and the wrong one for a
cell with vacuum, where a fixed screening over-damps the modes that must stay
free in the vacuum region. `precond="local_tf"` on `scf` and `scf_uspp` (following QE's
`mixing_mode='local-TF'`) lets the screening wavevector track the local density,
capped at the bare Kerker value so a homogeneous bulk is unchanged. It is
applied by a short warm-started conjugate-gradient solve, a couple of FFT pairs
per mixing step.

Measured on fcc Al slabs: the 4-layer Al(100) slab drops from 21 to 17
iterations and the 6-layer from 27 to 21, with the converged energy
bit-identical to bare Kerker (same fixed point, different route). Bulk Al is
unchanged (9 iterations either way), so the gain is specifically the
inhomogeneous cells (slabs and molecules) and grows with the vacuum fraction.
The iteration-count benchmark lives in `benchmarks/bench_precond.py`.

### Mixed precision

Opt-in `mixed_precision=True` runs fp32 draft solves while the adaptive diagonalizer
tolerance is above 1e-5, with the subspace reduction and the S-normalization always
in fp64. The generalized subspace reduction must stay fp64 because an fp32 Cholesky
of the near-singular USPP overlap produces invalid rotations.

This does not help in general, and whether it helps is a competition between two
effects. The fp32 draft speeds the Hamiltonian applies and the early subspace
eigensolves, a gain that grows with the cell, the band count, and the k-point count.
Against that, fp32 draft noise perturbs the self-consistent density on a smeared or
spin-polarized system and inflates the iteration count. On the seven-system battery,
few-k and small, the second effect dominates the metals. Mixed precision wins only on
the fixed-occupation insulators, on both this CPU and the RTX 3050 (Si 1.06 and 1.18
times, MgO 1.10 and 1.30), and regresses every metal and magnetic case (Cu 0.78 and
0.77, Al 0.77 and 0.94, FM Fe 0.79 and 0.74, AFM Cr 0.65 and 0.90). The fp32 draft
adds two SCF iterations on Fe and Cr, device-independent, landing hardest where the
systems are already the most expensive. On the consumer GPU, whose fp64 runs at a
fraction of the fp32 rate, the fp32 draft relieves more of the per-iteration cost than
it does on the CPU. The insulator wins therefore run larger on the RTX 3050 than on the
CPU, for the reason in the GPU section below. Measure it on your workload rather than assuming it helps,
and expect a metal or magnetic cell with few k-points to regress.

The metal-versus-insulator axis sets the sign, and size sets the magnitude once the
sign is positive. Among the fixed-occupation systems, where it wins, the RTX 3050
speedup grows with the cell, from 1.18 times on 2-atom Si through 1.28 times on 16
atoms to 1.39 times on 54, because the fp32 draft only pays once the dense subspace
eigensolve and the big-sphere Hamiltonian applies dominate the per-iteration cost. The
moderate-grid USPP/PAW cases with many k-points or spin-orbit reach about 1.45 times,
where the per-k eigensolve is the large share. Size scales the win but does not flip a
regression, so a large metal still loses. A 2-atom cell is the wrong regime to judge
it. There the eigensolve is negligible and the density-build FFTs, which stay fp64 for
charge conservation, set the floor. The draft costs nothing in
accuracy at any size. On a frozen geometry the mixed and fp64 free energies agree to
1e-9 meV at every convergence threshold from 1e-7 to 1e-10, with identical iteration
counts, so the fp64 polish removes the draft error whether or not the calculation stops
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

torch.compile does not help the complex FFT-bound Hamiltonian apply, the entry below
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
The gain concentrates where the XC transcendental chain runs many times per SCF
iteration and is not FFT-bound, namely the PAW one-center quadrature
(`scf/paw_onsite.py`) and learned-XC training. On a plain ground-state SCF the
end-to-end gain is a few percent, because XC is a minority of runtime and its
FFT-based gradient assembly in `core/density.py` is outside the compiled kernel.

Two limits set the scope. First, the first compile traces for about a minute
whether or not it succeeds, so it is only worthwhile over a long SCF or a training calculation,
never a single one, which is why the gate test is in the slow tier. Second,
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

These were built or measured and gave no gain. They are here so no one spends the
time again.

- **A structural GPU rewrite for small systems.** The small-system GPU gap is fp64
  precision, not launch latency or eager-mode overhead. See the GPU section.
- **Ozaki-scheme emulated-fp64 GEMM for the CUDA Toeplitz local apply (built and
  measured 2026-08-20, RTX 3050; avenue closed).** The idea: rebuild a
  near-fp64 `M @ c` from true-fp32 SGEMMs by mantissa-slicing both operands
  after a power-of-two row/column equilibration into integer-valued fp32
  slices, computing every slice-pair product with TF32-off cuBLAS SGEMM, and
  recombining in fp64 with a runtime-certified elementwise error bound —
  fp64-class accuracy from the fast fp32 units, so no fp32 draft noise
  perturbs the SCF and no crippled-fp64 polish phase remains. The accuracy
  side works exactly as designed: measured error vs. the fp64 FFT path is
  ~3e-16 relative with a certified bound of ~5e-15, SCF outer-iteration
  parity held on the full si2/al/cu/mgo battery even with the path forced on
  (d_iters = 0 everywhere, |dE| <= 2e-11 eV). The wall-clock side is a clean
  structural loss: exactness of the fp32 accumulation demands slice width
  `2t + ceil(log2 npw) <= 23`, so fp64 coverage costs p = 7-9 slices per
  operand and roughly 4p^2 ~ 256x the FLOPs of a plain-fp32 GEMM, while plain
  fp32 beats the fp64 FFT path by only ~10-20x at Toeplitz-eligible sizes —
  the emulation overspends its entire budget several times over. Measured:
  0.09-0.14x the FFT path at every battery signature (npw 64-588), so the
  per-signature timed trial declined it everywhere. The decomposition pins
  the blame on arithmetic, not implementation: the stacked batched SGEMM
  itself ran at 3-4.3 TFLOP/s (near this card's practical fp32 peak), and
  even at zero slicing/recombination cost the SGEMM alone could not clear the
  trial's 0.7 adoption margin (the eager slice/recombine glue added a 5-19 ms
  launch floor on top). Fewer slices are not an out — 2-3 legal-width slices
  cover only ~2^-12..2^-24, i.e. fp32-class noise, the documented
  extra-iterations-on-metals failure. A contributing structural fact: at the
  small npw where the Toeplitz path lives, the complex128 dense GEMM is
  closer to bandwidth- than ALU-bound (it even beat the FFT path ~2x at
  npw <= 300 on this card), so there is little crippled-fp64 tax for an
  emulation to reclaim in the first place. The negative is consumer-GPU-
  specific *by construction* and is NOT an H100 avenue either: emulation
  exists solely to dodge a crippled-fp64 ALU, and H100 fp64 is native-fast
  (the same reversal as the fp32-subspace trick above). Full implementation
  — slicing/recombination kernels with the certified bound, the three-way
  `_use_toeplitz` trial, exact-operator convergence certification through
  `davidson_batched(certify_apply=)`, tests and the apply/SCF/decomposition
  benches — lives at branch tip `f236973a9c2541414c81b1e237e447fec607a1ca`
  (`ozaki-toeplitz-cuda`, not merged) if other hardware ever wants a
  re-trial.
- **Sync-free Davidson.** Removing per-round host syncs (the convergence scalar and
  the expansion tally) with pinned async copies and event queries measures slower
  than the synchronous path at every size tested, and the delayed expansion count
  does extra work. The code path stays in the solver, default off, because a future
  fp32-deep redesign would want it, but on its own it does not help.
- **CUDA graphs.** Capturing the real batched Hamiltonian apply replays
  bit-identically at 1.0 to 1.1 times eager speed. The kernels are already
  back-to-back, so there is no launch gap to remove. Extended (2026-07-28):
  capturing a whole Davidson round's post-eigh math (Rayleigh-Ritz
  combination, residual, Teter precondition, orthonormalize, restart) on the
  sync-free skeleton -- the branch-free substrate a capture needs, since
  `torch.linalg.eigh` on CUDA is not itself capturable (a host `info` check)
  -- also replays bit-identically at 1.0x, including the pure-GPU remainder
  once both `eigh` and the QR below are excluded. Same verdict as the apply:
  no launch gap anywhere in the round on this hardware. The investigation was
  not wasted, though -- it is what found the actual QR outlier below, which is
  a real, measured, much larger win landed the same day. Building the
  per-shape graph-cache/warmup machinery this would have needed was
  abandoned once the isolated-op measurement made the null result clear.
- **PyGraph-style glue capture (measured 2026-08-06, RTX 3050).** The one thread
  the prior investigation left open: capture only the real-valued glue kernels the
  profile names -- XC `energy_density`, Hartree, density build -- outside the solver
  and between the FFTs, leaving the uncapturable Fermi solve and mixer eager (the
  PyGraph cut). Measured on the real iteration-2 glue tensors snapshotted from the NC
  loop (`benchmarks/bench_glue_capture.py`, Si 4x4x4, box 25^3, nk 36, nb 8). The
  result splits. The small non-FFT tiny-kernel work *does* reclaim its launch
  overhead -- `xc.energy_density` 1.3x (LDA) to 2.0x (PBE), `hartree_potential_r`
  1.8x to 2.8x, at or above the 1.2-1.5x estimate -- but it is ~1% of the glue. The
  density build (`density_b`, ~10.4 ms) dominates the real-valued glue on a multi-k
  cell and is FFT-compute-bound, replaying at 1.00x, so the *composite* glue capture
  is 1.00-1.01x. The reclaim is real but lands only on kernels that are a percent of
  the step, so it does not pay for the per-shape graph-cache/warmup machinery on a
  multi-k periodic cell. It would matter where `density_b` does not dominate --
  Gamma-point / few-band / small-grid (small-molecule) regimes -- which is not the
  multi-k periodic target. The vxc *potential* (an autograd backward of the same
  elementwise kernels) and the uncapturable Fermi/mixer were excluded.
- **torch.compile on the Hamiltonian apply.** Inductor does not codegen complex
  operations, and the real-decomposed slice that would compile is too small next
  to the FFTs. It was tried and removed for the complex apply. The real-valued XC
  layer is a separate live gain and is not covered by this line, see "Compiled XC
  layer" below.
- **fp32 drafting on a small metal or magnetic cell.** The fp32 draft noise perturbs
  the self-consistent density on a smeared or spin-polarized system and adds SCF
  iterations, so every metal and magnetic case in the seven-system battery regressed on
  both the CPU and the RTX 3050. The mixed-precision wins are on the fixed-occupation
  insulators and on the moderate-grid USPP/PAW many-k and spin-orbit cases. See "Mixed
  precision" above.
- **Γ-point real wavefunctions.** Half-basis real algebra at Γ can at best halve the
  Hamiltonian-apply share, which caps the end-to-end gain at roughly 1.3 to 1.5
  times, for the most invasive change in the stack. Mixed precision already gives
  1.2 times on the same system at a fraction of the risk. This is deferred, not
  rejected, and worth revisiting only if Γ-only molecular workloads dominate.
- **Widening `_QR_CPU_MAX_COLS` past 16 for large-`nb` systems.** Measured directly
  (not assumed) at a large-`nb` magnetic mineral's actual shape (`nk=13`,
  `npw=6746`, `cols` up to 64, "Case study, a large-nb magnetic mineral on GPU"
  below): CPU-offload is a clear win only through `cols=16`, already a loss by
  `cols=24`, and mixed-to-negative through `cols=60-64`. The existing cap is
  correctly conservative, not overly so, at this larger `npw` than the diamond-C
  sweep that set it covered.
- **fp32-compute for the Rayleigh-Ritz GEMMs (`s`/`x`/`hx`) in isolation from the
  rest of the SCF.** 13-18x on the raw GEMM at hematite's shape, but embedding it
  in the actual Davidson loop stalls convergence at a ~1-2e-5 residual floor
  instead of reaching `tol=1e-9` -- the same fp32-subspace failure issue #136
  documents for the generalized USPP reduction, now shown to also hold for the
  plain RR GEMMs. See "Case study, a large-nb magnetic mineral on GPU" below.
- **Streaming/chunking `v`/`hv` to avoid keeping the full basis GPU-resident for
  the Rayleigh-Ritz step.** Measured with real H2D transfers: chunked streaming
  is worse than either keeping the basis fully GPU- or fully CPU-resident, and a
  full CPU-resident redesign nets out roughly flat once every consumer of
  `v`/`hv` (build, combine, `h_apply`, the orthonormalize projection) is
  accounted for, not just the one op that looks good in isolation. Same case
  study.

## Case study, FLAPW per-iteration wall on production TiO2

The `gradwave.flapw` stack (all-electron, numpy/scipy, separate from the plane-wave
torch core) got its first dedicated perf pass on the production rutile-TiO2 fullpot
config (ecut 300, lmax 3, fullpot_lmax 4, k222+kerker 0.7, 6 atoms, npw ≈ 740).
Everything below was measured on asus from one shared warm full state,
OMP_NUM_THREADS=2, and the box was carrying 2-3 unrelated jobs throughout (noted per
number; medians of 10 iterations, plus a serial cProfile taken in a quieter window).

What worked, in profile order:

- **Batch the radial Numerov integrations and reuse their outputs.** The per-(l,
  energy) `numerov_log_np` recurrences (~2200 sequential numpy steps each) ran
  ~190 times per iteration: per l-channel in `radial_channel`, then *again* per
  (k, atom, l) in the density pass (`_us_ext`), and again in the aspherical
  integrals and LO build. Now one batched recurrence per sphere per iteration
  integrates every (l, E±h) row at once (elementwise, bit-identical), and the
  channel dict carries `u_in`/`ud_in` so every downstream consumer reuses them.
- **Cache the potential-independent per-k geometry across iterations.** The k+G
  enumeration, |Δk|/cos matrices, Legendre prefactors `P_l(cos)` (npw² × (lmax+1)),
  ball form factors, structure phases, Ylm blocks, and warp Miller-difference
  indices are pure geometry, but were rebuilt at every k every iteration.
  `_k_geometry` builds them once per k per run for the serial path (~50 MB per
  k-point at this size — the pool workers still rebuild locally, since shipping
  npw² matrices through pickle costs more than recomputing them).
- **Spawn the k-worker pool once per run, not once per iteration.** The
  spawn+re-import tax (~2 s) was written off as negligible against multi-minute
  fullpot iterations; at ~10 s/iteration it no longer is. The pool now lives on
  the run context (shared across Newton F evaluations) with explicit shutdown.

Serial (kworkers=1) per-iteration wall went 21.9 → 13.8 s in back-to-back cProfile
runs (1.59×); the contended bench medians moved 20.9 → 19.1 s with minima
15.1 → 12.8 s. The cold 22-iteration warmup at production kworkers=5 went
189 → 106 s (1.79×). A later interleaved base/tip/base/tip rerun taken while the
box was saturated (four concurrent jobs, ≥26 foreign threads on 22 cores) measured
~55 s/iteration on BOTH sides — under saturation the iteration wall is pure
scheduler contention and a 1.6× code delta is invisible, which is why the numbers
above lean on matched-load pairs (the cProfile A/B, the coldwarmup A/B) and
in-process ratios rather than cross-window medians. Correctness gate: the serial
path is **bit-identical** —
same `info["state"]` sha over 4 cold iterations and over an 11-iteration warm
fullpot bench. The pooled path is bit-identical per solve (the metamorphic pool
test) but its *trajectory* is a different ulp draw than the serial one — that was
already true before this pass (the baseline kworkers=1 and kworkers=5 benches from
the same warm state end at different shas), and this map's chaotic amplification of
ulp noise is documented in the TiO2 campaign notes.

The secular warm start (`subspace_reuse=True`) now uses an *augmented*
Rayleigh-Ritz span — last iteration's eigenvectors plus their residuals under the
current pencil, which carries the first-order eigenvector rotation the plain
previous-span projection missed (the historical fullpot-ramp blow-ups) — so it is
allowed on fullpot iterations too. On a real consecutive-iteration production
pencil the projected solve costs 14-17 ms against the ~500 ms dense solve
(28-38×). Its acceptance gate had to be rebuilt: the old `||Hc−εSc||/||Hc||`
metric is scale-degenerate for bands near the interstitial zero (measured 0.97
"relative residual" at 3.5e-3 eV true error — the gate could never engage and
every subspace iteration silently fell back to dense, verified bit-identical to
the exact run). The gate now uses the eV-scale bound `||Hc−εSc||/||Sc||` with a
1e-4 eV default — 10× under the degenerate-occupation tolerance, and measured
conservative (it reports 0.23 eV on an early re-equilibration pencil whose true
eigenvalue error is 3.5e-3 eV, so it rejects exactly the iterations it should).
With the fixed gate the warm start engages on the late, small-residual iterations
of the production run and the trajectory stays on the fixed point (final span
agrees with the all-exact run to 3e-5 eV; convergence is still only accepted on
exact iterations). An end-to-end wall A/B for this knob could not be measured
cleanly — the box was at load ~30 during that window — so the per-solve 28-38×
and the ~28% solve share of a serial iteration bound the win at ≲1.3× serial.
Exact solves remain the default (`subspace_reuse=False`): Newton's F must stay
deterministic.

Not pursued, with numbers (same pencils, warm production + cold ecut-500):

- **LOBPCG on the (H, S) pencil loses at every relevant size** — 0.3× dense at
  dim 737 and 1.2× at dim 1559, and it exits its 200-iteration cap short of
  tolerance both times even warm-started from the previous iteration's
  eigenvectors. Same verdict as the plane-wave-side LOBPCG that was removed
  after measurement; do not rebuild it FLAPW-side.
- **The hand-rolled Lanczos variants lose too** — a Python-loop
  full-reorthogonalization Lanczos measured 0.22× dense (the
  reorth-against-every-basis-vector loop dominates), and a block-seeded FOM
  converges only linearly (FIFO block processing advances the Krylov depth once
  per block, so the interior/top-occupied bands never reach tolerance). The win is
  ARPACK's compiled implicitly-restarted Lanczos, not a Python inner loop.

Now built (opt-in, default off):

- **Shift-invert secular solves (`lapw.solve_geneig_shift_invert`,
  `crystal_scf_multi(shift_invert=True)`)** wrap `scipy.sparse.linalg.eigsh`
  (ARPACK implicitly-restarted Lanczos on `(H−σS)^{-1}S`, its own internal dense LU
  of `H−σS`) with σ warmed just below the previous iteration's occupied window and
  a deterministic start vector, guarded by a one-factorization **Sylvester-inertia
  completeness certificate**: the LDL^H inertia `n_hi` at a midgap `σ_hi` must equal
  `nbands`, which (with all returned eigenvalues below `σ_hi` and a real gap)
  certifies the returned set is exactly the lowest `nbands` — a missed Γ multiplet
  copy or a mis-placed σ breaks it and forces a LOUD dense fallback. Default stays
  `shift_invert=False` so the exact dense solve keeps Newton's F deterministic and
  the pool bit-equality metamorphic test untouched. Measured A/B on real captured
  production pencils (asus, OMP=2, dense vs SI, medians): **2.29× at dim 737**
  (warm, parity 6e-13 eV, 6/6 engaged), **~2.9× at dim 1559** (cold, 3.5–4.5× on the
  uncontended iterations, parity 3e-5 eV — a 2.6e-4 eV outlier on the first
  ill-conditioned cold-staging pencil) — the win grows with dim, and a deliberately
  mis-placed σ is DECLINED by the certificate. The D5 aug6 config itself is dim~737
  (ecut300; aug-lmax raises the augmentation channels, not npw), so the dim-737 A/B
  is the D5-scale point. Harness: `experiments/autoapw/si_ab.py`.
- **The `sphere_density_multipoles_bands` hotspot (~40% of a serial fullpot
  iteration) is collapsed by a Gaunt-contraction rewrite** (`efg.py`): per-channel
  density matrices contracted against precomputed Gaunt blocks in the
  (l,m)-channel basis, no angular grid. Same exact integral of a band-limited
  integrand, ulp-equivalent to the retained angular-grid reference (worst relative
  `|Δρ_LM|` = 9.6e-16 over 288 real density-pass calls). Measured **30.8× per call**
  (86.4→2.81 ms on the production amps) and **1.90× the whole serial fullpot
  iteration** (8.53→4.48 s). Parallelizing the per-k density accumulation would
  further help kworkers>1 runs (the serial Amdahl floor).

## Case study, geometry relaxation vs QE

Relaxing displaced diamond with an identical pseudo, cutoff, and k-mesh in both
codes on the same cores, both reaching the same minimum to 1e-4 Å, first looked like
a large deficit and turned out to be two small things plus kernel maturity.

| run | ionic steps | wall |
|---|---|---|
| QE, BFGS | 5 | 14.0 s |
| gradwave, BFGS | 3 | 47.6 s |
| gradwave, FIRE | 25 | 405 s |

Two separate factors made the original calculation slow.

- The optimizer default was FIRE, which took 25 steps where BFGS takes 3, an 8.5
  times penalty. The default is now `bfgs`.
- One real defect remained. The norm-conserving batched Davidson fed unnormalized
  preconditioned residual rows into orthonormalization, whose rank threshold then
  replaced near-converged rows with random jitter and wasted most expansion rounds.
  The USPP solver already had the fix. Back-porting it halved the per-iteration cost
  with an identical trajectory and closed the gap from 3.4 times to 1.9 times.

The ruled-out causes are as informative as the real one. QE keeps no mixer
state between ionic steps, so an early guess that it did was wrong. Forces cost 0.07
seconds per step, so the theory that the autograd backward was expensive was wrong.
QE's smaller band count for a fixed-occupation insulator gives about 10 percent
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

The gradwave rows are on AC power. QE is the reference calculation. An earlier set read 118 s
on the CPU and 976 s on the GPU on battery, which had throttled the CPU (its turbo is
capped unplugged) but not the fp64-bound GPU, flattering the GPU by shrinking the CPU
baseline. See the AC-power caveat under "Measuring performance" below.

The energies agree to sub-meV: QE and gradwave both give −10167.53 eV, matching to
0.25 meV with every term within 3 meV, re-verified fresh at 6×6×6 and 12×12×12. An
earlier −10167.30 QE figure recorded here was a bad reference, not a real offset. So
this is a clean speed gap. It factors into three independent terms that multiply to
the 283 times CPU-to-GPU-vs-QE spread.

- 13.5 times, the same gradwave code on the RTX 3050 versus the CPU (903/67). Pure
  consumer-GPU fp64 tax. The card is far slower than the CPU it ships with for a
  one-atom cell that never fills it, running at 100 percent utilization while drawing
  only 25 W of its 60 W budget at a full 1942 MHz, the fp64 units saturated while the
  rest of the die idles. The GPU is slower than the CPU here, and AC power widens this gap
  rather than closing it, because it unthrottles the CPU and cannot feed the
  arithmetic-bound GPU.
- 9 times, gradwave-CPU versus QE per iteration (4.2 / 0.46). PyTorch dispatch and a
  less-tuned 400 Ry augmentation against decades of Fortran.
- 2.3 times, gradwave takes 16 iterations to QE's 7. Its default mixing converges a
  metal slower than QE's.

So the honest per-regime picture is 1.9 times for an NC insulator relax and about 21
times for a hard PAW metal on the CPU, and the laptop GPU makes the metal case worse,
not better, until the cell grows past the fp64 crossover. Threading did not help this
small problem past 8 cores, 16 threads were marginally slower. Run small PAW-metal
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
For the iteration count, the mixing scheme drives it and the smearing kernel does not.
Sweeping fcc Pt, `johnson` converges in 13 iterations against `pulay` 17 and `broyden`
20, and gaussian, cold, and mp1 agree to within one iteration at fixed scheme. The converged
free energy is bit-identical, so johnson gives 1.3 times on a smeared metal at no accuracy cost (now the
metal-campaign default). It does not reach QE's 7 iterations, which is a starting-density
and preconditioner-quality gap. For the per-iteration 9 times, the largest single contributor
is the dense-grid wavefunction FFT (34 percent). The dual grid now runs the batched
H-apply local term on the smooth `ecutwfc` box instead of the dense `ecutrho` box, exact
by the bandwidth argument (see the wisdom notes) and verified two ways, the batched path
matches the dense per-k reference to 2e-13 eV and the Pt free energy is unchanged. It
halves the FFT time on Pt for about 1.2 times, and the gain grows with `ecutrho/ecutwfc`.
The density-build FFT is still dense, a further increment.

## Case study, a large-nb magnetic mineral on GPU

The Davidson batched-QR CPU-offload fix above (diamond-C, 50 Ry, `nb` a few
bands) gave 1.55-1.67x per iteration. Re-running it on
`benchmarks/minerals` hematite (α-Fe₂O₃, 10 atoms, nspin=2, Johnson mixing,
Shubnikov-folded 4x4x4 k-mesh, `nb=60`, `npw≈6746`, RTX 3050) gave only ~1%
(593.9 -> 586.8 s over 17 iterations, `benchmarks/minerals/README.md`
"GPU" section) -- asking why is what this section answers, and it changes
the picture: the diamond-C shapes (`nk` 8-112, `npw` 465-2500, `cols` 8-64)
the fix's own sweep covered do not reach hematite's regime, and a large `nb`
changes which op dominates.

Instrumenting `davidson_batched` directly (stage-level wall-clock timing,
`torch.cuda.synchronize()`-bracketed, 3 representative SCF iterations,
`nk_irr=13`, `nspin=2` so both spin channels' Davidson calls are summed):

| stage | share of `davidson_batched` | note |
|---|---:|---|
| Hamiltonian apply (`h_apply`, two FFTs + nonlocal projector) | 32.5% | legitimate FFT/GEMM work |
| `_orthonormalize_b` (incl. its own QR) | 22.8% | mostly the QR below |
| subspace `eigh` (`_eigh_subspace`, CPU-offloaded, `n` up to 240) | 3.3% | correctly offloaded (`n <= 256`) |
| `teter_b` preconditioner | 0.6% | negligible |
| Rayleigh-Ritz subspace build + combination (`s`/`x`/`hx`, plain batched `matmul`/`einsum`, not a `cuSOLVER` factorization) | 40.8% | **new finding, see below** |

`davidson_batched` itself is 97.7% of total SCF wall time (h_apply, mixing,
XC, and the magnetic symmetrizer combined are the other 2.3%) -- consistent
with the earlier finding that the eigensolver dominates, but at a much
higher share than the small-system O₂/diamond benchmarks in this file, where
FFTs and XC are each a double-digit percentage.

**Why the QR fix barely moves the needle here.** `_orthonormalize_b`'s QR
runs at `cols = n_add`, the round's unconverged-band count, which for
`nb=60` reaches `cols=60` in early rounds -- above `_QR_CPU_MAX_COLS=16`, so
those calls correctly stay on GPU. A fresh sweep at hematite's *actual*
shape (`nk=13`, `npw=6746`, the diamond-C sweep only went up to `npw=2500`)
confirms the cap is not just conservative but right: CPU-offload is a clear
win at `cols=8` (2.96x) and a real one at `cols=16` (1.13x), but by
`cols=24` it is already a **loss** (0.69x), and stays mixed-to-negative
through `cols=60-64` (0.84-0.87x). Extending the cap to cover hematite's
`cols=60` rounds would make this system slower, not faster -- the transfer
cost of the larger `(nk, npw, cols)` tensor at this bigger `npw` erodes the
win the small-`npw` sweep found. Only the tail of a Davidson call, once most
bands have converged and `n_add` drops below 16, gets the fix's benefit,
which is why the end-to-end gain on hematite is ~1% instead of diamond-C's
55-67%.

**A new instance of the fp64-throughput limit, not a new bug.** The 40.8%
"Rayleigh-Ritz" bucket -- `s = matmul(v.conj(), hv.mT)` (the subspace-matrix
build) and the `x`/`hx` Ritz combination -- is plain batched `cuBLAS` GEMM,
not a `cuSOLVER` factorization with the host `info`-readback tax that made
QR/eigh CPU-offload a win. Isolating it at hematite's worst-case subspace
width (`nk=13`, `dim=240` -- `max_dim = 4*nb` just before a restart,
`npw=6746`) shows the RTX 3050 is *slower* than the 22-core CPU for this
shape: 0.53x on the subspace build, 0.67x on the Ritz combination. This is
the same fp64-crippled-GPU story the "GPU limit is precision, not
structure" section below documents for `eigh`/FFT kernels, just showing up
in a different op family because `nb=60` (vs. a few bands for diamond-C or
one-atom Pt) makes the subspace wide enough for its GEMMs, not just its
factorizations, to become throughput-bound rather than launch-bound.

**Why this is not a small targeted fix (and is not attempted here).**
Unlike the QR/eigh offload, which round-trips only the small `(dim, dim)` or
`(nk, npw, cols)` piece being factorized, `s`, `x`, and `hx` all read the
FULL basis `v`/`hv` (~336 MB each at this shape), which stays resident on
GPU across the whole round for `h_apply` and the next round's
`_orthonormalize_b(..., against=v)` projection. Offloading just the
subspace build would mean shipping `v`/`hv` to CPU and back on top of every
other op in the round that also needs them there -- a device-residency
redesign of the round, not a one-call swap, and its net benefit after real
transfer cost is unverified. Flagged here as a candidate for a future
investigation with its own controlled measurement, matching this file's
existing rule not to build machinery a targeted probe has not first shown
pays for itself.

Despite both findings, the GPU SCF is still 3.9x faster than the (fix-unaffected,
CPU offload is CUDA-only) CPU baseline end to end (586.8 s vs. 2274.9 s) --
the FFT-heavy `h_apply` and the correctly-offloaded moderate-width `eigh`
calls still favor the GPU enough to outweigh the large-subspace GEMM
slowdown at this system's size. The lesson generalizes the existing
"GPU limit is precision, not structure" finding rather than overturning it:
a bigger `nb` moves *which* op pays the fp64 tax (GEMM alongside
factorization), not whether the tax exists.

### Follow-up: the flagged device-residency redesign was investigated and does not pay off

The section above flagged the RR GEMM slowdown as "a candidate for a future
investigation with its own controlled measurement" rather than something to fix
on the spot. That investigation happened. The short answer is nothing beats the
status quo, for three independently measured reasons.

**Reproducing the baseline first.** A clean, uncontended rerun on the same RTX
3050 at the identical shape (`nk=13`, `dim=240`, `npw=6746`) against 8 CPU
threads (asus's Core Ultra 7 155H, the same "22-core hybrid CPU" from the
thread-default section above) partially confirms the original numbers. The
subspace build reproduces the direction and rough magnitude (GPU 329 ms vs. CPU
204 ms, 0.62x, against the earlier 0.53x), but the Ritz combination does not --
the clean isolated read gives the GPU 1.31x *faster* (164 ms vs. 215 ms), not
0.67x slower. Both GPU numbers check out against a FLOP roofline for this card
(~123 GFLOP/s effective on both ops, self-consistent with each other and with
the earlier fp64-throughput story), so the isolated measurement is trusted
here; the original combine-op number was most likely measured in-situ inside a
live round, where `v`/`hv` are the product of repeated `torch.cat` growth (a
different memory layout than a fresh contiguous tensor) running alongside the
rest of the round's queued kernels and allocator activity, not an
apples-to-apples isolated GEMM comparison. Neither number is wrong so much as
measuring slightly different things; both are recorded here rather than
picking one.

**fp32-compute, fp64-accumulate for `s`/`x`/`hx` alone.** Isolated, the GEMM
speedup is real and large: computing `s = matmul(v.conj(), hv.mT)` and the
`x`/`hx` combination in complex64 and upcasting only the *result* to
complex128 (the same precision split `mixed_precision`'s draft phase already
uses, just scoped to this one step instead of the whole solve) measures 18.7
ms and 12.7 ms against the fp64 329/164 ms above -- 13-18x on the raw GEMM. It
fails on convergence, though: reimplementing the exact `davidson_batched` loop
body with only this one substitution and running it against a synthetic
large-`nb` (`nb=60`) gapped-spectrum operator, the fp64 path converges to
`tol=1e-9` in 131-146 iterations while the fp32-RR path hits a hard ~1-2e-5
residual floor and never reaches `1e-9` even at 300 iterations, on both an
easy- and a hard-conditioning test case. This is the same failure the
`.to(torch.complex128)` upcast in `davidson_batched`'s `s` computation and
issue #136 already document for the generalized USPP subspace reduction -- an
fp32 subspace computation corrupts the Ritz rotation past what a tight SCF
tolerance can tolerate -- now confirmed to also hold for the plain
(non-generalized) RR GEMMs, not only the Cholesky-based generalized reduction
it was first found on. Scoping the precision drop to one GEMM family rather
than the whole draft phase does not dodge this; the RR combination *is* what
determines the Ritz vector, so narrowing where the noise enters does not
narrow how much it matters.

**Streaming/chunked residency.** Tested with real H2D transfers (a first
attempt with `v`/`hv` already GPU-resident was a no-op and discarded):
chunking the subspace build over `k` while streaming from CPU is worse than
either extreme at every chunk size tried (386-498 ms for chunk sizes 1/2/4/13,
pageable or pinned, against 329 ms fully GPU-resident or 198-204 ms fully
CPU-resident) -- per-transfer-call overhead dominates at this size, so finer
streaming buys nothing. There is a real wrinkle worth recording: *if* `v`/`hv`
already lived on CPU, computing the subspace build there and shipping back
only the small `(nk, dim, dim)` result (198 ms) beats the status quo's
fully-GPU-resident compute (329 ms) -- but that number only holds for this one
op. The Ritz combination is GPU-favorable by this file's own clean measurement
above (164 vs. 215 ms), so moving it to a CPU-resident design costs back
roughly what the build step would save; `h_apply` (32.5% of the round,
FFT-heavy) needs `v` on GPU and would need its own transfer; and
`_orthonormalize_b`'s `against=v` projection is a fourth `v`-consuming GEMM not
measured here at all. Summing what is known, the wins and losses roughly
cancel before any transfer/synchronization overhead or implementation
complexity is even counted. This does not overturn the residency-redesign
caution above -- it replaces "unverified" with a specific accounting that
comes out roughly flat, which is a stronger reason not to build it than the
earlier absence of a number.

**einsum vs. matmul.** No difference at this shape (329 vs. 330 ms build, 164
vs. 164 ms combine) -- PyTorch dispatches both call patterns to the same
kernel here.

No candidate beats the status quo, so `davidson_batched`'s Rayleigh-Ritz step
is unchanged. The hematite GPU SCF remains the 3.9x-faster-than-CPU result
recorded above.

**The verdict inverts on datacenter fp64 (H100).** Everything above is scoped
to the RTX 3050. The 4x H100 session (issue #206) re-ran the same hematite-shape
microbenchmark (`nk=13`, `dim=240`, `npw=6746`) on real fp64 hardware, and the
finding reverses on both counts. The fp64 GPU GEMM measures 1.26 ms against 139.6
ms for the CPU-resident path, so the CPU offload the 3050 profile flirted with is
a 100x loss here, not a candidate. The fp32-compute variant measures 1.16 ms, no
margin over the fp64 tensor cores, so the fp32-subspace trick that would have
dodged the consumer card's fp64 tax gains nothing on a card that has none. On a
datacenter card the Rayleigh-Ritz step belongs on the GPU in fp64 and the whole
offload/precision-split question is moot. The consumer-card scope of the sections
above holds unchanged. Benchmark per device class, do not port a GPU tuning
verdict across it, and see the QR-offload note above for the gated version of the
same lesson.

## The GPU limit is precision, not structure

The kernel-level claim verifies. On the exact hot shapes from the
laptop profile, a consumer GPU is much faster than the laptop CPU, and faster still
in single precision.

| kernel | laptop CPU c128 | 3050 c128 | 3050 c64 |
|---|---|---|---|
| batched FFT | 13.1 ms | 6.5 ms | 1.1 ms (12×) |
| batched Hermitian eigh | 4.9 ms | 2.4 ms | 0.5 ms (10×) |

Yet the same small SCF gains only 11 to 15 percent end to end on that GPU. Scaling
the k-mesh nearly 5 times did not widen the speedup, which rules out a pure launch-
latency story. Three structural fixes were built and each failed to move it, listed
above.

The reason is arithmetic. A double-precision Hamiltonian apply is two c128 FFTs plus
fp64 einsums, running on a GeForce card whose fp64 executes at 1/64 the fp32 rate.
The single-precision twin kernels are 6 to 12 times faster on the same device, so
the gap is precision, not structure. The fp32 draft window, the first few SCF
iterations above diagonalizer tolerance 1e-5, is too short to close the deficit at 9-iteration
solves, and everything after runs in crippled fp64.

Direct evidence the bound is arithmetic and not clocks or power: on AC an fcc-Pt SCF
ran the RTX 3050 at 100 percent utilization while drawing only 25 W of its 60 W budget
at a full 1942 MHz, and unplugging barely changed the wall time (903 vs 976 s). A
clock- or power-limited kernel draws its whole budget. A card with few fp64 units saturates them and leaves the rest of the die idle, which is this trace exactly.

What would move it is an fp32-dominant solver schedule that drafts far
deeper and reserves fp64 for a final polish, or a datacenter-class fp64 GPU. Larger
grids and heavier bands amortize the fp64 handicap on their own, which is why the
larger norm-conserving and USPP benchmarks run faster on the GPU while one-atom cells do
not. On the RTX 3050 a 16-atom Si cell already runs 1.69 times faster than the
8-core CPU at fp64, so the card is worth using once the cell reaches production size,
even though the 2-atom cell is slower than the CPU on kernel-launch and transfer overhead.
The earlier impression that this GPU was hopeless came from a magnetic PAW metal that
ran 250 iterations over 27 k-points and two spins with the one-center work on the
CPU, which is iteration count and host round-trips, not a regime that uses the GPU's
throughput.

### Confirmed on a datacenter fp64 GPU (A100)

The prediction held. On an NVIDIA A100-SXM4-40GB (UCLA Hoffman2), a double-precision
GEMM clocks ~4.9 TFLOP/s, roughly 35× the RTX 3050's fp64, and a constrained
non-collinear bcc-Fe spin-spiral point (2-atom cell, 60 Ry, 3×3×3, 24 bands) runs in
97 s against ~516 s on the 22-core asus CPU at 16 threads, about 5×. The GPU energy
bit-matches the CPU (−6430.0154 eV), so the fp64 path is correct, not merely fast,
and the 40 GB lifts the 6 GB grid ceiling that capped the 3050. This is the
datacenter-class fp64 card the section predicted would change the result, and it is
what makes the spin-Hamiltonian and MAE work tractable at useful cell sizes. The
three-SCF Fe exchange benchmark ran there in minutes. One caveat: the *CPU cores* of a shared GPU node can be far slower than a dedicated CPU box
(a single Fe SCF on 8 such cores ran past a one-hour walltime), so benchmark the GPU
against a real CPU reference, never the GPU node's own cores.

### Measured on a datacenter fp64 GPU (4x H100)

The A100 result held on a full battery. A four-GPU H100 SXM session (vast.ai, issue #206,
data in `benchmarks/results/h100-session/`) ran the production-size cases the RTX 3050
sections could only bound. The speedups below are against the recorded CPU and 3050
baselines, all at bit-matching energies.

| case | RTX 3050 / CPU | H100 | speedup |
|---|---|---|---|
| Cr2O3 eskolaite (magnetic USPP) | 1128 s (asus CPU) | 46.8 s | 24x vs CPU |
| Fe2O3 hematite (magnetic USPP) | 2275 s CPU, 3050 slower | 39.4 s | 58x vs CPU, 15x vs 3050 |
| delta-gauge point (many-k 1-atom) | 14 to 97 s/vol (3050) | 0.3 to 0.4 s/vol | 35 to 240x |
| PBE0 hybrid Si2 to Si16 | -- | seconds/SCF, 0.33 GB | 4.6x vs 32-thread CPU |
| Si-216 memory ceiling | OOM past 128 atoms (6 GB) | 29.4 GB peak, 3.9 s/iter | ~350 atoms in 80 GB |

Iteration counts are identical to the CPU and 3050 runs, so these are pure
kernel-throughput gains, not solver-logic differences. The eskolaite and hematite rows are
the magnetic USPP minerals that ran in tens of minutes on the 3050. The memory row removes
the 128-atom cliff the "GPU limit is precision, not structure" section documents, and the
~350-atom projection is what makes the amorphous glass models for the kappa_min plan
tractable on one card.

Two limits stay in place on this hardware. ISDF is not wired into the hybrid SCF (the
driver hard-codes a direct Fock build plus ACE per step, see
`docs/manual/hybrid-functionals.md`), so the hybrid speedup is on the direct path. And a
metallic 16-atom 4x4-mesh slab PBE0 timed out at 600 s, since the full-BZ Fock build over a
metal is the cost wall regardless of card.

The solver A/B holds here too. Chebyshev-filtered subspace iteration is 2x slower than
davidson at 10 atoms and 13x slower at Si-128 (15.316 vs 1.165 s/iter, identical energy),
worsening with size and with no crossover in 10 to 216 atoms, so davidson stays the
unconditional default on datacenter fp64 as well as on the consumer card.

**Multi-GPU.** The k-point-sharded SCF (`docs/manual/distributed.md`) runs unmodified
across the four devices via Gloo host staging, all ranks bit-identical to 1e-11. Toy cells
are slower than one GPU (startup- and collective-bound), but a real-scale Si-16 6x6x6 on 2
ranks reaches 3.2 s/iter against about 16.8 s/iter amortized single-rank, so the sharding
pays once per-k work dominates the collectives. Large multi-rank runs are blocked by issue
#216, where the post-SCF result gather deadlocks on large CUDA payloads. Use it for k-heavy
systems and watch that issue before relying on it in production.

### Size-scaling crossover, autograd, and SOC (single H100)

A later single-H100 SXM session (vast.ai, 80 GB, 192 cores) pinned down three things
the earlier batteries left implicit: *where* the GPU crossover is, and that the two
capabilities most associated with gradwave — autograd through the SCF, and the
spinor SOC path — win at least as hard as the plain SCF.

**The crossover is cell size, and the card is compute-bound above it.** A Γ-only
fcc-Al SCF swept from 4 to 256 atoms, GPU against the CPU's best thread count (8):

| atoms | npw | GPU | CPU 8t | speedup | GPU %compute |
|---|---|---|---|---|---|
| 4 | 1,237 | 0.76 s | 0.40 s | 0.5× (GPU loses) | 48% |
| 32 | 9,939 | 0.81 s | 9.75 s | 12× | 72% |
| 108 | 33,401 | 2.87 s | 152 s | 53× | 74% |
| 256 | 79,597 | 18.1 s | 1007 s | 56× | 96% |

Below ~a few tens of atoms the GPU loses to launch overhead; above it the gap opens
to 50–56× and the fraction of wall spent in real compute kernels climbs to 96%. That
last number matters: the small cell is the only launch-bound point, so fusing many
small ops into fewer kernels (spin-batching the two spin channels into one Davidson,
folding configs onto a batch axis) has nothing to reclaim on a saturated card. A
spin-batched Davidson A/B confirmed it — 0.93–0.95× on the isolated kernel and a flat
full-SCF wall even where it cut iterations — so the op-fusion family is a dead lever
on the H100, and the datacenter payoff is throughput at size, not fusion.

**Autograd through the SCF wins as hard as the SCF.** The response-carrying gradient
(`scf.implicit.density_loss_param_grads`: an Anderson-accelerated adjoint solve plus
an f_xc double-backward) had never been timed at scale on a full-fp64 GPU. On diamond
Si it runs 11× faster than the 8-thread CPU at 16 atoms and 50× at 54, and the
*backward is cheaper than the forward* (adjoint 1.8 s vs SCF 4.0 s at 54 atoms), at
2.5 GB peak. Differentiable DFT is not a GPU tax here — it is the same throughput win
as the forward solve.

**The spinor SOC path is the most GPU-favorable, and improves with k.** A
non-collinear fcc-Ni SCF with a fully-relativistic pseudopotential (the spinor basis
doubles the plane-wave axis and the j-resolved projectors act on it) ran 4.8× faster
than the CPU at a 4×4×4 mesh and 6.0× at 6×6×6 — the win grows with k-density because
the doubled basis is exactly the dense-linear-algebra regime the card wants. The
denser mesh also stabilized the moment (0.607 μB, versus a spurious collapse at the
coarse mesh), so the fast run is the physical one.

Iteration counts match the CPU at every point above, so all of these are
kernel-throughput gains, not solver-logic differences — the same reading the A100 and
4×H100 sections reached, now with the crossover located and the autograd/SOC paths
measured.

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
  true one-iteration map reduces that to milliseconds and measures the gain
  spectrum. The rig sees local convergence only, never basin selection, so confirm
  the winner once on a real SCF.
- **Freeze the geometry when comparing precisions or codes.** A benchmark that
  rattles the structure with a fresh random draw on each calculation compares different
  systems, not different methods. A per-call rattle once showed a 200 meV
  mixed-versus-fp64 energy gap that was entirely the structural difference between two
  rattles, and it vanished to 1e-9 meV the moment the perturbed geometry was built
  once and reused. Draw the structure before the loop, not inside it.
- **Benchmark on AC power.** A laptop on battery caps CPU turbo, so a CPU-vs-GPU or
  cross-code comparison taken unplugged flatters the GPU by handicapping the CPU. The
  fcc-Pt CPU point moved 118 → 67 s plugged in while the fp64-bound GPU held near
  900 s, turning an apparent 8.3× GPU deficit into the true 13.5×.
