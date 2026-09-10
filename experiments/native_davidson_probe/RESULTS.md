# Native Davidson probe — results (2026-09-09, asus, 8 threads)

**Question.** The small-cell SCF wall was attributed to ~48% eager-mode
dispatch glue. Does that glue convert to wall-clock when the `no_grad` inner
Davidson loop leaves PyTorch entirely (C: FFTW + CBLAS + LAPACKE, zero
framework dispatch)?

**Setup.** Real mid-SCF warm solve states captured from the production NC/PBE
SCF (capture_state.py, iteration 4, `GRADWAVE_TOEPLITZ=off` so both paths use
the FFT local term):

| tag | system | nk | nb | npw_max | box | nproj |
|---|---|---|---|---|---|---|
| small | Si-2 fcc, 4×4×4 MP | 36 | 8 | 754 | 25³ | 16 |
| medium | Si-8 conv, 2×2×2 MP | 8 | 20 | 3016 | 35³ | 16 |

Same state replayed through (B) production `davidson_batched`, (C) the native
C kernel (same algorithm incl. restart drift-repair), (D) the SAME eager
solver split per-k over a Python thread pool (torch pinned to 1 thread).
Agreement gate for C: identical n_iter and band·k H-apply tally, eigenvalues
to ~1e-13 (met everywhere except tight-tol medium, which hits max_iter=40 and
diverges by BLAS round-off late — eig still agrees to 1.6e-11; the reference
itself is BLAS-round-off-dependent there, same reason the solver A/B tests
assert eigenvalue agreement, not iteration counts).

## Numbers (best of reps, warm)

| config | eager B | native C | speedup | k-pool D | D speedup |
|---|---|---|---|---|---|
| small, 8t, captured tol 5.5e-5 | 299–322 ms | 70 ms | **4.2–4.5×** | 221 ms | 1.35× |
| medium, 8t, captured tol 2.2e-4 | 505–539 ms | 221 ms | **2.3–2.4×** | 264 ms | 1.92× |
| small, 8t, tol 1e-9 | 876 ms | 225 ms | **3.9×** | — | — |
| medium, 8t, tol 1e-9 | 2051 ms | 933 ms | **2.2×** | — | — |
| small, 1 thread | 306 ms | 790 ms | 0.39× | — | — |
| medium, 1 thread | 506 ms | 1416 ms | 0.36× | — | — |

## The three findings

1. **The pure-glue hypothesis is REFUTED.** Single-threaded, the native C is
   0.36–0.39× — torch's per-op kernels (pocketfft, bundled BLAS) beat naive C
   loops + FFTW_ESTIMATE. Removing dispatch overhead alone buys nothing.

2. **The real wall: eager gets ZERO thread scaling at these sizes.** Eager
   8-thread == eager 1-thread (306 vs 299–322 ms small; 506 vs 505–539 ms
   medium). Batched CPU `eigh`/`qr` loop LAPACK serially, and the many small
   ops between never engage intra-op parallelism — the whole small-cell
   eigensolve runs one-core-equivalent on a 22-core box. The measured "~48%
   glue" was this in disguise. The k axis is embarrassingly parallel and
   completely unexploited on CPU.

3. **Exploiting k-parallelism is worth 2.2–4.5×, bit-exact.** The native
   kernel (per-k problems distributed across cores, OpenBLAS pinned serial,
   FFTs parallel over (k, band)) delivers it despite per-op kernels ~2.6×
   worse than torch's. A pure-Python thread pool over per-k chunks of the
   SHIPPED solver captures most of it at medium size (1.92× of 2.29×; per-op
   kernels big enough that the GIL fraction shrinks) but is GIL-bound at
   small size (1.35× of 4.23×).

Bonus measured number: per-k convergence retirement (each k stops at its own
tol instead of riding the uniform batch) saves ~9% of H-applies (2052→1871
small, 984→896 medium) — the physics-blind "per-k retirement" survivor,
quantified.

## Implications / ship paths

- **Medium cells (npw ≳ 3k): a ~20-line change ships ~1.9×.** Per-k
  ThreadPoolExecutor around the existing `davidson_batched` (chunked like the
  existing `k_chunk` streaming, torch pinned to 1 thread inside). No C, no
  new numerics — each k converges by the identical solver. Natural insertion:
  `_solve_bands` next to the `k_chunk` streaming path.
- **Small cells: the full 4× needs out-of-GIL execution** — the C kernel
  (this probe, needs hardening: jitter path, DFT+U, USPP, fp32 modes), OR
  free-threaded CPython, OR a process pool with per-SCF workers. The C
  kernel's per-op quality can also be improved (FFTW_MEASURE+wisdom, real
  -march, fused loops) — its 1-thread deficit (2.6×) bounds what better
  kernels would add on top of 4.5×.
- The GPU path is unaffected (kernels there are launched, not looped; the
  CUDA-graphs probe already showed back-to-back kernels).

Caveats: Si insulator only (metals: same solver structure, more bands/rounds
— structure argument unchanged); nk=36/8 give good 8-way balance, nk<8 would
cap the win at nk×; asus P/E hybrid cores mean "8 threads" is the measured
sweet spot, not a law.
