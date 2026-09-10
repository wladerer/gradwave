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

## Metal control (Al fcc, Fermi-Dirac 0.1 eV, same protocol)

| tag | system | nk | nb | npw_max | box |
|---|---|---|---|---|---|
| al-small | Al-1 fcc, 8×8×8 MP | 260 | 10 | 331 | 20³ |
| al-medium | Al-4 conv, 4×4×4 MP | 36 | 27 | 1260 | 27³ |

Stage C2 = native kernel with **per-k retirement** (each k converges to tol
and drops out; the uniform batch drags every k to the slowest k's round
count). C2's trajectories match stage D exactly (identical napply, n_iter,
eigenvalues) — two independent implementations of per-k-independent
convergence agreeing bit-for-bit-in-trajectory validates both.

| config (8t) | eager B | native C | C× | C2 (retire) | **C2×** | k-pool D× |
|---|---|---|---|---|---|---|
| al-small | 1260 ms | 205 ms | 6.2× | 148 ms | **8.5×** | 0.86× |
| al-medium | 3096 ms | 1059 ms | 2.9× | 467 ms | **6.6×** | 5.4× |
| small (Si) | 309 ms | 79 ms | 3.9× | 69 ms | **4.5×** | 1.26× |
| medium (Si) | 511 ms | 278 ms | 1.8× | 271 ms | **1.9×** | 1.85× |

(al-small tight tol 1e-9: native 6.0×; al-medium tight: native 2.8×, k-pool
3.5× — max_iter-limited runs, eig round-off divergence documented above.
Si-medium native B-baseline varied 1.8–2.4× across runs; asus contention.)

Metal-specific findings:

4. **The win GROWS on metals.** More k (smearing needs dense meshes), more
   bands, more rounds — all feed the k-parallel axis. al-small's nk=260 is
   the ideal case: 6.2× batch-exact, 8.5× with retirement.
5. **Per-k retirement is worth 1.2–2.3× ON TOP of the substrate win on
   metals** (al-medium: 39 uniform rounds vs ≤21 per-k; 5472 → 3693
   applies). It costs the k-batch its uniformity — which is exactly why the
   eager batched path can't have it (the batch IS the uniformity) and the
   native per-k loop gets it for free.
6. **The Python k-pool inverts with k-granularity:** 5.4× at nk=36/npw=1260
   (big per-op work, GIL fraction small) but 0.86× — a LOSS — at
   nk=260/npw=331 (260 tiny solves serialize on dispatch). The pure-Python
   ship path is medium-cells-only; small-cell/many-k needs the native loop.
7. Retirement contract: returned bands satisfy rn ≤ tol at their OWN final
   RR (rn_max ≈ tol observed), vs the batch's over-polishing of
   already-converged k. Same convergence contract the SCF asks for, ~1e-6-eV
   trajectory-level eig differences at loose tol, well inside tol.

## Implications / ship paths

- **Medium cells (npw ≳ 1-3k, nk ≲ ~40): a ~20-line change ships 1.9-5.4×.**
  Per-k ThreadPoolExecutor around the existing `davidson_batched` (chunked
  like the existing `k_chunk` streaming, torch pinned to 1 thread inside).
  No C, no new numerics — each k converges by the identical solver, and per-k
  retirement comes free. Natural insertion: `_solve_bands` next to the
  `k_chunk` streaming path / `scf.memory` knob family.
- **Small cells and many-k metals: the full 4.5-8.5× needs out-of-GIL
  execution** — the C kernel (this probe; hardening needed: jitter path,
  DFT+U, USPP, fp32 modes), OR free-threaded CPython, OR a process pool with
  per-SCF workers. The C kernel's per-op quality can also be improved
  (FFTW_MEASURE+wisdom, real -march, fused loops) — its 1-thread deficit
  (~2.6×) bounds what better kernels would add on top.
- The GPU path is unaffected (kernels there are launched, not looped; the
  CUDA-graphs probe already showed back-to-back kernels).

Caveats: Si insulator only (metals: same solver structure, more bands/rounds
— structure argument unchanged); nk=36/8 give good 8-way balance, nk<8 would
cap the win at nk×; asus P/E hybrid cores mean "8 threads" is the measured
sweet spot, not a law.
