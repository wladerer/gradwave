# GPU-resident SCF on the RTX 3050 + Ozaki-int8 RR-GEMM gate — research probe

Follow-on to `experiments/fp64_emu_fft/RESULTS.md` (the FFT-emulation NO-GO). Question here:
(A) what does a fully GPU-resident fp64 SCF actually cost on the 3050 and where does its wall
go; (B) does Ozaki-int8 emulation of the complex fp64 GEMMs (the RR/subspace step — the
compute-bound part the FFT probe redirected to) clear an Amdahl gate of ≥1.3× end-to-end
**against the best CPU path (davidson-native)**.

**Verdict: Phase B NO-GO at the gate (best-case composed 0.72× on Al-4, 0.97× on Si-8 —
both below 1.3×). No prototype built, per protocol.** The GPU-resident SCF itself is
2.0× / 1.65× *slower* than CPU davidson-native today, and even free GEMMs (S=∞) leave it at
0.90× / 1.35× — the honest emulation speedup bound is ≤3.2×, giving 0.72–0.97×.

All numbers measured on asus (RTX 3050 6GB Laptop, GA107 cc 8.6, torch 2.12.1+cu130;
CPU arms 8 threads), under `/tmp/BENCH_LOCK.fftemu`, load < 0.1, 2026-09-10. Identical
free energy across every arm of each case (all digits printed). Scripts in this directory
(`profile_scf.py`, `eigh_micro.py`).

## Phase A — GPU-resident fp64 SCF: e2e + wall decomposition

Cases (same-harness, sym on, etol 1e-8 / rhotol 1e-7, 10 iters each):
- **al4** — Al conv cubic a=4.05, 4 atoms, PBE, 40 Ry, 4³ MP → nk=10, npw=1935,
  grid 32³, nb=24, fermi-dirac 0.1 eV. (The "Al-4 conv 4³ FD" system; my config's
  absolute times differ from the perf-smash record's 34.0/7.5 s — different
  nbands/symmetry details — so the same-harness arms below are the comparison basis.)
- **si8** — diamond-Si conv a=5.43, 8 atoms, LDA, 30 Ry, 2³ MP → nk=4, npw=2969, grid 35³.

### e2e SCF wall (clean runs, no profiler)

| arm | al4 | si8 |
|---|---|---|
| CPU eager, 8t | 8.04 s | 5.84 s |
| **CPU davidson-native, 8t (best CPU)** | **2.91 s** | **3.85 s** |
| GPU-resident fp64 (eager batched Davidson) | 5.77 s | 6.36 s |
| GPU vs best CPU | **0.50× (2.0× slower)** | **0.61× (1.65× slower)** |
| peak CUDA mem | 0.67 GiB | 0.38 GiB |

GPU-resident does beat CPU *eager* on al4 (1.39×) and roughly ties it on si8 (0.92×) — but
eager is the strawman; davidson-native is the shipped best CPU path.

### GPU wall decomposition (torch.profiler, CUDA kernel-only rows)

Method note: profiler op-rows (`aten::bmm`, `aten::_fft_c2c`, …) double-count their child
kernels (category sum 9.88 s > 8.64 s profiled wall), so shares use **kernel events only**;
they cross-check (fft kernels 0.926 s ≡ `aten::_fft_c2c` 0.926 s) and sum to ≈ the
unprofiled wall (al4: ≈5.9 s vs 5.77 s).

| component (kernels) | al4 | share | si8 | share |
|---|---|---|---|---|
| complex fp64 GEMM (cutlass `z884gemm` = bmm/mm: RR build, Ritz combines, projections) | 2.54 s | **44%** | 3.53 s | **55%** |
| QR (`geqr2` + children via `linalg_qr`) | ~0.98 s | 17% | ~0.91 s | 14% |
| FFT (cuFFT `regular_fft`/`vector_fft`) | 0.93 s | 16% | 0.91 s | 14% |
| eigh (cuSOLVER) | 0.06 s | 1% | 0.03 s | <1% |
| elementwise/copy/reduce/mul | ~1.3 s | 22% | ~1.0 s | 16% |

So the measured headline: **GPU-resident fp64 SCF is 0.50–0.61× vs CPU-native today, and its
wall is ~44–55% fp64 complex GEMM, ~16% FFT, ~14–17% QR, ~1% eigh, ~20% elementwise glue.**
The FFT-probe's premise inversion holds here too: FFT is a minor term; the fp64 GEMMs are
the single biggest bucket — but they are less than half the wall, which caps any GEMM-only
fix hard.

### Third angle (measured, one microbench, not built further): cuSOLVER zheevd

| dim | GPU zheevd | CPU 8t | GPU eff GFLOP/s |
|---|---|---|---|
| 64 | 5.39 ms | 0.41 ms | 0.4 |
| 128 | 8.81 ms | 0.99 ms | 2.1 |
| 256 | 13.9 ms | 3.80 ms | 10.9 |
| 520 | 51.6 ms | 19.7 ms | 24.5 |
| 1024 | 219 ms | 125 ms | 44.1 |

At Davidson subspace dims (≤4·nb ≈ 64–128) zheevd is **latency-bound** — flat ~5–9 ms
regardless of size, 0.4–2 GFLOP/s, far below both the 130 GFLOP/s fp64-ALU roof and the
memory roof; CPU is 9–13× faster there. Neither fp64-ALU-bound nor memory-bound at our
dims, and at 0.6–1% of SCF wall it is no lever in either direction.

## Phase B — Ozaki-int8 emulated fp64 GEMM for the RR step: the gate

Best-case local GEMM speedup S from measured anchors (see fp64_emu_fft/RESULTS.md):
measured int8 `_int_mm` = 15 TOPS; fp64-parity Ozaki needs s ≈ 8 slices (ozIMMU: 7–9),
triangular truncation → s(s+1)/2 = 36 int8 GEMMs per fp64 GEMM; complex via 4M/3M scales
native and emulated identically. Split/reconstruct overhead assumed ZERO (generous —
published <10% on A100-class bandwidth, worse on a 192 GB/s bus).

    S_best = (15 / 36) TFLOPS-equiv / 0.130 TFLOPS native = 3.2×

Composed end-to-end, GPU-resident with emulated GEMMs, vs **CPU davidson-native**:

| | al4 | si8 |
|---|---|---|
| GPU wall × (1 − g + g/S), g = GEMM share, S = 3.2 | 5.77×0.698 = 4.02 s | 6.36×0.622 = 3.96 s |
| vs CPU-native | 2.91/4.02 = **0.72×** | 3.85/3.96 = **0.97×** |
| absurd limit S = ∞ (GEMMs free) | 3.23 s → **0.90×** | 2.86 s → **1.35×** |

**Gate: 0.72× and 0.97× < 1.3× → NO-GO. Stop; no correctness prototype, no throughput
phase.** Even the physically impossible S=∞ limit fails on al4 and barely crosses on si8;
the honest ceiling does not come close. Two structural reasons:

1. **The GEMMs are <56% of a wall that is itself 1.65–2× underwater** vs the CPU baseline.
   Amdahl composes the deficit before the win: emulation must first pay back the 2× gap.
2. **The non-GEMM 45–56% is latency/glue** (QR panel kernels, elementwise, launch
   overhead, latency-bound eigh) — the same small-cell latency wall as CPU eager, which
   int8 tensor cores do not touch at all.

## What would change the verdict (honest, not a recommendation)

The GPU-resident deficit *shrinks with size* in these two points (0.50× → 0.61×), and VRAM
headroom is large (0.67 GiB used of 6). If a few-k large cell (Si-64-class, nb ≈ 130,
npw ≈ 15k — the measured 26× QE hole where CPU k-parallelism is stranded) fits in 6 GB and
its GPU GEMM share rises toward 70–80% with GEMMs big enough to amortize launches, the gate
arithmetic could pass there. That is a different probe (large-N GPU residency), with OOM as
the first risk — it is NOT licensed by this one, whose small/medium-cell verdict is a clean
NO-GO on both investigated angles (FFT emulation; RR-GEMM emulation).
