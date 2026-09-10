# fp64-emulated-FFT on the RTX 3050 — research probe

**Verdict: NO-GO.** Building a production fp64-emulated (Ozaki int8) FFT path for the
gradwave SCF on the RTX 3050 has an Amdahl ceiling of **≤ 1.0×** (realistically a net
**loss**, ~0.7–0.85×). The probe stops at Phase 2 per protocol: the ceiling lands below
the 1.3× go threshold.

The reduction the probe set out to validate is *mathematically sound* — an FFT does reduce
to a sequence of GEMMs against fixed twiddle matrices, and those GEMMs can be Ozaki-emulated
to fp64 accuracy (prior art confirms this on data-center fp8 hardware). It fails not on the
math but on the hardware economics of *this* card, and on a wrong premise: **the native fp64
FFT is not the bottleneck on the 3050.** It is memory-bandwidth bound, so the crippled fp64
ALUs barely touch it, and it already beats the 8-thread CPU FFTW by 2.5–3× at slab sizes.
Emulation would replace an already-good native kernel with a strictly heavier one.

All numbers below measured on `asus`: NVIDIA GeForce RTX 3050 6GB Laptop GPU (GA107, cc 8.6),
torch 2.12.1+cu130, under `/tmp/BENCH_LOCK.fftemu`, load < 0.1. cuda-event timing, warmup+sync.

---

## Phase 1 — prior art

| Result | What it establishes | Relevance to this probe |
|---|---|---|
| **Ozaki scheme / ozIMMU** (Ootomo, Uchino, Ozaki, Imamura; enp1s0/ozIMMU; arXiv 2306.11975) | fp64-equivalent DGEMM via **int8** tensor cores, 3–18 slices (7–9 for well-conditioned), <10% overhead **on data-center GPUs with fast int8** | The GEMM building block is real and proven. |
| **"DGEMM without FP64 Arithmetic"** (arXiv 2508.00441) & **Ozaki-II** | Extends to fp8; guaranteed-accuracy variants | Confirms slice/limb accuracy control. |
| **"FP8 is All You Need" Pt.1/2** (arXiv 2606.06510, 2606.23698) — the one paper that does **fp64-accuracy FFT via tensor cores** | FFT → **Ozaki-emulated Bailey GEMMs** (validates the exact reduction this probe proposed), needs **r=12 pairwise-coprime moduli / 118-bit budget** for fp64 (K=32 radix). **Targets Blackwell B300 / Rubin fp8 only, explicitly not Ampere/consumer.** Even on B300 it is **memory-bound, 4.9–6.7× below the memory roof**, dominated by the **integer CRT epilogue** (deconstruct/reconstruct/carries), *not* the tensor cores. **Whole-transform accuracy is still deferred "future work" (§9)** — nobody has measured a fp64-parity emulated FFT end-to-end yet. |

**Genuinely open vs closed:** the *reduction* is not open — it is published. What was never
measured is (a) fp64-parity emulated FFT *end-to-end accuracy*, and (b) any of it on
*consumer int8 Ampere*. Both are moot here because Phase 2's ceiling kills the motivation
before accuracy or throughput matter. The published work's own binding constraint (memory-bound
integer epilogue, even with B300's ~8 TB/s HBM3e) is the same wall that sinks the 3050 harder.

---

## Phase 2 — Amdahl ceiling (the gate) — FAILS

**Ceiling = (FFT+local-apply wall fraction) × (best-case local speedup of emulated vs current path).**

Wall fraction (from the measured record): FFT + local apply = **0.40–0.46** of SCF wall at
12–16-atom slabs (use f = 0.43).

**Best-case local speedup — measured, three independent kills:**

1. **The native fp64 FFT already beats the CPU baseline** (the SCF's current path is CPU FFTW):

   | shape (N³ × bands), complex128 | GPU native fp64 | CPU 8-thread | GPU/CPU |
   |---|---|---|---|
   | 24³ × 32   | 640 µs   | 410 µs   | 1.56× (GPU slower — latency) |
   | 48³ × 32   | 5.43 ms  | 9.50 ms  | **0.57× (GPU 1.75× faster)** |
   | 48³ × 130  | 14.1 ms  | 36.0 ms  | **0.39× (GPU 2.6× faster)** |
   | 64³ × 32   | 7.51 ms  | 23.7 ms  | **0.32× (GPU 3.1× faster)** |
   | 96³ × 8    | 8.52 ms  | 19.6 ms  | **0.43× (GPU 2.3× faster)** |

   FFT effective rate is ~85–100 GFLOP/s on GPU, memory-bound at 192 GB/s — the crippled fp64
   ALUs are *not* the FFT limiter. **So there is nothing for emulation to fix on the FFT.**
   Emulation is strictly heavier than this native kernel (adds slice deconstruction + s²/2 int8
   GEMMs + 12-modulus CRT reconstruction, all re-touching the same 192 GB/s bus). Local
   speedup of emulated vs native FFT is therefore **< 1.0** — a slowdown, not a speedup.

2. **Measured int8 throughput is only ~15 TOPS**, 4× below the 61-TOPS spec (torch `_int_mm`
   reality on this laptop card):

   | GEMM M=N=K | int8 `_int_mm` | native fp64 `@` |
   |---|---|---|
   | 1024 | 15.0 TOPS | 0.130 TFLOPS |
   | 2048 | 17.0 TOPS | 0.130 TFLOPS |
   | 4096 | 14.8 TOPS | 0.132 TFLOPS |

   int8/fp64 ratio = **115×**. After the Ozaki penalty for fp64 FFT (r≈12 moduli → order
   s²/2 ≈ 36–64 int8 GEMMs per fp64 GEMM), the emulated fp64-GEMM-equivalent rate is only
   ~0.23–0.42 TFLOPS — barely above the 0.13 TFLOPS native fp64 GEMM, and *below* what the
   memory-bound native FFT already delivers. The tensor cores are not the win they look like
   on paper, and the FFT isn't GEMM-flop-bound anyway.

3. **PCIe = ~12 GB/s measured** (H2D 11.9, D2H 12.3 GB/s; link negotiates below its gen4-x16
   max under this load). A single 48³ × 130 band batch is **1.84 GB** → **~150 ms per
   direction, ~300 ms round-trip** — vs **14 ms** to compute it on-GPU or **36 ms** on CPU.
   FFT-only offload is **8–20× underwater** on transfer alone. The only way to hide PCIe is to
   keep the whole SCF resident on-GPU — but then the FFT is *already* native-fast and the
   bottleneck becomes the fp64 GEMMs in Davidson (0.13 TFLOPS, catastrophic), which an
   FFT-emulation path does not touch.

**Ceiling arithmetic:** with the most generous honest local speedup S ≤ 1.0 (emulation cannot
beat the native FFT it replaces):

    end-to-end = 1 / ((1 − f) + f/S)  with f = 0.43, S = 1.0  →  1.00×
    realistic S ≈ 0.2–0.5 (emulation 2–5× slower than native)  →  0.72–0.85×  (net LOSS)

**1.00× ≤ 1.3× threshold → NO-GO. Stop.**

---

## Phases 3 & 4 — not reached

Correctness prototype and throughput measurement were **not built**: Phase 2 gates them and it
fails. Building an int8-Ozaki radix-split FFT to prove accuracy parity would be effort spent on
a path whose *best possible* end-to-end outcome is break-even and whose realistic outcome is a
regression, on hardware where the native fp64 FFT is already the faster option.

---

## Verdict & next step

**NO-GO for a production fp64-emulated-FFT path on the RTX 3050.** Reasons, in priority order:

1. **Wrong target.** The fp64 FFT is memory-bound, not ALU-bound; native fp64 FFT already beats
   8-thread CPU FFTW 2.5–3× at slab sizes. Emulation fixes a bottleneck that does not exist.
2. **Strictly heavier.** Ozaki emulation adds slice + s²/2 int8-GEMM + 12-modulus CRT epilogue
   over a kernel that already saturates the 192 GB/s bus; the published work is memory-bound on
   a card with ~40× more bandwidth.
3. **PCIe.** FFT-only offload is 8–20× underwater; whole-SCF-resident makes emulation pointless.
4. **Consumer int8 is weaker than advertised** (~15 vs 61 TOPS measured).

**The single actionable redirection (not a GO for this probe):** the real fp64 wall on the 3050
is the **dense fp64 GEMMs in the Davidson Rayleigh-Ritz / subspace path** (measured 0.13 TFLOPS,
~59× below fp32). *That* is the Ozaki-int8-emulation target with a real ceiling — the RR GEMMs
are compute-bound and the 115× int8/fp64 ratio survives the s² penalty there far better than in
a memory-bound FFT. If any consumer-GPU emulation probe is worth running next, it is
**Ozaki-int8-emulated fp64 GEMM for the RR step**, gated by its own Amdahl fraction (RR GEMM
share of SCF wall) — not the FFT.
