# Large-N GPU residency + fp32-draft on the RTX 3050 — final probes

Follow-on to `RESULTS.md` in this directory (small/medium Phase A + Ozaki-RR gate NO-GO).
Two questions, run in sequence on the same branch and rules:

1. **Large-N few-k GPU residency** (the residual opening): does GPU-resident fp64 beat the
   best CPU path at Al-32 / Si-64 (the 26× QE hole regime), and does the trend justify a
   production GPU path for large N?
2. **fp32-draft** (posit/takum-literature-motivated, mapped onto the 3050's 64× fp32:fp64
   ALU ratio): do the shipped mixed-precision modes (`mixed_precision` SCF draft,
   `GRADWAVE_FP32_EXPANSION`, `GRADWAVE_SUBSPACE_STORAGE=complex64`) clear a composed
   ≥1.3× gate vs best-CPU?

Hardware/conditions: asus RTX 3050 6GB Laptop (5.67 GiB usable), torch 2.12.1+cu130.
Every timed number: /tmp/BENCH_LOCK.fftemu held, no other lock, 1-min load < 0.6, CPU
arms 8 threads. Systems are the exact `/tmp/gw_size.py` configs (30 Ry, 2³ MP, PBE;
al32 = 2×2×2 of conv-Al, a=8.10, nk=8, npw=9939, grid 54³, ne=352, nb=212, FD 0.1 eV;
si64 = 2×2×2 of conv-Si, a=10.86, nk=8, npw=23871, grid 72³, ne=256, nb=154).

**CPU baseline provenance:** the band-parallel agent's arms in `/tmp/bandpar.log`
(same gw_size configs, same box, measured under its own bench lock earlier the same day):
al32 off=1298.2 s / band_parallel=8 **1069.0 s** (12 iters); si64 off=2257.9 s /
band_parallel=8 **2153.3 s** / bp4=2524.7 s (14 iters). Free energies cross-check with
the GPU arms to all printed digits.

## 1. Large-N ladder — measured

| system | GPU-resident fp64 | best CPU (bandpar bp8) | GPU vs best CPU | peak VRAM |
|---|---|---|---|---|
| al4 (from Phase A) | 5.77 s | 2.91 s (davidson-native, same-harness) | 0.50× | 0.67 GiB |
| si8 (from Phase A) | 6.36 s | 3.85 s (davidson-native, same-harness) | 0.61× | 0.38 GiB |
| **al32** | **901.6 s** (12 iters, F −60055.27829773, = CPU F) | 1069.0 s (12 iters) | **1.19× — GPU WINS** | 3.31 GiB |
| si64 | **does not fit** (OOM, see below) | 2153.3 s | — | >5.67 GiB |

al32 GPU arm conditions: shipped code + env knobs only
(`GRADWAVE_MAX_DIM_FACTOR=2 GRADWAVE_K_CHUNK=2 GRADWAVE_GPU_DENSE_BUDGET=2e8
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`); vs the off-arm (1298.2 s) it is 1.44×.

**The crossover is real and measured: 0.50× → 0.61× → 1.19× (al4 → si8 → al32).**

### Why: al32 GPU wall decomposition (3-iter CUDA-only profile, kernel rows)

| component | share |
|---|---|
| complex fp64 GEMM (cutlass z884gemm + zgemm_largek + trsm) | **77.1%** |
| FFT (cuFFT radix-54) | 9.8% |
| QR (geqr2_gmem_domino) | ~6.0% |
| elementwise/copy/reduce | 5.0% |
| eigh | 0.8% |

At large nb the wall collapses into the fp64 GEMMs (44–55% at 4–8 atoms → 77% at 32),
and batched cuBLAS zgemm at 0.13 TFLOPS beats the CPU's threaded zgemm + Python glue at
these sizes. The latency/glue wall that sank the small cells is amortized away.

### Si-64 does NOT fit 6 GB — the OOM ladder (all clean Python OOMs, no host damage)

Successive blockers, each a per-atom/per-k all-at-once materialization:
1. `core/batch.projectors_b` — transient (nk, npw, nproj) gather: **1.46 GiB** alloc.
   Patched on the research branch per-k (verified **bitwise-identical** on al4).
2. `core/energies/local_pp.local_potential_g` — (na, 72³) structure-factor tensors:
   **366 MiB** ×3 transients. Patched per-atom accumulation (verified ≤1 ulp,
   rel 9.5e-17 — reduction-order only).
3. `scf loop _solve_bands_streamed → batch.reindex` — **188 MiB** alloc with 5.36 GiB
   already resident, even at `K_CHUNK=1, DENSE_BUDGET=1e8`, in fp64 AND in the c64 draft.

Death by transients: each exact fix surfaces the next allocation. Si-64 GPU residency on
6 GB requires a systematic streaming-memory campaign (projectors, structure factors,
solver working set), not knob tuning. The two research-branch patches are exact and
disclosed; the al32 timed arm predates them (shipped code); the al32 profile arm ran
with patch 1 (identical math).

## 2. fp32-draft probe — measured

GPU-vs-GPU, same harness, fp64 GPU baselines al4 5.77 s (10 it) / si8 6.36 s (10 it):

| arm | wall | iters | vs fp64 GPU | F check |
|---|---|---|---|---|
| al4 `mixed_precision` draft | 9.38 s | 12 | **0.62× (regression)** | exact match |
| si8 `mixed_precision` draft | 8.31 s | 10 | **0.77× (regression)** | exact match |
| al4 `FP32_EXPANSION=on` | 12.24 s | 10 | **0.47× (regression)** | exact match |
| si8 `FP32_EXPANSION=on` | 8.82 s | 10 | **0.72× (regression)** | exact match |
| al4 `SUBSPACE_STORAGE=complex64` | 135.3 s | 100 | **NOT CONVERGED** (ΔF 1.2 meV) | fails |
| al32 `FP32_EXPANSION=on` (K2 DB2e8) | 1799.4 s | 13 | **0.50× vs 901.6 s fp64 (regression)** | exact match |
| al32 `mixed_precision` draft (K2 DB2e8) | 1638.0 s | 16 | **0.55× (regression; +4 handoff iters, and slower per-iter: 102 vs 75 s)** | exact match |
| si64 (any fp32 mode) | OOM | — | no data | — |

Prior measured record on this exact card (solver battery 2026-08-06,
`benchmarks/solver_battery/results/mixed_precision/cuda/`): davidson mixed vs fp64 —
insulators 1.16–1.35×, small metals 0.76–0.97× (regression), large supercells
si16 1.34× / cu8 1.23×; draft handoff costs 0–2 extra SCF iters; ΔF ≤ 3e-11 eV.
My small-cell arms reproduce the metal-regression class and si8 lands regressed too
(smaller than the battery's winning si16).

### Composed gate vs best CPU

Small cells: every fp32 mode is already a GPU-vs-GPU regression; composed vs
CPU-native = 0.3–0.5×. **NO-GO, measured.**

Large-N (al32, the GEMM-bound 77% regime that was the fp32 modes' best hope): BOTH modes
regress — fp32-expansion 0.50×, mixed_precision 0.55× vs plain fp64 GPU. The certified
re-polish / draft overhead exceeds whatever the c64 applies save in this batched-zgemm
regime, and the draft costs +4 SCF iterations. Composed vs best CPU:
1.19× × 0.50–0.55 = **0.60–0.66× — NO-GO, measured, in the most favorable regime that
fits the card.** No fp32 configuration measured on this GPU beats plain fp64 GPU
residency, let alone the 1.3× composed bar.

## Verdicts

1. **Large-N GPU residency: the crossover is REAL.** GPU-resident fp64 beats the best
   measured CPU path by 1.19× at Al-32 today, on shipped code, with F identical to CPU.
   But the 6 GB card runs out exactly one rung later: Si-64 cannot run at all. Wiring a supported production GPU path for large N is
   justified only conditionally: the measured payoff on THIS card is ~1.2× over one rung
   (32-atom class) before the VRAM wall; the memory-streaming campaign needed for Si-64+
   (projectors, structure factors, solver working set — three OOM sites measured) is real
   engineering. The trend line (0.50→0.61→1.19× with GEMM share 44→77%) says the win
   grows with size, so the same wiring on a ≥12 GB card is where the payoff lives. On the
   6 GB 3050: technically GO but marginal — a production path is defensible only as
   groundwork for bigger-VRAM hardware, not for this card's own sake.

2. **fp32-draft: NO-GO, measured everywhere.** Small cells: 0.47–0.77× GPU-vs-GPU
   (latency-bound; fp32 doesn't touch the wall). Large cells: 0.50–0.55× (draft/certify
   overhead + handoff iterations exceed the c64 gains). `SUBSPACE_STORAGE=complex64`
   diverges outright on the al4 metal (100 iters, unconverged). Composed vs best CPU
   everything is ≤0.66×. The posit-literature intuition (loose iterations tolerate low
   precision) does not map onto a win on this card with the shipped modes.

3. **The consumer-GPU story, closed end-to-end.** Across the whole campaign: fp64-FFT
   emulation NO-GO, Ozaki-int8 RR-GEMM emulation NO-GO, fp32-draft NO-GO in every mode
   and regime measured. The single surviving positive is plain fp64 GPU residency at the
   32-atom-class sliver (1.19×), immediately strangled by 6 GB VRAM one system-size rung
   later. On this hardware there is nothing further to build; the transferable assets are
   the measured crossover trend, the OOM-site map, and the two exact streaming patches —
   all of which point at bigger-VRAM cards, not the 3050.
