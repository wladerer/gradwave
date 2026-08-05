# gradwave Optimization Panel — Final Report

One candidate survived verification. The shortlist is therefore a single item; the killed appendix records 24 dead ends so they are not re-proposed.

## 1. Ranked shortlist

| # | Candidate | Verdict | Ranking rationale (realistic factor × breadth ÷ effort) |
|---|---|---|---|
| 1 | `fp64-via-fp32-ozaki` | plausible | Only surviving candidate; ~1.2–1.5× end-to-end on production-size cells, device-A-only (narrow regime), medium effort — modest but real, gated on one cheap microbenchmark. |

No candidate reached "strong." Nothing ranks above it and nothing ties.

## 2. Shortlisted item

### 1. fp64-via-fp32 Ozaki emulation on the 1/64-fp64 consumer GPU

**What it is.** Emulate every double-precision skinny/subspace GEMM as ~3 fp32 slices per operand using the Ozaki error-free scheme, so the work lands on the RTX 3050's fast fp32 CUDA cores instead of its crippled (1/64-rate, measured penalty 38.3×) native fp64 units. Recovers correctly-rounded fp64 (~1e-13), so it is built to clear the 1e-9 residual / autograd-force accuracy gate that sank the plain fp32-RR attempt. Explicitly excludes the 3D FFTs (accuracy-fatal in low precision) and the fp64 m×m eigensolve; it is a throughput fix and does nothing for the 32–46% launch gap.

**Named literature.** Ozaki, Ogita, Rump & Oishi, *Numer. Algorithms* 2012; Mukunoki, Ozaki, Ogita & Imamura, *ISC-HPC* 2020; Higham & Mary, *Acta Numerica* 2022.

**Where it lands in the code (verifier's notes).** The target objects are the two skinny complex-c128 subspace GEMMs in `solvers/davidson.py`: `s = matmul(v.conj(), hv.mT)` at line 444 and the Ritz combines at lines 449–450. The fp64 tax on those is the documented binding constraint on device A ("The GPU limit is precision, not structure": fp64 at 1/64 fp32, 100% util at 25 W of a 60 W budget = arithmetic-throughput-bound; named remedy is an "fp32-dominant solver schedule … reserves fp64 for a final polish"). Reuse the existing `_fp64_penalty` timing pattern in `davidson.py` and extend `benchmarks/bench_matrix.py`.

**Realistic factor and regime.** FLOP-ceiling upper bounds are ~3.5× (complex via 3M, 64/18) and ~10× (real, 64/6), reached only on compute-bound large-m shapes. Amdahl on profiled shapes: bmm-GEMM is 43% of GPU-busy on Si8 (23% on Si2) and GPU-busy is only ~68% of wall, giving ~1.25× overall on Si8, less on Si2. The full 13–18× fp32 headroom exists only at the hematite dim=240 shape (measured 329→18.7 ms), where Ozaki's 6–9 slices claw it back to ~2–3×. At small skinny shapes (m=16–40, launch-bound) the extra operand traffic and slice/round/accumulate kernels can erode or reverse the win and inflate the very launch gap the diagnosis flags. Honest ceiling: ~1.2–1.5× on production-size cells, near-neutral on small ones. **Device A (RTX 3050) only** — not device B (AVX2 fp32 is only 2× fp64, slice overhead is a net loss), not the transforms, not the launch-gap glue. Caveat: at the sizes where it pays, device C (datacenter fp64) already delivers 15–58× and is the documented path.

**Effort.** Medium (1–3 weeks, real convergence risk): correct 3-slice split with scaling, error-free pairwise products, complex 3M wrapping, integration into the two Davidson subspace GEMMs, plus convergence verification across the seven-system battery and per-shape benchmarking. Invariants: 1e-13 accuracy is << the 1e-9 gate and native DGEMM is not bitwise-reproducible across BLAS anyway, so strict cross-device bit-match is not a blocker.

**First experiment.** On the RTX 3050, extend `benchmarks/bench_matrix.py` to time native complex128 `s = matmul(v.conj(), hv.mT)` at the hematite subspace shape (nk=13, dim=240, npw=6746) against a 3-slice fp32 Ozaki prototype, asserting max-abs error < 1e-13 vs the fp64 result. Then substitute the Ozaki GEMM into the `davidson_batched` loop body (`davidson.py:444–450`) on the nb=60 gapped-spectrum synthetic operator and confirm it reaches tol=1e-9 in ~131–146 iters rather than the 1–2e-5 floor. **Falsifier:** Ozaki ≤ native fp64 at that shape once slice/accumulate overhead is counted, OR 3 slices miss 1e-13 / convergence stalls.

## 3. Independently converged

No candidate was proposed by two or more lenses. `fp64-via-fp32-ozaki` was proposed by a single lens (`mixed-prec`). Nothing to report.

## 4. Killed appendix (do not re-propose)

- **sketched-rayleigh-ritz** — Self-contradictory: names a 40.8% target that exists only on the fp64-crippled device A but restricts its win to device C where the same RR GEMM was measured at <1% of solve; randomized sketch injects ~5e-2 subspace distortion, fails 1e-9 worse than the already-rejected fp32-RR; SRHT does not reuse the FFT.
- **ozaki-error-free-rayleigh-ritz** — Accuracy invariant respected, but accuracy was never binding on device C: RR GEMM already runs 1.26 ms native fp64 with fp32 giving no margin; shape too small to saturate, so int8/tensor-core Ozaki only adds launch + reconstruct overhead. Regime does not transfer to device C.
- **grassmann-subspace-extrapolation** — Invariant-safe but regime mismatch: geometry-sequence density-matrix extrapolation needs a drifting chain of converged subspaces (relax/MD/EOS), absent on the fixed-geometry single-point solve; in its real regime it is an already-recognized ~1.3×/point iteration-count play.
- **grassmann-transported-momentum** — Already landed as warm-start (9→2 outer steps); intra-solve P block exists in `lobpcg.py`; class capped at ~1.3×/point. Claimed 1.2–1.5× is measured vs a cold-restart straw man; the named 9× deficit is dispatch + augmentation overhead, not the eigensolver.
- **soft-locking-deflation-active-block** — Soft-locking half already shipped (`davidson.py:460/490`, `n_add`); hard-locking is a deliberate documented rejection for the batched path (ragged block width destroys the single batched GEMM/eigh/FFT), and would break the bit-identical-energy gate.
- **pruned-fft-operator-action** — Sub-box FFT corrupts V(r)·ψ(r) via aliasing (box is deliberately larger for anti-aliasing); safe form (dual grid + Γ real-FFT) already landed; hand-rolled FFTW sits outside autograd and breaks forces/stress; FFT-is-the-wall premise false on device A.
- **nystrom-jacobian-preconditioner** — Named hard cases are not charge-channel: AFM Fe is the Stoner/spin mode, spinor Ni is a residual-gate artifact (already fixed). Kerker is 0/0 at the G=0 pin where the slow mode lives; the probe repair is already in `response_from_residuals()`.
- **safeguarded-regularized-anderson** — Already implemented (condition filtering, Tikhonov ridge, trust-region fallback in `scf/mixing.py`); targets wrong subsystem — limit cycle is an occupation-lag pathology (fix is `hub_occ_mix`), the 80-floor is a residual-gate artifact.
- **block-per-field-kerker-preconditioner** — Verbatim measured-negative in `ideas.md` (BlockPrecond + const-term MultipoleKerkerPrecond); 80-floor already resolved by the energy-metric gate; mag channel is the Stoner mode (gain ≈ −6) needing vigorous mixing, wrong form for a Kerker envelope.
- **two-level-deflation-low-q-modes** — Coupled slow mode is a rigid magnetization rotation at G=0; deflating it is the rejected reverse-Kerker damping that drives moments to anti-alignment; the named targets are not low-q charge modes; the fitting-regime mechanism needs the unsolved forward χ0 solve with a ~20%→0% ceiling.
- **riemannian-direct-minimization** — 80-floor is already-converged energy (residual-gate artifact); limit-cycle regime (smeared metal + DFT+U) is exactly where fixed-occupation Stiefel/Grassmann is invalid and the ensemble-MV alternative was measured at ~0.4×; Kerker vs Teter preconditioner is a category error; re-deriving forces/stress/USPP/PAW/spinor is multi-month.
- **randomized-gram-schmidt-tall-qr** — Competes on the wrong axis: the 1.55–1.67× baseline is a host-offload escaping cuSOLVER's launch tax, and RGS re-pays it on-device; caps overall wall at 1.11×/1.31×; single-pass sketch is only sketch-metric orthonormal (fails precision-fragile RR), and its own falsifier concedes a reortho pass erasing the advantage.
- **randomized-rangefinder-remove-n2-cliff** — Subspace reduction already forms only (nk,m,m) and reconstructs in O(nm) — no n×n object to remove; the real 37 GB O(npw²) alloc is unconfirmed and corresponds to no npw-sized eigensolve; sheet's own win condition prescribes exact tiling.
- **ozaki-error-free-gram** — Conj-copy relief already shipping (`davidson.py:444` lazy-conj matmul); device-C profit contradicted by H100 measurement; best-realistic 1.22–1.31× device-A-only, loss on device C, never touches the ortho+eigh bundle; 37 GB cliff misattributed to the m×m Gram.
- **shifted-choleskyqr-tall-qr** — On-GPU QR already fine on real fp64 (CPU-offload was a measured 13% penalty on H100); device C is GEMM-bound not QR-bound; on device A CholeskyQR runs on-GPU at the same 1/64 tax; QR is 14% of device-busy (≤1.11×); the tall block is routinely rank-deficient so the rank-repair apparatus is still required.
- **herk-gram-no-conjugated-copy** — HERK is mathematically inapplicable to the cross-term s = Vᴴ(HV); the 7 GiB conj-copy spike is already eliminated (commit 059b93d lazy-conj matmul); the 37 GB cliff is a separate allocation. Already-tried, inapplicable, zero factor.
- **metric-nonorthonormal-rayleigh-ritz** — cond(S) reorth trigger is a measured-negative (fires constantly, not rarely); forming S=VᴴV squares the condition number and can yield Ritz values below the true minimum (breaks variational bound for forces); adds a second Gram + costlier generalized eigensolve on the m>32 cliff.
- **mrrr-eigensolve-tiled-back-transform** — Subspace eigh is O(m²) on (nk,m,m) — cannot be the 37 GB O(npw²) alloc; the tiled Ritz back-transform is already implemented (`davidson.py:449`, `_combine`); lazy-conj Gram already lands; the true O(npw²) alloc is documented as unidentified and elsewhere.
- **batched-jacobi-eigensolver** — (test entry; no substantive kill reason recorded).
- **scf-chefsi-deferred-rayleigh-ritz** — Deferred-RR CheFSI is the implemented, permanently-closed no-go; CheFSI cannot remove the RR; the sole benefit (not over-solving) is already captured by the adaptive Davidson tolerance schedule; 2–3× H-apply loss can't be recovered by trimming a minority RR term.
- **polynomial-spectrum-slicing** — Strictly-worse CheFSI variant multiplying the H-apply count that killed CheFSI; target regime (metals) has no spectral gap so window boundaries fall inside dense near-degenerate clusters; mis-locates the bottleneck (real pain is the O(npw²) cliff + fp64 tax).
- **jackson-damped-chebyshev-filter** — Attacks filter degree inside a solver that is off-by-default because it is a closed no-go (2–3× H-applies, ~90% of wall); a 10–30% degree cut cannot overcome a 2–3× gap; damping needs equal-or-higher degree on metals where CheFSI is worst.
- **three-precision-iterative-refinement** — Already landed as `mixed_precision_solve` (c64 draft → c128 metric/Gram/eigh → fp64 polish); novel increments already swept-rejected (fp16 transforms accuracy-fatal, FP8 Blackwell-only, fp32-subspace obsolete); no target device has block-scaled fp16 hardware; 2×-on-device-B contradicted by measurement (1.06–1.10× insulators, <1× metals).
- **libxsmm-fftw-device-b-backend** — Misdiagnoses device B's 9× deficit: FFTs already route through MKL FFT and skinny products through MKL BLAS (the same MKL QE links); real cause is PyTorch dispatch overhead + less-tuned 400 Ry augmentation; addressable small-GEMM fraction ~12% and complex128 (libxsmm's weak path); best-case ~1.1× vs 9×.
- **colored-banded-hessian-probe** — Exact-HVP + column-collapse are the landed analytic Γ-DFPT path (`gamma_hessian`, HessianSymmetry) which dominates coloring on high-symmetry cases; inter-site-decay win is largely the landed Born–von-Kármán reduction; home-cell atoms are co-located so there is no banding to color; attributes its biggest win to the FePt metal single-SCF delta-gauge eval, not a phonon Hessian.

## Measured outcome for the survivor (2026-08-04, asus RTX 3050, torch 2.12.1+cu130)

The first experiment ran as a standalone script against the exact hematite subspace
shape (nk=13, m=240, npw=6746) and the falsifier fired on both counts.

- The literal 3-slice fp32 claim is wrong on accuracy. Naive round-to-fp32 slicing
  with fp32 accumulation lands at max relative error 3.1e-6, identical to a plain
  fp32 GEMM, because the leading slice-pair GEMM still rounds its accumulation at
  2^-24. It misses the 1e-13 target by seven decades and runs only 1.44x, strictly
  worse than the already-rejected pure-fp32 path.
- The correct error-free construction (10-bit Ozaki slices, chunk-64 fp32
  accumulation exact by design, fp64 chunk reduction) delivers the accuracy
  (8.4e-15) and loses the race at 0.17x, six times slower than native c128. The
  21 slice-pair GEMMs plus the fp64 reduction swamp the 64x fp32 rate advantage.
  TF32 tensor cores do not change the verdict (0.17x).
- The raw fp32 headroom is real (14.8x at the hematite shape) but no error-free
  route through fp32 GEMMs captures it at these shapes.

One variant remains untested, Ozaki on int8 tensor cores with exact int32
accumulation (the Ootomo-Ozaki DGEMM-on-tensor-cores line), which needs no
chunked reduction. torch exposes only 2D `_int_mm`, so it would need a batching
loop and real engineering. Given 0-for-25 on the panel, it is recorded here as
an open question, not a recommendation.

Final panel score: 25 candidates, 25 dead. The negative is the finding: the
conventional and adjacent numerical space around this code is exhausted, and
speed research should go to learned convergence operators and Hvp-based
second-order pipelines instead.
