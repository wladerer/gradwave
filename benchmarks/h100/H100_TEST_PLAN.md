# H100 SXM validation plan — wall-time / datacenter-GPU features

Prepared, **not deployed** (tomorrow's job). These answer the questions CPU/asus can't:
the "full-rate fp64 + launch-bound at small op sizes" regime, where batching/fusing many
small ops into fewer large kernels can win. The RTX 3050 (asus) is a *dev* card only
(fp64 = 1/64) — never the perf verdict for an fp64-GPU idea.

**Gate on every test:** ΔE vs the CPU reference < 1e-9 eV (same fixed point). A GPU speedup
that moves the answer is disqualified.

Run each with `uv run python ...` **on the H100 box** (pull the relevant branch first).

---

## 0. Calibration baseline — RUN FIRST (gates everything). READY.

**Script:** `benchmarks/h100/test0_calibration.py` (this branch).
**Decides:** (a) does H100 fp64 beat the 22-core CPU for small/medium metal cells? (b) is the
SCF launch/dispatch-bound (low % of wall in FFT/GEMM)? Everything below is only worth doing if
this shows the launch-bound regime.
**Run:** `uv run python benchmarks/h100/test0_calibration.py` (defaults: fcc-Al 4/32/108 atoms).
**Read:** `gpu/cpu < 1` ⇒ H100 wins that size; `gpu%cmp` low (<~50%) ⇒ launch-bound ⇒ batching helps.

## 1. Spin-Batched Davidson A/B — READY (existing branch).

**Script:** `benchmarks/spin_batch/spin_batched_davidson_bench.py` on branch `spin-batched-davidson`
(auto-selects CUDA). **Decides:** does halving Davidson's per-op dispatch win on full-fp64,
launch-bound hardware? It LOST on CPU (no launch latency) and the crippled 3050 (fp64 1/64) — H100
is the honest test. If it wins here, the whole batch/fuse family is validated for the datacenter.
**Run:** `git checkout spin-batched-davidson && uv run python benchmarks/spin_batch/spin_batched_davidson_bench.py --threads 8`
(don't pass `--quick` for the real numbers).

## 2. Batched multi-config SCF — the FLAGSHIP. NEEDS BUILDING (build only if 0+1 are positive).

**Idea:** ideas.md ~1700 "batched multi-structure SCF" — fold N configs (EOS volumes / a U-scan /
strains) into ONE batched eigensolve, config as a batch axis alongside k. Its whole value is
H100-gated (full-rate fp64 + launch-bound). **Decides:** the 2–4x the analysis predicts on a real
fp64 GPU. **Status:** not built — the config-batch feature (block-independent mixing / per-config
Fermi + convergence masking on a config axis in `core/batch.py`) is the next build IF test 0 shows
the launch-bound regime. Do not build it blind.

## 3. SeedPool (CPU process pool) vs GPU-batch, on the H100 node. READY once #2 exists.

**Decides:** for the (reduced) campaign spokes, does CPU-process-parallel (SeedPool, PR #276) or
one GPU-batched tensor (#2) win on a GPU node? Mutually exclusive for the same spokes; picks the
campaign execution strategy on datacenter hardware. **Run:** a reduced phonon/elastic campaign both
ways, same system, compare wall + ΔE.

## 4. CUDA-graph launch-reduction on the inner loop. READY-ish (probe).

**Decides:** the "whole-step CUDA-graph = null" verdict was on the 3050. Graph *replay* removes
launch overhead — exactly the H100-small-cell bottleneck, and DISTINCT from the inductor-complex-
codegen block (which is hardware-independent-dead). One probe: capture the post-eigh Davidson round
math as a CUDA graph and replay. Low priority; only if 0 shows a large launch-bound fraction.

---

## Suggested first session

**0 → 1.** Both are ready (0 = script here, 1 = existing branch) and together answer the load-bearing
question: *is gradwave's small-cell SCF launch-bound on H100, and does dispatch-fusion win there?*
- If **yes** → greenlight building #2 (the flagship batch-into-one-tensor) as the datacenter payoff.
- If **no** (H100 fp64 is throughput-bound even at small sizes) → batching won't help; the campaign
  play stays CPU-process-parallel (SeedPool + the symmetry reductions #274/#275), and we've saved
  the effort of building #2.

## What does NOT need H100 (already settled on asus)

- Reduction correctness (#274 DisplacementStar, #275 StrainStar) — exact, hardware-independent.
- Reduction SCF-count cut (3×) — a count, identical on any card.
- Tol-ladder (#273, iteration count), Import Diet (#271, import time).
- SeedPool correctness (#276) — parallel == serial, validated on asus.
