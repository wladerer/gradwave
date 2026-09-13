# Decision records — the ledger for long-form "why" comments

**Status: active convention (adopted 2026-09-12; piloted on `src/gradwave/core/batch.py`).**

The codebase carries two kinds of "why" commentary. Short contracts a reader
needs to modify the next twenty lines correctly — sign conventions, padded-slot
invariants, shape notes — belong inline and stay inline. But multi-paragraph
*decision narratives* ("we tried X, measured 0.9×, so Y; a data-center GPU may
invert this") grow without bound at the call site, and the same story often
governs three functions at once. Those move here, to one appendable ledger,
and the code keeps a one-line greppable pointer.

## The convention

- **One record = one settled decision**, with a stable ID `[D-NNN]` allocated
  monotonically in this file. IDs are never reused or renumbered; a reversed
  decision gets a new record and the old one's status becomes
  `superseded by [D-MMM]`.
- **The code pointer is one line:**
  `# [D-001] Toeplitz gate is a measured verdict — docs/design/decision-records.md`
  (gist first, then the doc path so the doc-refs guard would catch a rename of
  this file if the pointer ever migrates into a markdown doc). Grep `\[D-` to
  find every governed site; grep `\[D-001\]` for one decision's sites.
- **Each record carries:** date · status (`active` / `superseded by [D-x]`) ·
  the measured evidence (numbers, hardware, PR/issue) · the code sites it
  governs · the decision itself · rejected alternatives. Keep records terse —
  the evidence line is the payload.
- **What stays inline** (do NOT move): physics/units/sign contracts (e.g. the
  normative header of `src/gradwave/core/fftbox.py`), data invariants
  (`batch.py`'s padded-slot rules), shape/API notes, safety warnings on the
  exact line that needs them, and a one-line "why" wherever the code would
  otherwise look wrong. Rule of thumb: if deleting the comment could make the
  *next edit* incorrect, it stays; if it only explains *history* (what was
  tried, what it measured, why an alternative lost), it moves.
- **Relationship to existing docs:** `docs/design/*.md` campaign notes stay as
  they are — a record may cite one as its evidence trail. `docs/ideas.md`
  holds *proposals not yet decided*; this file holds *settled, code-governing
  verdicts*. Auto-memory notes are session context, not authority — the record
  is the durable home.
- **Zero new tooling to start.** The doc-refs guard
  (`scripts/check_doc_refs.py`) already verifies every `src/...py` path named
  in a record's Sites line, so a record cannot silently point at moved code.
  Natural later extension (its planned Phase 2 already envisions symbol
  checks): scan `src/**/*.py` for `\[D-\d+\]` and fail on IDs missing from
  this ledger, and warn on records whose Sites files no longer contain the ID
  — bidirectional, still one script.

---

## Records

## [D-001] Toeplitz local apply is gated by a measured per-geometry verdict, not a size threshold

- **Date:** 2026-08 (PR #316 + auto-gate follow-up) · **Status:** active
- **Sites:** `src/gradwave/core/batch.py` — `_use_toeplitz`, `_TOEPLITZ_MODE`,
  `_TOEP_TRIAL_MARGIN`, `_TOEPLITZ_M_BUDGET_BYTES`, `BatchedHamiltonian.apply`
- **Decision:** the small-cell dense-GEMM local apply (V·ψ as `M @ c`,
  bit-identical to the FFT path — an algebraic identity) is adopted per
  (device, nk, npw_max, shape, dtype) signature only after a one-time timed
  trial on the real block shows `t_toep < 0.7 · t_fft`; the verdict is cached
  for the process lifetime.
- **Evidence:** the win is strongly size-dependent — measured 14× per apply at
  npw≈190 (ecut 12 Ry Si) and ~1.5× on a 512-k Al whole-SCF, but *losses* at
  production cutoffs (whole-SCF Si 0.80×, GaAs 0.79×, Al neutral). The
  crossover is machine- and geometry-dependent, so any hand-tuned npw
  threshold is wrong somewhere; timing both exact paths once costs one extra
  local-term evaluation per geometry and its answer is usable either way. The
  30% margin covers the per-iteration M rebuild amortized over that
  iteration's applies.
- **Rejected:** fixed npw cutoff (machine-dependent); always-on (regresses at
  production ecut). Budget: M is nk·npw²·16 B cached for the H's lifetime
  (+~half again for the index table); the 256 MiB default deliberately
  conservative (covers npw≈260 up to ~250 k-points) so the path never
  surprises a memory-tight GPU — raise to extend coverage.

## [D-002] Toeplitz path stays off on CUDA by default

- **Date:** 2026-08 (RTX 3050 measurements) · **Status:** active
- **Sites:** `src/gradwave/core/batch.py` — `_TOEPLITZ_ON_CUDA`
- **Decision:** `_TOEPLITZ_ON_CUDA = False`; GPU use is an explicit flip for
  data-center hardware validation, never a silent default.
- **Evidence:** on the tested RTX 3050 the isolated fp32 `M @ c` is
  tensor-core-fast, but the whole-SCF picture does not hold: the non-apply
  fp64 FFTs (density build, Hartree, XC) dominate and Amdahl-dilute the gain,
  and fp64 GEMM is crippled (~1/64 fp32) on consumer GPUs, so a pure-fp64 GPU
  SCF would *regress*. A real-fp64 data-center GPU may invert this — flip the
  flag to test there.

## [D-003] Dense-box band-chunking: GPU always chunked (4e8 default), CPU unchunked unless forced or auto-estimated

- **Date:** 2026-08/09 (slab memory campaign, PRs #417/#418 lineage) ·
  **Status:** active
- **Sites:** `src/gradwave/core/batch.py` — `_gpu_dense_budget_bytes`,
  `_cpu_dense_budget_bytes`, `set_cpu_dense_budget_override`,
  `_dense_band_chunk`
- **Decision:** the dense-grid FFT-box transient (~4 temporaries live at once
  in the apply/density chain) is band-chunked to a byte budget. GPU: always,
  default 4e8 B, `GRADWAVE_GPU_DENSE_BUDGET` overrides (read per call so
  benchmarks can A/B in-process). CPU: unchunked historical path by default;
  the explicit `GRADWAVE_CPU_DENSE_BUDGET` env var wins over the
  process-local override that the SCF auto estimator
  (`scf.loop._auto_cpu_dense_budget`) sets only for cells whose unchunked box
  would be a large fraction of RAM — small/medium cells run byte-for-byte as
  before.
- **Evidence:** chunking is BIT-EXACT for the H-apply local term (no
  cross-band reduction; only the tiling changes) and reorders the density
  build's band sum at the ~1e-16 ulp level. The dense box is the binding
  memory transient on big slabs: a ~5e8 budget keeps it near 0.5 GB, fitting
  a ~200-atom slab SCF on a 16 GB laptop. The env-over-override precedence
  keeps an operator's hand-set budget authoritative; the SCF sets/clears the
  override in try/finally so relax/EOS chains never leak one cell's budget
  into the next.

## [D-004] No resident `p.conj()` — fold the conjugation into the BLAS matmul

- **Date:** 2026-09 (projector-table dedup, PR #481 lineage) · **Status:** active
- **Sites:** `src/gradwave/core/batch.py` — `becp_b`,
  `BatchedHamiltonian._tables` (and the `_tab_cache` construction),
  `BatchedHamiltonian.apply_cols`
- **Decision:** ⟨β|ψ⟩ overlaps are computed as `matmul(c, p.conj().mT)` — a
  conjugate-transpose *view*, folded by CPU BLAS into `cgemm(ConjTrans)`,
  exactly as the native C kernel does with `zgemm(CblasConjTrans)`. `becp_b`'s
  `p_conj` parameter is retained for call-site compatibility and IGNORED.
- **Evidence:** the einsum form it replaced could not fold the conj, so hot
  callers cached a resolved `p.conj()` — a second full (nk, nproj, npw_max)
  projector table resident for the Hamiltonian's lifetime, measured ~0.73 GiB
  at Si-64/30 Ry, for zero arithmetic benefit. The matmul removes both the
  resident copy and the per-round re-materialization.

## [D-005] Projector phases: batched gather only on the grad path; per-k loop under no_grad

- **Date:** 2026-09 (Si-64 memory profile) · **Status:** active
- **Sites:** `src/gradwave/core/batch.py` — `projectors_b`
- **Decision:** with autograd enabled (forces path), keep the all-k batched
  gather — one clean autograd node. Under no_grad (the SCF itself), build the
  table per k into a preallocated output.
- **Evidence:** the all-k gather materializes a transient (nk, nproj,
  npw_max) complex block on top of the equally-large `proj_phase_free` and
  the returned table — three full projector-table copies at once, measured
  0.73 GiB each at Si-64/30 Ry. The per-k loop holds one k-slice transient
  and is byte-identical (same indexing, same multiply, no cross-k reduction).

## [D-006] Toeplitz difference-index table lives on the BatchedK and is revalidated against gather_idx

- **Date:** 2026-08 (Sternheimer NaN postmortem) · **Status:** active
- **Sites:** `src/gradwave/core/batch.py` — `BatchedK.toep_idx_cache`,
  `BatchedK.reindex`, `BatchedHamiltonian._toeplitz_idx`
- **Decision:** the (nk, npw, npw) difference-index table is geometry-only
  (Miller indices fixed for the whole SCF), so it is cached on the BatchedK —
  shared across the per-iteration Hamiltonian rebuilds — and a cache hit is
  trusted only if built from THIS operator's `gather_idx` (identity fast
  path, else O(nk·npw) equality check — noise next to the O(nk·npw²) build).
  `reindex` drops the cache rather than inheriting it.
- **Evidence:** the previous per-ctor build ran every SCF iteration (the
  Hamiltonian ctor does) and cost ~one FFT apply per iteration for nothing.
  The revalidation exists because a derived BatchedK
  (`dataclasses.replace` with a new flat_idx — the k+q reindex in
  `src/gradwave/postscf/dfpt_q.py`) inherits the parent's dict, and a table
  built for the parent's spheres is silently wrong physics on the derived
  one: the k+q Hamiltonian applies the wrong local term and the Sternheimer
  CG diverges to NaN.

## [D-007] CUDA batched tall-skinny QR: CPU offload gated by cols ≤ 16 AND a measured fp64 penalty ≥ 8

- **Date:** 2026-07 (PR #174 tune) / 2026-08 (H100 retest) · **Status:** active
- **Sites:** `src/gradwave/solvers/davidson.py` — `_QR_CPU_MAX_COLS`,
  `_QR_OFFLOAD_PENALTY_THRESHOLD`, `_qr_offload_active`, `_qr_offload`
- **Decision:** offload the (nk, npw, cols) reduced QR to CPU LAPACK only when
  cols ≤ 16 (size gate) and the device's measured fp64/fp32 GEMM ratio is ≥ 8
  (device gate; `GRADWAVE_QR_OFFLOAD` forces either way, read per call).
- **Evidence:** found while investigating CUDA-graph round capture (which was
  bit-identical at 1.0× — no launch-overhead gap anywhere in the round): this
  QR, isolated, was the single biggest cost in an RTX 3050 round, BIGGER than
  the two-FFT H-apply next to it (~3.9 ms vs ~2.3 ms at diamond-C, 50 Ry,
  nk=8, npw=465, cols=8); a D2H + CPU LAPACK QR + H2D round trip ran the same
  shape in ~0.3 ms (>10×) — cuSOLVER's fixed per-call batched geqrf/orgqr tax
  on a tiny problem, the same mechanism as issue #133's eigh offload. A sweep
  (nk 8..112, npw 465..2500, cols 8..64) put the clear-to-break-even boundary
  at cols ≤ 16 (worst 0.98×, typically 2-12×), mixed above. The fp64-penalty
  threshold is bracketed by an order of magnitude on each side: the RTX 3050
  measures ratios ~20-60 (offload a clear win), while the 4×H100 session
  (issue #206, benchmarks/results/h100-session) measured the offload as a
  ~13% PENALTY (Cr2O3: 37.1 s on vs 32.7 s off, identical energy to
  1.8e-10 meV/atom, identical 16 iterations) — a datacenter fp64 GPU
  (ratio ~1-2) has no cuSOLVER tax to escape. Threshold 8 clears both.

## [D-008] CholQR2 orthonormalization exists but defaults off

- **Date:** 2026-08 · **Status:** active (retest on datacenter GPUs)
- **Sites:** `src/gradwave/solvers/davidson.py` — `_cholqr_mode`, `_cholqr2`
- **Decision:** `GRADWAVE_CHOLQR=on` swaps the tall-skinny QR for CholQR2
  (Gram GEMM → fp64 Cholesky → triangular solve, twice; breakdown falls back
  to QR per call). Default off.
- **Evidence:** measured neutral on CPU (0.94-1.03× across the battery) and
  neutral-to-slightly-negative on RTX 3050 whole-SCF — the QR round is too
  thin a slice at these sizes to matter. Kept (GEMM-shaped, removes the
  D2H/H2D round trip of the QR offload entirely) for a datacenter-GPU retest.

## [D-009] Γ-point real-wavefunction path defaults OFF, not "auto"

- **Date:** 2026-08 · **Status:** active
- **Sites:** `src/gradwave/scf/loop.py` — `_gamma_real_mode`,
  `_resolve_gamma_real`; `src/gradwave/core/gamma.py`
- **Decision:** `GRADWAVE_GAMMA_REAL` defaults to "0" (complex path,
  byte-for-byte) even though "auto" is provably safe and exact whenever it
  engages. Opting in with "auto"/"1" is exact either way.
- **Evidence:** the SCF *iteration count* near the convergence boundary is
  not bit-reproducible between the two eigensolvers. Si's degenerate valence
  top under smearing has a gauge-ambiguous density from a partially-occupied
  degenerate subspace; the real embedded Davidson picks a different (equally
  valid) orthonormal basis than the complex batched Davidson, so the SCF
  residual differs at ~1e-10 — below the ~-30 eV energy agreement (machine
  precision vs the complex path) but right at the rhotol=1e-9 boundary. A
  warm start that saves one SCF iteration on the complex path can converge in
  the same count on the Γ path
  (tests/integration/test_calculator_warmstart_grid.py). The warm DENSITY
  seed is threaded identically on both paths — a boundary effect, not a
  missing warm start — so it cannot be cleanly removed by threading
  coefficients. H-apply telemetry is transparent across both paths
  (GammaHamiltonian bumps the shared core.batch tally), so that contract no
  longer blocks "auto"; the iteration-count nonreproducibility is the one
  remaining reason for the conservative default.

## [D-010] Per-band diagonalization tolerance: closed by prior measurement + shipped mechanisms

- **Date:** 2026-09 (survivors round) · **Status:** closed (do not re-propose)
- **Sites:** `src/gradwave/solvers/davidson.py` (expansion selection),
  `src/gradwave/solvers/native_davidson.py` (per-k retirement)
- **Decision:** no per-band stop-tolerance vector. What the idea actually
  buys is already shipped: the eager batched Davidson expands only the
  worst *unconverged* residuals each round (`n_add = (rn > tol)` count,
  worst-first selection), so converged bands cost only the Rayleigh-Ritz
  ride; the native solver retires whole k-points as they converge
  (`GRADWAVE_NATIVE_RETIRE`, default on).
- **Evidence:** measured closure 2026-09-02 (occupation-weighted tol):
  metals put the slowest band at E_F (occupied → no loosening headroom;
  Al n_occ=8/12), insulators are diluted below 5% whole-SCF by mixing
  domination + the adaptive diago-tol schedule already making early solves
  cheap. The remaining sliver (per-band locking within a k) targets the
  subspace-algebra share, which the survivors-round profile puts at ~2%
  (`_eigh_subspace`) on small cells — below any shipping bar.

## [D-011] Orbital (wavefunction) extrapolation across SCF/ionic steps: closed by prior measurement

- **Date:** 2026-09 (survivors round) · **Status:** closed
- **Sites:** `src/gradwave/scf/loop.py` `_seed_orbitals` (warm start),
  density extrapolation across sweep steps shipped separately (#475)
- **Decision:** no orbital-velocity extrapolation seed. Prior measurement
  (2026-09-02, 4-atom Al relax): per-geometry SCF iterations
  none=[12,11,11,11,11], density-reuse=[12,11,11,10,10], full quadratic
  density extrapolation=[12,11,10,10,9] — the SCF is mixing-convergence
  dominated; the orbital seed is a sub-effect of an already ~7% effect,
  and costs coefficient-history state (~GBs at slab scale) plus
  cross-geometry subspace alignment. The wavefunction analog of #475
  remains unbuilt on purpose.

## [D-012] Transform dedup (density-build ↔ warm-apply FFT reuse): skipped by Amdahl ceiling

- **Date:** 2026-09 (survivors round) · **Status:** closed (measured ceiling)
- **Sites:** `src/gradwave/core/batch.py` (`density_b`, `BatchedHamiltonian.apply`)
- **Decision:** not built. The proposed reuse — cache ψ(r) from iteration
  n's density build and skip the forward transform of the warm block in
  iteration n+1's first Davidson round — has a measured ceiling below the
  1.05× gate, before its costs (a (nk, nb, n1n2n3) complex cache, an
  ortho-skip flag to keep the warm block bit-identical to the density's).
- **Evidence:** asus 8-thread cProfile, Al-4 eager (14.0 s wall): ALL
  H-apply FFT work is 2.75 s (20%); the dedupable slice (round-1 forward
  transforms of the warm block ≈ the density build's `g_to_r_b` twin,
  0.60 s) is 2–4% end-to-end. Si2: smaller still (density_b = 0.05 of
  1.0 s). On the native-solver headline path the solver-side transforms
  live inside the C kernel and the duplicate cannot be threaded across —
  ceiling ~1%. Eager wall is glue/subspace algebra, not FFT.

## [D-013] Chord / frozen-C tail (single-RR solves in the mixing tail): parked — one-config win, losses elsewhere

- **Date:** 2026-09 (survivors round) · **Status:** parked (probe wiring REVERTED; implementation in branch history, commit 3af4ebe4 on perf/survivors-round)
- **Sites:** probed via `GRADWAVE_CHORD_TAIL` in `scf.loop` + `max_iter=1`
  solver rounds; the exact `it == max_iter` early break in
  `solvers/davidson.py` is the one piece kept (byte-identical results,
  strictly fewer applies on max_iter-limited solves)
- **Decision:** not shipped, wiring reverted. Once |Δρ| < X·rhotol the probe
  caps the eigensolve at one Rayleigh-Ritz round (frozen-C + subspace
  rotation + exact eigenvalue refresh — one nb-wide apply) and forces one
  full solve before any convergence claim (the stale-solve clause cannot
  see a chorded solve's unreached tol_eff).
- **Evidence (asus, 8 threads, min of reps; X=30/100 vs baseline):**
  Al-32 native: 162.5 → 156.7 / 149.5 s (+1 iter, E to 1e-10) = up to
  1.087×, the only win. Si-64 native: 283.8 → 283.6 (null, +1 iter).
  Small cells all lose — outer iterations grow faster than chorded solves
  save: Si2 native 0.468 → 0.489 s (+1 it), Al-4 native 4.22 → 4.98 /
  8.19 s (11 → 15/33 iters!), Fe-1 native 1.95 → 2.14 / 2.43 s (37 →
  40/51), eager likewise. A single ~9% regime-specific win under a
  size+metal gate does not meet the shipping bar (meaningful-broad-win
  directive, 2026-09-12); the frozen-C density response is the missing
  physics, and the mixer pays for it in iterations.

## [D-014] Thick Ritz buffer across outer iterations: parked — narrow eager-only win

- **Date:** 2026-09 (survivors round) · **Status:** parked (probe wiring REVERTED; implementation in branch history, commit 3af4ebe4 on perf/survivors-round)
- **Sites:** probed via `GRADWAVE_RITZ_TAIL` in `scf.loop` + an `n_gate`
  argument on `davidson_batched` (buffer rows ride the Rayleigh-Ritz,
  never gated/expanded)
- **Decision:** not shipped, wiring reverted. Carrying nb+2..8 Ritz vectors across outer
  iterations does cut H-apply work (band-vector applies drop 8–14%), but
  the wall-clock verdict is regime-dependent and eager-path-only (the
  native C solver, the CPU headline path, has no seed-buffer entry point).
- **Evidence (asus, 8 threads, eager, min of 3 reps, buffer 0/2/4/8):**
  Si2 1.051 → 0.997/0.944/0.919 s (1.14× at 8; hpsi 4792 → 4144; same 9
  iters; E identical). Fe-1 3.30 → 3.19/3.10/3.25 s (1.07× at 4; 38 → 37
  iters). Al-4 14.34 → 14.65/15.98/14.65 s (LOSS — wider RR/ortho glue
  outgrows the apply savings at nk=36-large batches). A win that only
  exists on the slower-by-3× eager arm of small insulators does not meet
  the shipping bar; revisit only if the native solver grows a seed-buffer
  interface AND the apply share climbs back above ~50% there.

## [D-015] Davidson glue retune (subspace depth, expansion-width cap): defaults confirmed; no new knob

- **Date:** 2026-09 (survivors round) · **Status:** closed (defaults stand)
- **Sites:** `solvers/davidson.py` (`max_dim_factor`, already user-facing via
  `scf.memory.max_dim_factor` / `GRADWAVE_MAX_DIM_FACTOR`); an expansion-width
  cap (`n_add_cap`) was probed and reverted (branch history, commit 3af4ebe4)
- **Decision:** keep `max_dim_factor=4` as the default; do not add an
  expansion-width cap. Tuning stays available through the existing
  `scf.memory` knob for users chasing a specific config.
- **Evidence (asus, 8 threads, min of 3 reps, factor 2/3/4/6):**
  n_add cap: iteration-count roulette, not a mechanism — Fe-1 eager
  38 iters → 24 (cap 2, 4.28 s LOSS) → 21 (cap 4, 2.73 s "win") → 41
  (cap 8, 3.67 s LOSS) vs 3.32 s baseline; Al-4 ±9%; Si2 null. The
  trajectory perturbation swamps the glue effect; nothing robust to ship.
  max_dim_factor: Si2 native 0.47/0.47/0.47/0.46 (null; eager mildly
  prefers 6: 0.96 vs 1.02), Al-4 native 5.10/4.26/4.28/5.93 (4≈3), Fe-1
  native 2.62/2.01/2.19/2.53 (~1.09× at 3), Al-32 native
  156/148/163–194/(6: see log) — metals mildly prefer 3, insulators are
  flat, eager Al-4 regresses at 3 (16.3 vs 15.0 s). A ~9–10% metal-only,
  native-only preference (confounded by late-session thermal drift on the
  Al-32 arms) is below the shipping bar for a default flip; the knob
  already exists for targeted use.
