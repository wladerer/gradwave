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

## [D-015] Davidson glue retune: no expansion-width cap; eager default max_dim_factor=4 stands

- **Date:** 2026-09 (survivors round) · **Status:** closed
- **Sites:** `solvers/davidson.py` (`max_dim_factor`, user-facing via
  `scf.memory.max_dim_factor` / `GRADWAVE_MAX_DIM_FACTOR`); an expansion-width
  cap (`n_add_cap`) was probed and reverted (branch history, commit 3af4ebe4)
- **Decision:** do not add an expansion-width cap; the EAGER solver keeps
  `max_dim_factor=4`. The one robust finding — the native solver prefers 3 —
  ships as the native adapter's default ([D-016]).
- **Evidence (asus, 8 threads, min of 3 reps, factor 2/3/4/6):**
  n_add cap: iteration-count roulette, not a mechanism — Fe-1 eager
  38 iters → 24 (cap 2, 4.28 s LOSS) → 21 (cap 4, 2.73 s "win") → 41
  (cap 8, 3.67 s LOSS) vs 3.32 s baseline; Al-4 ±9%; Si2 null. The
  trajectory perturbation swamps the glue effect; nothing robust to ship.
  max_dim_factor on the eager path: Si2 0.96/1.06/1.02/0.96, Al-4
  17.4/16.3/15.0/19.1 (4 best, 3 REGRESSES), Fe-1 5.8/3.6/3.3/3.1 —
  no consistent winner ≠ 4.

## [D-016] Native Davidson defaults to max_dim_factor=3 (eager stays 4)

- **Date:** 2026-09 (survivors round) · **Status:** active
- **Sites:** `src/gradwave/solvers/native_davidson.py`
  (`native_davidson_adapter`, `max_dim_factor=None` → 3 native / 4 on any
  eager fallback); overrides unchanged (`GRADWAVE_MAX_DIM_FACTOR` /
  `scf.memory.max_dim_factor` layer over it in `_resolve_max_dim_factor`)
- **Decision:** the native C solver's wall is subspace algebra
  (Rayleigh-Ritz/ortho), not H-applies, so a shallower grown subspace —
  more frequent but cheap restarts — wins where the solve is expensive and
  is a null elsewhere. Exact by construction (restart cadence changes the
  path, not the converged eigenpairs) and cuts the V/HV subspace bytes 25%.
- **Evidence (asus, 8 threads; interleaved A/B for the headline pairs,
  min of reps; E agreement 1e-10 or better):** Al-32 native 162.3/162.6 →
  145.9/144.9 s (**1.12×**, same 13 iters), Si-64 15 Ry native 285.0 →
  266.0 s (**1.07×**, 38 → 35 iters, same E), Fe-1 native 2.19 → 2.01 s
  (1.09×), Si2 and Al-4 native null (0.467→0.469, 4.28→4.26). No native
  config loses. The eager solver is NOT flipped: eager Al-4 regresses at 3
  (15.0 → 16.3 s — larger apply share, deeper subspace amortizes it).

## [D-017] TRIM antiunitary realification: exact at every TRIM, but the broad win is already shipped (Γ) and the non-Γ extension is narrow

- **Date:** 2026-09 (higher-mathematics performance moonshot, door-closer
  probes) · **Status:** active
- **Sites:** `src/gradwave/core/gamma.py`, `src/gradwave/scf/loop.py`
  (`_resolve_gamma_real`, `_gamma_real_mode`);
  `benchmarks/moonshot_doorclosers/probe_a1_trim_commute.py`,
  `benchmarks/moonshot_doorclosers/probe_a2_gamma_real.py`
- **Decision:** do NOT build a non-Γ TRIM realification path. At a
  time-reversal-invariant momentum (2k ≡ G0 a reciprocal-lattice vector) the
  complex Hermitian eigenproblem is provably the complexification of a real
  symmetric one, and transporting to the real form halves the bandwidth-bound
  subspace bytes — but the broad case (Γ-only: molecules, defects, large
  supercells) is ALREADY the shipped Γ real-wavefunction path, and the only
  novel content (general TRIM k) helps just the ~8 TRIM points of a k-mesh
  (broadly useful only for all-TRIM coarse meshes / Γ-only, which is already
  covered). The genuinely-unexploited lever is the shipped Γ path's OFF
  default, adjudicated separately — see [D-009] and its re-adjudication.
- **Evidence (asus, 8 threads):** *A1 (correctness, exact):* the antiunitary
  J = P∘K (P the G-shift permutation G → −G−G0, K conjugation) commutes with
  the SHIPPED H-apply (`core.batch.BatchedHamiltonian`) to machine precision
  at every TRIM of an unshifted 2×2×2 Si mesh — ‖JHc−HJc‖/‖Hc‖ = 2.8–3.5e-17
  at Γ AND at the zone-boundary points (½,0,0), (½,½,0), (½,½,½), with
  J²=+1. So realification is mathematically available at general TRIM in this
  basis, not just Γ. *A2 (economics of the shipped Γ path, Si-64 Γ-only
  30 Ry, subspace-dominated, locked A/B pairs, exact dE=0.0):* the shipped Γ
  real path (`GRADWAVE_GAMMA_REAL=1`) runs 275.2 s (15 it) → 135.4 s (13 it)
  and 276.4 s (15 it) → 134.9 s (13 it) — a robust **~2.04× e2e speedup
  INCLUDING the half↔full conversion** (~1.77× per iteration + a reproducible
  15→13 iteration reduction). Conversion does NOT eat the win, and it is not
  the default. So the "realification is an unexploited perf lever" claim
  fails on scope, not on physics: the 2× is real but already banked as an
  opt-in for the broad workload; the buildable remainder (non-Γ TRIM) is
  narrow.
- **Rejected:** building the general-TRIM real path (a G-shift-permutation
  eigensolver for zone-boundary k) — narrow (mesh TRIM subset only) for a
  large code investment, does not clear the broad-class bar. The high-value
  follow-up is the Γ default policy (off→auto), which is NOT this record's to
  make — see [D-009] and the separate re-adjudication task.

## [D-018] s-step (Chronopoulos-Gear) reassociation of the incremental Rayleigh-Ritz: dead — the target ZGEMMs are compute-bound at Si-64 dims

- **Date:** 2026-09 (moonshot door-closer probes) · **Status:** active
- **Sites:** `src/gradwave/solvers/native/davidson_native.c` (the two
  incremental hc/sc ZGEMMs, `cblas_zgemm(ColMajor, ConjTrans, NoTrans, …)`
  building `hc[:,dim:newdim]` and `sc[:,dim:newdim]`, ~lines 658-663);
  `benchmarks/moonshot_doorclosers/probe_b_sstep_traffic.py`
- **Decision:** do NOT patch the incremental-RR kernel to generate s Krylov
  directions per Rayleigh-Ritz. The reassociation would divide the streamed
  V-bytes per useful flop by ~s (higher arithmetic intensity), which is the
  right medicine for a bandwidth-bound kernel — but these two ZGEMMs are NOT
  bandwidth-bound at Si-64 subspace dimensions, so the traffic saving does
  not convert to wall time.
- **Evidence (asus idle, 8 threads, real Si-64/30 Ry dims npw=23871,
  newdim=628; GFLOP/s vs panel width p, the per-round `notcnv`):** the model
  bytes-per-flop drops ~s× as claimed (0.503 → 0.253 → 0.128 → 0.066 → 0.035
  at p = 4,8,16,32,64 — arithmetic identity, V-read amortized), BUT measured
  GFLOP/s is 89.9 (p=4, AI≈2 flop/byte) then plateaus 156.8 / 158.9 / 187.0 /
  191.4 / 171.5 / 186.0 across p = 8,16,32,64,128,256 (AI 4 → 90). The only
  bandwidth-starved regime is p ≤ 4; every realistic cegterg panel width
  (notcnv ≈ 8–128) sits on the compute-bound plateau, where fattening the
  panel buys ≈0. The Si-64 "43% bandwidth-bound subspace algebra" cost
  therefore lives in the OTHER subspace operations (the generalized
  eigensolve, the Ritz rotation V·C, orthogonalization, density build), not
  in these two GEMMs — s-step aimed at them misses.
- **Rejected:** s ∈ {2,4} panel fattening of the hc/sc build — model traffic
  drops but the operation is compute-bound at the operating point, so no wall
  win for a substantial kernel rewrite (Chronopoulos-Gear also risks
  orthogonality loss). Existence test only; no kernel change made.

## [D-019] CheFSI-for-memory: dead — the Si-64 "10 GB working set swaps" premise is stale (fixed by [D-003] band-chunking)

- **Date:** 2026-09 (moonshot door-closer probes) · **Status:** active
- **Sites:** `src/gradwave/core/batch.py` (`_dense_band_chunk`, the
  band-chunked dense-box transient — see [D-003]);
  `docs/design/memory-profile-si64.md`;
  `benchmarks/moonshot_doorclosers/probe_c_si64_rss.py`
- **Decision:** do NOT pursue Chebyshev-filtered subspace iteration as a
  memory lever (shrinking a supposedly-swapping [V,HV] working set). The
  premise is stale on two counts: the [V,HV] subspace is only ~1.5 GB at
  Si-64/30 Ry (never the ~10 GB hog), and the real hog — the eager dense-grid
  FFT box — is already band-chunked to a byte budget ([D-003], PR #479). With
  that fix shipped the run fits RAM with headroom and does not inherently
  page, so CheFSI reverts to being the already-measured-negative
  CheFSI-as-solver (see the static-subspace-solver history), not a new
  memory win.
- **Evidence (asus, shipped main, Si-64 = 2×2×2 diamond, 30 Ry, 2×2×2 MP →
  4 IBZ k, npw_max=23984, grid 72³, nb=128, 8 threads):** peak RSS (VmHWM)
  **7998 MB (7.81 GB) eager `davidson`** and **6247 MB (6.10 GB)
  `davidson-native`** — both well under the 11.39 GB unchunked peak recorded
  in `memory-profile-si64.md` and under the 15 GB box, with no paging
  attributable to the run in isolation (a brief vmstat si/so blip was
  co-tenancy with a concurrent campaign momentarily exceeding available RAM).
  Two side findings: [D-003]/#479's band-chunking win is intact (no
  regression), and `davidson-native` NO LONGER segfaults at Si-64/30 Ry — the
  `memory-profile-si64.md` closing note that it does is now stale.
- **Rejected:** CheFSI as a memory-footprint reducer — its target working set
  is neither the largest transient nor a swapping one on shipped main.
