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
