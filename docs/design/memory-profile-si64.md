# SCF memory profile: Si-64, and where the peak actually lives

Campaign task: kill the SCF memory hog that OOMs / pages a large-cell run. This
note records the profile that drove the fix, so a future reader does not
re-derive it. The headline: **the Davidson subspace is not the hog — the eager
batched dense-grid FFT boxes are.**

## System and method

Si-64 = a 2×2×2 repetition of the 8-atom conventional diamond cell (64 Si),
`Si_ONCV_PBE-1.2.upf` (valence-only), LDA, `smearing="none"`, 2×2×2 MP mesh
folded to **`n_k = 4`** irreducible k-points (Fd-3m), **`nb = 128`** (the exact
insulator occupied count, matching the QE recipe of #474). Eager `davidson`
solver (the default; the opt-in `davidson-native` currently segfaults at this
shape — see the closing note). Run on asus (22 cores, 14 GB RAM, ~12 GB free),
8 threads.

Instrumentation: a background thread sampling `/proc/self/status` `VmRSS` every
20 ms, tagged with the currently-active instrumented phase (the hot allocation
functions — `density_b`, `projectors_b`, `local_potential_g`, `_solve_bands` —
are wrapped to mark phase boundaries and record the RSS delta across each call).
Peak RSS recurs every SCF iteration, so a 4–5 iteration run captures the
steady-state peak; these numbers are the true per-run peak.

Two shapes:

| ecut | grid | npw_max (m) | n_grid | unchunked density box | full-subspace apply box |
|---|---|---|---|---|---|
| 15 Ry | 54³ | 8 480 | 157 464 | 1.20 GiB | `n_k·4nb·n_grid·16` = 4.8 GiB |
| 30 Ry | 72³ | 23 984 | 373 248 | 2.85 GiB | **12.2 GiB** |

## Profile verdict — phase → peak RSS → top allocation sites

**Si-64 / 30 Ry, unchunked (main).** Peak RSS **11 665 MB (11.39 GB)**.

| phase | peak RSS while active (MB) | dominant allocation |
|---|---|---|
| **solve** (Davidson) | **11 665** | dense FFT box in the H-apply local term, over up to `4·nb=512` subspace vectors: `n_k·512·n_grid·16` ≈ 12 GiB (band-chunked → bounded) |
| **density** build | 10 068 | `g_to_r_b` scatter box + FFT, `(n_k, nb, n_grid)` complex128 = 2.85 GiB ×~3 live copies |
| projectors setup | 4 390 | `projectors_b` all-k phase gather transient, **0.73 GiB measured** (749.8 MB delta) |
| veff (local pp) | 4 563 | `local_potential_g` structure factors `(na, n_grid)` = 0.36 GiB |

Analytic transient sizes (30 Ry, complex128) — all confirmed against the RSS
deltas:

```
projectors_b all-k gather (n_k, npw, nproj)  0.732 GiB   (measured 749.8 MB)
proj_phase_free resident   (n_k, nproj, npw) 0.732 GiB   (BatchedK, whole SCF)
local_pp structure factors (na, n_grid)      0.356 GiB
density psi box unchunked  (n_k, nb, n_grid) 2.848 GiB   (×~3 in g_to_r_b)
Davidson full-subspace apply box             12.2 GiB    (the OOM)
```

Suspects from prior sessions, adjudicated by the profile:

- **(a) batched FFT intermediates in `core/batch.py`** — **CONFIRMED, the #1
  hog.** The H-apply local-term box and the density box, both materialized over
  all bands at once on CPU (the CPU dense-box budget defaulted to *unchunked*),
  are the two phases that peak >10 GB.
- **(b) per-k setup tables scaling with `n_k`** — **CONFIRMED but secondary.**
  `projectors_b` (0.73 GiB) and `local_pp` (0.36 GiB) transients are real and
  measured, but they peak in the setup/veff phases (~4.4 GB), well below the
  binding solve peak. Resident projector data entering the solve is ~2.9 GB
  (proj_phase_free + the phased projectors + the `_tables` `p`/`p.conj()` cache).
- **(c) native adapter all-k copies** — **N/A.** The native solver segfaults at
  Si-64 (both RR modes), so the eager path is the operative one; its adapter
  copies never run.
- **(d) allocator not returning freed blocks** — **refuted as a driver.** The
  RSS timeline oscillates *down* between phases (e.g. solve swings 3.6 ↔ 8.0 GB
  every ~1 s at 15 Ry), so the torch CPU caching allocator does return / reuse
  freed blocks. Peak reduction has to come from shrinking the single largest
  live temporary, not from `del` ordering.

## The lever, measured

Band-chunking the CPU dense boxes (the existing `GRADWAVE_CPU_DENSE_BUDGET`
machinery, bit-identical for the apply, ~ulp for the density reduction) is the
decisive fix. `k_chunk` is nearly irrelevant here: with `n_k=4` it can only
quarter the per-solve subspace, and each chunk still runs full-band applies —
the dense box, not the subspace, is the hog.

| config (Si-64 / 30 Ry, 4 iters) | peak RSS | wall |
|---|---|---|
| unchunked (main) | 11 665 MB (11.39 GB) | 566 s |
| `CPU_DENSE_BUDGET=5e8` | 9 108 MB (8.89 GB) | 345 s |
| `CPU_DENSE_BUDGET=2e8` | 8 371 MB (8.17 GB) | 342 s |
| `5e8` + `max_dim_factor=2` | 8 504 MB (8.30 GB) | 347 s |

Chunking does not cost wall time (it removes allocator/paging pressure), and the
solve-phase floor (~8.2 GB) is set by the resident projector tables + the eager
Davidson subspace, not the dense box.

### Converged headline (native solver, exact)

Once the native `.so` was rebuilt (see the closing note), the `davidson-native`
solver runs Si-64 and is the memory-lightest path (its C eigensolve streams per
k). The decisive exactness + memory result — auto dense-box chunking ON (branch
default) vs OFF (`GRADWAVE_CPU_DENSE_AUTO=off`, the pre-fix behaviour), **same
solver, full convergence**:

| Si-64 / 30 Ry, native, converged | iters | peak RSS | E (eV) |
|---|---|---|---|
| auto-chunk **OFF** (unchunked) | 23 | **10.64 GB** | −6852.55482302 |
| auto-chunk **ON** (branch) | 23 | **6.39 GB** | −6852.55482302 |

Identical iteration count, energy identical to all 8 printed digits — the
density band-sum's ~ulp reorder does not move the converged fixed point. Peak
RSS cut 40% with no change to the result. (Eager `davidson` at 4 iters: 11.4 GB
unchunked → 8.3 GB chunked, same trend; peak recurs every iteration so the
4-iter peak is the run peak.)

### Al-32, the many-k stressor (correctness)

A 32-atom fcc Al supercell (`nk=10` IBZ, `nb=212`, 63³ grid, metal) has an
unchunked dense box of **33.9 GB** — an unconditional OOM on any workstation.
With auto chunking it completes at 10.7 GB peak. This is the slab-relevant
regime (many bands × large grid) where the win matters most.

## The fix

1. **Auto CPU dense-box budget** (`scf.loop._auto_cpu_dense_budget` →
   `core.batch.set_cpu_dense_budget_override`): when the unchunked box would
   exceed ~half of available RAM, cap each dense temporary (default target ~3e8,
   clamped) and band-chunk the H-apply / density. Small/medium cells stay
   byte-for-byte unchunked. Overwrite semantics per `scf()` call → no cross-call
   leak. `dense_budget_gb` (Input) / `GRADWAVE_CPU_DENSE_BUDGET` (env, bytes)
   force it; `GRADWAVE_CPU_DENSE_AUTO=off` disables it.
2. **`projectors_b` / `local_potential_g` no-grad streaming**: materialize the
   projector phase per k and the structure factors per atom, dropping the 0.73 /
   0.36 GiB all-k / all-atom transients. Guarded to `no_grad`; the differentiable
   forces / alchemical path keeps its clean batched autograd node.

## What remains vs QE's 0.83 GB

QE does the same physics in 0.83 GB; the chunked gradwave run is ~8 GB. The gap
is structural, not a leak: gradwave keeps the full padded `(n_k, nb, npw_max)`
plane-wave blocks and several resident copies of the `(n_k, nproj, npw_max)`
projector table (phase-free, phased, and the `p`/`p.conj()` working cache — ~2.9
GB together at 30 Ry) live for the whole SCF, where QE streams per-k from disk /
recomputes. Closing it further means shrinking the resident projector footprint
(share one copy, drop the cached conjugate under memory pressure) and the padded
subspace storage — a deeper change than this bit-exact chunking pass.

## Note: a STALE native `.so` (not a code bug) — rebuild after #469/#477

While profiling, `GRADWAVE_EIGENSOLVER=davidson-native` **segfaulted (SIGSEGV)**
on Si-64 at both cutoffs in both RR modes, AND two `tests/unit/test_native_solver.py`
tests failed the fast tier identically on `main` and this branch. Root cause: the
cached `~/.cache/gradwave/libdavnative.so` on the box was built **2026-09-09**,
before #469/#477 changed `davidson_native.c`'s signature (#477 added the
`rr_mode` argument). The Python adapter called the stale library with a
mismatched ABI → memory corruption → segfault. **This is a stale build artifact,
not a code bug.**

Rebuilding it (`scripts/build_native_solver.sh`) fixed both the segfault and the
fast-tier failures, and the native solver then runs Si-64 cleanly and is the
memory-lightest path (the converged headline above). Operational takeaway: the
native `.so` must be rebuilt whenever `davidson_native.c` changes — a machine
with a stale `.so` shows a red fast tier and a segfaulting native solver. A
future hardening could stamp the `.c` hash into the build and have the adapter
refuse a mismatched library instead of segfaulting.
## Projector-table dedup: one BLAS-folded conjugate instead of a resident copy

Follow-up to the profile above (campaign task #24). The #479 note flagged, as
the deepest remaining resident cost after dense-box chunking, "several resident
copies of the `(n_k, nproj, npw_max)` projector table (phase-free, phased, and
the `p`/`p.conj()` working cache — ~2.9 GB together at 30 Ry)". This pass
adjudicates that claim and removes the one copy that was pure redundancy.

### What is actually resident, and where

At Si-64 / 30 Ry a single `(n_k=4, nproj, npw_max=23984)` complex128 projector
table is ~0.73 GiB. Three forms appeared in the accounting:

1. **`bk.proj_phase_free`** — the position-independent generator
   `f_ylm · e^{-iG·(r_atom-origin)}`-free table, resident on `System.batch`.
   It is the source `projectors_b(bk, positions)` phases at the fixed atomic
   positions, and it is **required after the SCF loop**: `postscf.forces` (and
   the alchemical path) rebuild the phased projectors *with grad* from it. Not
   redundant.
2. **The phased table `projs_b`** — `projectors_b` evaluated once at the fixed
   positions, handed to `BatchedHamiltonian` as `p` and consumed every apply.
   The native C solver reads exactly this (`h.p` → `cx->p`) for the whole
   solve, so it is unavoidably resident during a native solve. Not redundant.
3. **`p.conj()` cache** — `BatchedHamiltonian._tables` cached
   `p.conj().resolve_conj()`, a full second table, for the H's lifetime. This
   existed **only because `torch.einsum` cannot fold a conjugation into the
   BLAS call** — the becp contraction `⟨β|ψ⟩ = Σ_g conj(p) c` was written as
   `einsum("kpg,kbg->kbp", p.conj(), c)`, and materializing `p.conj()` fresh
   every Davidson round (many applies) was slower than caching it once. Pure
   redundancy in the information sense: it is a bitwise function of (2).

The native path never built (3): `davidson_native.c`'s KB term is a
`cblas_zgemm(..., CblasConjTrans, ...)` against `cx->p` — the conjugation is a
BLAS flag, no second table. So the "~2.9 GB" figure is the *eager*-path
accounting; a **native** Si-64 solve holds only (1)+(2) ≈ 1.46 GiB of projector
tables plus whatever the C kernel's per-thread becp scratch adds.

### The change

Make the eager path mirror the C kernel. `becp_b` now computes
`torch.matmul(c, p.conj().transpose(-2,-1))`; on CPU the conj-transpose *view*
folds into the cgemm call (ConjTrans), so there is no materialized conjugate —
neither resident nor per-round transient. `BatchedHamiltonian._tables` drops the
cached conjugate (returns `(t, v_eff, p, dij)`); the USPP overlap `_pq` cache
drops its conjugate the same way. The retained `becp_b(p, c, p_conj=...)`
argument is accepted but ignored.

The contraction is mathematically identical; `tests/unit/test_dense_chunk_streaming.py`
pins `becp_b` bit-level (≤1e-14) against the pre-dedup
`einsum(p.conj().resolve_conj(), c)` in complex128 and ≤1e-6 in complex64, plus
the grad-path preservation. Scope is the norm-conserving collinear
`BatchedHamiltonian` (which both Si-64 and Cu/Al slabs use) and the trivial
USPP overlap; the spinor/SOC noncollinear path keeps its own `p_conj`/`q_conj`
caches (a separate follow-up — it is neither the Si-64 nor the slab regime).

### A stale-`.so` guard, while here

The #479 note asked for it: the build script now bakes a source hash of
`davidson_native.c` into the library (`-DGW_NATIVE_SRC_HASH`, stringized), and
the adapter (`solvers/native_davidson.py`) hashes the checked-out source at load
and **refuses a mismatched library with a clear rebuild error** instead of
calling a stale ABI and segfaulting. A library predating the stamp reports
`"unknown"` and is treated as stale.

### Measured (asus, 8 threads, this branch vs its base `main`@3c6a4327)

Converged unless marked; E agrees to ALL printed digits (8 decimals) and the
iteration counts are identical in every A/B pair — the conj-fold does not move
any fixed point. "eager4" arms are 4 fixed iterations (the eager peak recurs
every iteration, and converged eager Si-64 is ~2.3× slower than native, so 4
iterations bound the peak honestly).

| case | arm | peak RSS | wall | iters | E (eV) |
|---|---|---|---|---|---|
| Si-64 / 30 Ry native | main | 6.321 GB | 729.5 s | 28 | −6852.55482302 |
| | **branch** | **6.290 GB** | 731.4 s | 28 | −6852.55482302 |
| Si-64 / 30 Ry eager, 4 it | main | 8.935 / 8.952 GB | 234.8 / 233.3 s | 4 | −6848.51911433 |
| | **branch** | **8.070 / 8.142 GB** | 243.4 / 243.6 s | 4 | −6848.51911433 |
| Si-64 / 15 Ry native | main | 5.791 GB | 287.0 s | 37 | −6847.81040259 |
| | **branch** | 5.796 GB | 287.1 s | 37 | −6847.81040259 |
| Al-32 / 30 Ry native | main | 3.457 GB | 170.8 s | 13 | −59901.24050616 |
| | **branch** | 3.454 GB | 171.4 s | 13 | −59901.24050616 |
| Al-4 guard (native) | main | 0.995 GB | 4.5 s | 11 | −7487.65506327 |
| | **branch** | 1.005 GB | 4.5 s | 11 | −7487.65506327 |
| Al-1 guard (native) | main | 0.669 GB | 0.4 s | 9 | −1871.87393042 |
| | **branch** | 0.671 GB | 0.5 s | 9 | −1871.87393042 |

Reading:

- **Eager Si-64: −0.83…0.87 GB peak (the conj table + allocator slack), at a
  reproducible +4% wall** (two independent A/B pairs, order alternated). The
  wall cost is the PyTorch *conjugate fallback*: CPU matmul cannot consume a
  conj view directly, so it materializes `p.conj()` as a transient **per
  apply** (~40 applies × 0.73 GiB of memcpy over a 4-iteration run ≈ the
  observed +10 s). The transient lives under the solve-phase peak, which is
  why peak still drops by the full resident copy.
- **Native: unchanged in RSS and wall** (±0.03 GB, ±0.3%), as predicted — the
  C kernel never held the copy. The dedup's only native-path effect is
  removing the per-iteration transient conj in the energy-assembly `becp_b`.
- **Small cells: unchanged** (guards identical A/B) — nproj is tiny, the
  fallback is noise.

Verdict on the design choice: **(b) keep the per-k phased table, drop the conj
cache and nothing else** — option (a) (phase-free only, re-phase per use) was
not taken: it would add the same fallback-materialization to every consumer
including the native adapter boundary, for at most one more table (~0.73 GiB)
of savings, and the phase-free table cannot be dropped anyway (it is the
autograd source for forces). The +4% eager wall on projector-heavy cells is
accepted and documented: eager is the fallback path at this scale (converged
eager Si-64 ≈ 1700 s vs 731 s native), and under memory pressure −0.87 GB is
worth more than +4% wall. If it ever matters, the surgical fix is a
`negative()`-on-the-imag-view kernel or a fused conj-GEMM, not a resident copy.

## Slab memory profile (Part 2): what binds after #479 + the dedup

Same RSS-timeline + phase instrumentation, on the slab regime the prior
campaigns quoted at 7.5–10.4 GB (12–16 atoms + realistic vacuum). Configs
(4×4×1 MP → nk=6, 30 Ry, LDA, gaussian 0.1, 10 Å vacuum both sides):
Cu(100) 2×2×3 (12 atoms, ONCV 19e ⇒ nb=140, grid 35×35×160, npw_max 11 569),
Al(100) 2×2×4 (16 atoms, ONCV 11e ⇒ nb=110, grid 40×40×175, npw_max 16 051).

| slab case | arm | peak RSS | wall | phase peaks (solve / density / projectors) |
|---|---|---|---|---|
| Cu-12 eager, 8 it | main | 5.845 GB | 355.3 s | 5.85 / 3.25 / 1.39 GB |
| | branch | **5.648 GB** | 363.6 s | 5.65 / 3.26 / 1.37 GB |
| Al-16 eager, 8 it | main | 5.950 GB | 360.9 s | 5.95 / 3.27 / 1.33 GB |
| | branch | **5.784 GB** | 361.9 s | 5.78 / 3.25 / 1.37 GB |
| Cu-12 native, converged (71 it) | main | 3.683 GB | 987.0 s | 3.68 / 3.50 / 1.41 GB |
| | branch | **3.662 GB** | 978.2 s | 3.66 / 3.48 / 1.38 GB |

E identical to all 8 digits in every pair (eager at fixed 8 iterations,
native converged: −57917.57771247 eV Cu, −29946.36808510 eV Al at 8 it).

**Phase → peak → top sites, and the verdict.** On slabs of this size, with
current main's auto dense-box chunking active:

1. The **dense FFT box is no longer the binding term** — it is chunked, and
   the whole dense-grid field set is small anyway (n_grid ≈ 0.2–0.3 M points;
   a complex128 scalar field is ~3–4 MB, so even the ~60–64% vacuum fraction
   of the grid is memory-noise at this scale, ~1–2 dozen MB of waste).
2. **Projector residue is minor here**: one table is 0.18–0.22 GiB (nproj
   128–216); the dedup trims ~0.17–0.20 GB (~3% of peak).
3. The binding term is the **per-k padded solve state**. Eager peaks 5.6–5.9 GB
   entirely inside the solve phase: the grown Davidson subspace over all k at
   once (`nk · 4nb · npw_max` complex128 for V and HV — ~0.6 GiB *each* for
   these slabs, ~2.4 GiB with W/HW — plus the projected matrices), on top of the resident
   padded wavefunction block (0.15 GiB) and density/mixing state (~3.2 GB
   plateau). The native solver, which streams the eigensolve per k, holds its
   converged peak at 3.66–3.68 GB — only ~0.2 GB above the density-phase
   plateau. The eager−native gap (≈2.1 GB) IS the all-k subspace.

**Streaming-SCF v1 recommendation: deprioritized as a new build.** What a
streaming loop would stream first is the per-k solve state — and that already
exists twice: the native solver streams it in C (measured: the slab peak is
within 0.2 GB of the no-solver floor), and the eager path has `k_chunk`
(`scf.memory`) for the same effect. The remaining ~3.2 GB slab floor is
density/mixing history + padded wfc + the resident tables — dominated by
Pulay/Broyden mixing history on the dense grid and the wavefunction block,
each individually well under 1 GB. Nothing left at this scale justifies a new
streaming subsystem; revisit only for the 100+-atom slab regime, where the
padded `(nk, nb, npw_max)` wavefunction block itself (the term the dedup
campaign explicitly scoped OUT) becomes the next structural rung, as the #479
"what remains vs QE" section already concluded.

One more slab datum: these 12–16-atom eager peaks are **5.6–5.9 GB where the
pre-#479 campaigns measured 7.5–10.4 GB** — the dense-box chunking (#479) plus
the incremental-RR native default (#477) already moved the slab ceiling, which
is exactly why the streaming build lost its urgency.
