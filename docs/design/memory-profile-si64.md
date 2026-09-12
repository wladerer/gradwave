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

## Note: the native Davidson segfaults at Si-64

`GRADWAVE_EIGENSOLVER=davidson-native` (the `davidson_native.c` kernel, #469 /
#477) **segfaults (SIGSEGV) within seconds of the first solve** on Si-64 at both
15 and 30 Ry, in *both* RR modes (`classic` and `incremental`), reproducibly and
outside any profiling harness (`n_k=4, nb=128, m∈{8480,23984}, nproj=512`). This
is a pre-existing bug independent of this campaign; it means the native solver —
which the memory campaign expected to be the default at this size — cannot
currently run Si-64, and all numbers here are the eager `davidson` default. The
memory fixes are solver-agnostic (they live in the shared eager H-apply /
density path). The native crash is filed separately.
