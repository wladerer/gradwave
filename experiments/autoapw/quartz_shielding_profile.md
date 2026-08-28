# Quartz / MgO GIPAW shielding: profile + trustworthy Si+O σ_iso on the CG backend

Two deliverables. **(1)** a kernel-level profile of the GIPAW shielding response
solve, and the finding that a *full* shielding eval cannot be torch-profiled on a
14 GB box. **(2)** the trustworthy per-site Si and O isotropic shielding of
α-quartz on the ecut-stable CG backend (PR #401).

Run on `asus` (22 cores, 14 GB), `OMP_NUM_THREADS=8`.

## Deliverable 1 — profiling the shielding response solve

### A full shielding eval cannot be torch-profiled on 14 GB (measured)

torch.profiler retains an in-RAM event buffer for every op inside the
`profile()` context. A full GIPAW shielding eval issues *millions* of aten ops
(per-k × per-q × per-band × per-site CG/Sternheimer resolvent solves). That
buffer overflows the box in every configuration measured:

| config | outcome |
|---|---|
| full profiler (`with_stack`, `record_shapes`, `profile_memory` all on) | OOM-killed, **13.1 GB** anon-RSS (systemd-oomd) |
| `light=True` (`with_stack`/`record_shapes` off, `profile_memory` on) | OOM-killed, **12.5 GB** anon-RSS |

The un-instrumented eval is under 1 GB — the 12–13 GB is entirely the profiler's
trace, not the workload. `light` mode (a new `deep_profile` flag, added here)
drops the two per-op trace multipliers but the raw op-event **count** is the
wall: `profile_memory` alone still overflows. Conclusion: profile a single
kernel invocation, not the whole eval (documented in `docs/profiling.md`).

### The right granularity: one CG Sternheimer solve

`experiments/autoapw/shielding_profile_workload.py` profiles ONE
`_SMetricResolventCG.apply` — a single matrix-free S-metric CG resolvent solve,
the inner kernel the full shielding drives thousands of times. Context (a cheap
MgO SCF + one dense per-k S-metric context) is built once, unprofiled; only the
single solve is profiled. MgO is the cell the CG reroute was validated on (PR
#401). `deep_profile(wl, n_timed=5, warmup=1, light=True, command=None)`.

Size: **npw = 283, nocc = 4, cg max_iter = 100** (random conduction-space RHS →
runs the full budget; structural, not a converged solve).

**Top ops by self-CPU time** (the resolvent's compute):

| op | self-CPU | calls | note |
|---|---|---|---|
| `aten::mm` | 2.19 ms | 48 | dense H·x / S·x GEMMs — the CG matrix-free apply |
| `aten::einsum` | 1.18 ms | 147 | S-metric occupied projections ⟨u_o\|S·⟩ |
| `aten::bmm` | 0.77 ms | 147 | batched projection matmuls |
| `aten::mul` | 0.71 ms | 301 | CG update / scaling |
| `aten::linalg_vector_norm` | 0.38 ms | 23 | CG residual norms |

(`cudaDeviceSynchronize`, 96 ms, n=1, is a one-time CUDA-context/profiler
artifact from CUDA being available on asus — not per-op work.)

**Top ops by self memory** (the solve's transient bytes):

| op | self-mem | note |
|---|---|---|
| `aten::mul` | 3.55 MB | elementwise CG vectors |
| `aten::empty_strided` | 2.22 MB | working-buffer allocation |
| `aten::add` | 2.08 MB | CG search-direction / solution update |
| `aten::sub` | 1.34 MB | residual update |
| `aten::mm` | 0.87 MB | GEMM outputs |

**Where the memory lives:** a single solve's transient footprint is ~10 MB —
elementwise CG update vectors (`mul`/`add`/`sub`) plus the dense GEMM outputs
(`mm`). It is tiny. This is *why* the full eval's memory wall is the **count** of
solves accumulated in the profiler's op buffer, not any single solve's own
footprint: the resolvent is compute-bound on small dense GEMMs, not
memory-bound. The memory/speed levers for the whole eval are therefore (a) fewer
profiled ops (kernel-level profiling, done here) and (b) the `chunk_k`
per-k streaming already in the driver — not a per-solve memory reduction.

Artifact: `summary.html` + `op_table.{json,parquet}` + `trace.json.gz` under
`benchmarks/results/asus/<sha>/shielding-cg-kernel/`.

## Deliverable 2 — trustworthy α-quartz Si + O σ_iso on the CG backend

`experiments/autoapw/quartz_sigma_cg.py`, `gipaw` level, `response_backend`
auto → the cond(S) gate routes the hard-augmentation O sites onto the ecut-stable
matrix-free CG resolvent (PR #401). On the OLD dense-eigh path the O σ_iso
diverged with ecut (the #392/#399 trust boundary); on CG it is sane and stable.

k 2×2×2, `chunk_k=1`. Two ecut rungs for O ecut-stability.

_(numbers pending the run; see `quartz_sigma_*.json` / `quartz_sio.log`.)_
