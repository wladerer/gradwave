# Phonon config-batching — scope

**Goal.** Run a phonon force-constant campaign's N independent displacement SCFs as
one *config-batched* SCF — folding the config axis into the plane-wave batch dimension
`core/batch.py` already uses for k-points — instead of N separate SCFs (today: serial,
or `SeedPool` across worker processes). The payoff is GPU saturation (one big
`(N·nk, nb, npw_max)` solve fills an H100 where one spoke's `nk` does not) and, unlike
`SeedPool`, an **autograd-compatible** path (SeedPool is forced-serial for differentiable
campaigns because autograd graphs don't cross a process boundary).

## Why phonons are the clean first target
The 6·N_prim displacement configs share the **identical** supercell, ecut, k-mesh —
therefore identical G-sphere, kinetic operator, and *position-free* projector tables
(`BatchedK.{npw,mask,flat_idx,kpg,t,proj_phase_free,dij_full}` are the same for every
config). Only the atomic positions differ. Positions enter the Hamiltonian in exactly
two places, both cheap to make per-config:
- **KB projector phase** — `projectors_b(bk, positions_c)` applies `e^{-i(k+G)·τ_c}` to
  the shared `proj_phase_free`. Already a pure function of positions; batches trivially.
- **local potential `v_loc`** — the per-config structure factor `S_c(G) = Σ e^{-iG·τ_c}`
  times the species form factor. Per-config `v_eff_c` then carries Hartree + XC of the
  per-config density.

So no ragged-sphere padding across configs (EOS volumes / elastic strains change the
cell → different spheres → would need the ragged-k padding trick; that's a later step).

## What folds vs what is per-config
| quantity | shared across configs | per-config |
|---|---|---|
| G-sphere / kinetic `t` / `mask` / `flat_idx` | ✅ | |
| position-free projector table, `dij` | ✅ | |
| projector phases `p_c` | | ✅ (`projectors_b`) |
| `v_eff_c(r)` (v_loc + Hartree + XC) | | ✅ |
| density `ρ_c(r)`, Fermi `μ_c`, occupations | | ✅ |
| mixing history, convergence flag | | ✅ |

Fold as batch axis `(N·nk)`: `bk` fields tiled N×; `p = cat(p_c)`; `v_eff` becomes a
per-batch-row field. Then one `davidson_batched` / `BatchedHamiltonian.apply` /
`density_b` runs the whole set.

## Required changes (prototype → full)
1. **`BatchedHamiltonian.apply` per-row `v_eff`.** Today `v_eff` is one `(n1,n2,n3)`
   field broadcast over all batch rows (line `psi * v_eff`). Generalize to
   `v_eff` of shape `(batch, n1,n2,n3)` (each config's `nk` rows share that config's
   field), i.e. `psi * v_eff[cfg_of_row]`. Small, local change; identical FLOPs.
2. **A config-batched forward SCF driver** (new module, does NOT touch `scf/loop.py`).
   Orchestrates, all with a leading config axis:
   - build per-config `v_eff` (v_loc structure factor per config + shared Hartree/XC
     kernels applied per config),
   - one folded `davidson_batched` over `(N·nk)`,
   - per-config Fermi (`find_fermi` looped or batched over configs) + occupations,
   - per-config `density_b` (slice the folded batch into config k-blocks),
   - per-config mixing (N independent mixers or a batched mixer) + convergence,
   - **per-config freezing**: configs converge at different rates; drop converged
     configs from the batch (or mask them) so the batch shrinks — else the whole
     batch pays the slowest config's iteration count (the main batching overhead).
3. **Forces per config** → force constants. Either reuse `postscf.forces.forces` per
   config on reconstructed per-config result objects, or a config-batched force.
4. **Autograd**: keep the graph intact (no `no_grad`, no process boundary) so the
   differentiable phonon/inverse-design campaigns get the same acceleration — the
   feature `SeedPool` structurally cannot provide.

## Risks / open questions
- **Per-config freezing bookkeeping** is the fiddly part — without it the batch runs at
  the slowest config's rate, eroding the win.
- **Correctness parity**: the config-batched forces must match the serial spoke forces
  to ~µeV/Å; the v_eff assembly (v_loc phase, Hartree, XC) is where drift hides.
- **CPU vs GPU**: at one SCF's `nk` the batched-k path may already saturate CPU BLAS, so
  folding configs could be ~neutral on CPU and only win on GPU. **This is measured first**
  (`benchmarks/phonon_batch/config_batch_probe.py`) before committing to the driver —
  see the go/no-go below.

## Go/no-go gate (the probe)
`config_batch_probe.py` isolates the eigensolve lever: N separate `davidson_batched` vs
one folded `(N·nk)` solve, per-config eigenvalue correctness checked. If the folded solve
is materially faster (esp. on GPU), the driver build is justified; if it is ~neutral even
on GPU, config-batching is not the lever and `SeedPool` process-parallelism stays the
answer. Results recorded below.

### Measured (asus, 2026-08-13) — NO-GO on current hardware

Si 2³ supercell (na=16), ecut 20 Ry. `config_batch_probe.py`:

| run | shape | separate | folded | speedup | correctness |
|---|---|---|---|---|---|
| CPU fp64, kmesh 2² | nk=8, 12 cfg → 96 batch | 126.0 s | 274.1 s | **0.46×** | Δeig 7e-14 eV (exact) |
| GPU fp32, Γ-only | nk=1, 24 cfg → 24 batch | 8.28 s | 7.76 s | **1.07×** | Δeig 2e-4 eV (fp32 noise) |

**The folding is provably correct** (bit-exact eigenvalues in fp64). But config-batching
is **neutral-to-negative on every accessible device**:
- **CPU: 2.2× slower.** At one SCF's `nk` the batched-k path already saturates BLAS, so
  folding adds no throughput — and without per-config freezing the folded solve runs
  every config to the *slowest* one's convergence.
- **GPU (RTX 3050): ~neutral (1.07×).** The win only exists when a single spoke
  *under-fills* the GPU. Realistic phonon supercells are large enough that one Γ-point
  solve already fills a 6 GB 3050, so there is no idle to fold into. (Smaller spokes
  would show a win but aren't the use case.)

**The premise — one spoke under-fills a large GPU, folding N configs fills it — is
specifically an H100-class claim and is untestable on the 3050** (too small: it's already
filled by one realistic spoke, and OOMs on the 96-batch fp32 fold). So:

**Verdict: DO NOT build the full config-batched SCF driver yet.** On everything we can
measure it is a wash or a loss. The single measurement that would justify the build — an
H100 run showing one phonon spoke under-fills the device and a config-fold fills it — has
to come first. Until then, `SeedPool` process-parallelism remains the right campaign lever
(and, per the H100 plan's Test 4, GPU-serial spokes beat CPU-parallel once each spoke is
large). Config-batching's one unique advantage that survives — an **autograd-compatible**
batched campaign that SeedPool structurally cannot offer — is the reason to revisit it on
H100, not to build it blind now.
