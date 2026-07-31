# Distributed k-point parallelism

gradwave's single-process path already batches every k-point into one set of
tensor ops (`core/batch.py`; see [Performance](performance.md)) — that
saturates one CPU/GPU's BLAS instead of looping k-points in Python. This page
covers the next step: splitting that batch of k-points **across ranks/machines**
with `torch.distributed` (Gloo), for k-meshes large enough that one box's
core count is the bottleneck.

## When it's worth using

- A **large k-mesh** (e.g. a metal or a small cell needing 8×8×8 or denser)
  where diagonalization time scales with the number of k-points and you have
  more than one machine's worth of cores available.
- **Not** worth it for a coarse mesh (Γ-only, 2×2×2) — process/collective
  overhead dominates and IBZ symmetry reduction (`symmetry: true`, on by
  default) already gives 5–14× for free on a single box; reach for that first.
- **Not** a substitute for `use_symmetry=True` — the two are currently
  mutually exclusive (see Scope below), so distributing an IBZ-reduced mesh
  isn't available yet; distributing the FULL mesh may still win over a
  symmetry-reduced single-box run once the mesh is large enough.

## What's distributed, and what isn't

k-points are embarrassingly parallel except at the mixing step, where the
per-k density contributions must be summed over the whole mesh — the exact
reduction `core/batch.py`'s `density_b` already performs *within* one rank's
batch (its docstring: batching "saturates BLAS/GPU instead of looping small
problems in Python"). Distribution extends that same reduction **across**
ranks. Concretely, each rank diagonalizes a disjoint, contiguous shard of
k-points, and three small collective calls stitch the results back together
every SCF iteration:

1. **Occupations** — a metal's Fermi level depends on every k-point's
   eigenvalues, so eigenvalues are gathered into a global array before the
   Fermi search; each rank keeps only its own slice of the resulting
   occupations.
2. **Density** — each rank's local `density_b` call already sums over its own
   k-shard; an `all_reduce` SUM across ranks completes the sum over the full
   mesh.
3. **Energy** — kinetic and nonlocal (projector) energy are sums over k,
   computed per rank and `all_reduce`-summed; every other term (Hartree, XC,
   local pseudopotential, Ewald, entropy) is a function of the
   already-global density/eigenvalues and comes out identical on every rank
   without further communication.

The converged `SCFResult` is reassembled at the end (one more small gather)
so it looks exactly like an ordinary single-process, full-mesh run —
`result.system`, `.eigenvalues`, `.occupations`, and `.coeffs` all cover the
whole k-mesh, not just one rank's shard.

See `src/gradwave/distributed.py`'s module docstring for the implementation
detail (it's short — read it if you're modifying this).

### Scope (v1)

Implemented for the norm-conserving **collinear** SCF (`scf.loop.scf`), via
`task: scf` and `task: bands` (which calls the same driver), and for the
USPP/PAW collinear driver (`scf.uspp_loop.scf_uspp`).
`distributed.shard_uspp_system` slices a built `USPPSystem` to a rank's shard,
and `scf_uspp(dist_ctx=...)` reduces the per-atom augmentation `becsum`
alongside the smooth density, DFT+U included (see
`tests/integration/test_distributed_uspp_scf.py` for the launch pattern).
An input file with `distributed: true` now routes either formalism.
`api.run_scf` shards a `USPPSystem` through `shard_uspp_system` and a
norm-conserving `System` through `shard_system`, DFT+U carried through
unchanged (see `tests/integration/test_distributed_uspp_api.py`). Also not yet
supported, and rejected with a clear `NotImplementedError` rather than silently
ignored:

- IBZ symmetry reduction (`symmetry: true`) — build with `symmetry: false`
  for a distributed run.
- The noncollinear/spinor SCF (`noncollinear: true`), and fully relativistic
  (SOC) pseudopotentials.
- Hybrid (PBE0/HSE) Fock exchange — it couples orbitals across k-points in
  ways beyond the density/energy reduction implemented here. DFT+U (Dudarev)
  IS supported. The Hubbard occupation matrix `n_hub` is a k-extensive sum,
  reduced the same way as the density (`all_reduce`-summed across ranks). Its
  energy term is then recomputed from the already-reduced `n_hub` rather than
  summed per rank, since it is a nonlinear function of `n_hub` (see
  `scf.loop._hubbard_occ_update` and `scf.uspp_loop._hubbard_occ_update`,
  which mirror each other on the two drivers).
- `task: relax | eos | elastic | phonons | magnetism` — these don't route
  through the k-point-sharded path yet.
- Warm-starting a distributed run from a checkpoint (`restart:`) produced by
  a *different* world_size/shard layout. A checkpoint's orbitals are matched
  to the local shard's k-count, so a mismatch is silently ignored (falls back
  to the default seed) rather than warm-starting.

These are real, scoped-out follow-ups, not oversights — extending each one
is additional, separable work (see the module docstring's per-item reasoning).

## How to opt in

Set `distributed: true` in the input YAML (same convention as `device:` —
see `inputs.Input`):

```yaml
distributed: true
kpoints:
  mesh: [8, 8, 8]
```

This alone does nothing without a multi-rank launch — outside one, gradwave
sees `WORLD_SIZE=1` and runs the ordinary single-process path unchanged.

## How to launch

Distributed mode is driven by `torchrun`'s environment variables
(`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`), read by
`gradwave.distributed.init_from_env()` exactly the way any other torchrun job
would. `scripts/gradwave_distributed.sh` wraps the common invocations.

**Single box, N ranks** (the case this repo's tests actually exercise, N=2):

```bash
scripts/gradwave_distributed.sh input.yaml --nproc-per-node 2
```

**Two machines over Tailscale**, one rank per machine — run the *same*
command on each box, changing only `--node-rank`. `--master-addr` must be
reachable from every node (use the Tailscale IP, not a LAN/loopback one), and
`GLOO_SOCKET_IFNAME` pins the collective traffic to the Tailscale interface
rather than whatever interface torchrun's rendezvous would otherwise pick —
the same thinkpad/asus Tailscale pairing this repo already uses for
benchmarking (see [Performance](performance.md) and `CLAUDE.md`'s "Remote
compute" section):

```bash
# box A (rank 0), Tailscale IP 100.x.y.z:
GLOO_SOCKET_IFNAME=tailscale0 scripts/gradwave_distributed.sh input.yaml \
    --nnodes 2 --node-rank 0 --master-addr 100.x.y.z

# box B (rank 1):
GLOO_SOCKET_IFNAME=tailscale0 scripts/gradwave_distributed.sh input.yaml \
    --nnodes 2 --node-rank 1 --master-addr 100.x.y.z
```

Every rank must run against the **same gradwave revision** and the **same
input file** — `shard_system` partitions k-points deterministically from
`(nk, rank, world_size)`, so a mismatch (different mesh, different world_size
on one side) silently produces the wrong system on that rank rather than
raising an obvious error.

Only rank 0 writes `<task>.json`/`.out`/`checkpoint.pt` (every rank computes
the identical, correctly-reduced result, so the others would just race to
write the same files).

## What's been validated, and what hasn't

- **Validated**: the density/occupation/energy reduction, at 2-rank scale on
  one box — `tests/unit/test_distributed.py` checks the reduction math
  directly (no process group needed: summing `density_b` over two
  independently-built k-shards reproduces the full-mesh `density_b` exactly),
  and `tests/integration/test_distributed_scf.py` spawns 2 real
  `torch.distributed` (Gloo) processes and checks the converged free energy,
  kinetic/nonlocal energy, density, eigenvalues, and occupations all match a
  plain single-process run on the same system to numerical tolerance.
- **Not validated in this repo's test suite**: an actual 2-*machine* launch
  over Tailscale (the `GLOO_SOCKET_IFNAME=tailscale0` case above). The code
  path is identical to the single-box 2-rank case (`init_from_env()` doesn't
  distinguish "two ranks on one box" from "two ranks on two boxes" — both are
  just Gloo TCP rendezvous), but CI has no second machine to launch onto, and
  this is a documented, not-yet-executed follow-up: run the multi-machine
  invocation above by hand and confirm the free energy against a
  single-process reference before relying on it for real production k-meshes.

## Multi-GPU

The k-point-sharded path runs across several CUDA devices without code changes.
Gloo stages CUDA tensors through host memory for each collective, so no NCCL
build is required and one rank per GPU works out of the box. On the 4x H100
session (issue #206) a 4-rank run produced energies bit-identical to the
single-rank result at 1e-11, so the reduction is correct across devices.

Scaling follows the same rule as the CPU case, per-k work has to outweigh the
collectives. A toy cell is slower than a single GPU (startup- and
collective-bound), while a real-scale Si-16 6×6×6 on 2 ranks reaches 3.2 s/iter
against about 16.8 s/iter amortized single-rank. Reach for it on k-heavy
systems, not on coarse meshes.

Two cautions from that session. Large multi-rank runs currently deadlock in the
post-SCF result reassembly (issue #216): the final gather pickles each rank's
list of large CUDA coefficient tensors through `dist.all_gather_object`, and
both ranks block. It is payload-dependent, so a small cell gathers fine while a
54³-grid, 28-k-per-rank shard hangs. Until the gather is replaced with a sized
raw-tensor `all_gather` staged through CPU, treat big multi-rank GPU runs as
blocked. Separately, cosmetic only: a distributed run built with
`symmetry: false` still prints a header like "56 k(IBZ)", because the count
shown is the time-reversal reduction on the local shard labelled with the IBZ
tag. The mesh is the full one, the label is wrong, nothing about the calculation
changes.
