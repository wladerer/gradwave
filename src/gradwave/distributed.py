"""Distributed k-point parallelism (torch.distributed / Gloo) — opt-in.

k-points are embarrassingly parallel except at the mixing step, where the
per-k density contributions must be summed over the WHOLE k-mesh.
``core/batch.py``'s ``density_b`` already performs that reduction *within* one
rank's batched k (its own docstring: batching "saturates BLAS/GPU instead of
looping small problems in Python"). This module extends the SAME reduction
ACROSS ranks/machines: each rank diagonalizes a disjoint, contiguous shard of
k-points and the results are stitched back together at three small collective
points, all handled by :func:`shard_system` plus a few call sites in
``scf.loop.scf``:

1. **Occupations.** A metal's Fermi level depends on the eigenvalues of EVERY
   k-point, not just this rank's shard, so eigenvalues are ``all_gather``-ed
   into a global array before ``scf.common.shared_fermi_occupations`` runs;
   each rank then keeps only its own slice of the resulting occupations.
2. **Density.** Each rank's local ``core.batch.density_b`` call already sums
   over its own k-shard (weighted by kweights); an ``all_reduce`` SUM across
   ranks completes the sum over the full mesh.
3. **Energy.** Of the ``EnergyBreakdown`` terms, kinetic and nonlocal
   (projector) energy are sums over k — computed per rank on the local shard,
   then ``all_reduce``-summed. Every other term (Hartree, XC, local
   pseudopotential, Ewald, entropy) is a function of the ALREADY-global
   density/eigenvalues and so comes out identical on every rank without
   further communication.
4. **DFT+U (Dudarev).** The Hubbard occupation matrix ``n_hub`` is built by
   ``core.hubbard.occupation_matrices`` from exactly the same
   k-weighted-sum pattern as the density — a k-extensive sum, computed per
   rank on the local shard, then ``all_reduce``-summed (see
   ``scf.loop._hubbard_occ_update``). Its energy term ``e_hub`` is NOT itself
   k-extensive-linear (``hubbard_energy`` is the NONLINEAR Tr[n(1−n)] of
   n_hub), so it is recomputed from the already-reduced, full-mesh n_hub
   rather than summed per rank like kinetic/nonlocal energy.
5. **The Stoner spin preconditioner (``spin_precond=True``,
   ``scf.spin_precond``).** Unlike the Kerker/Thomas-Fermi/learned-multipole
   density-space preconditioners — which act purely on the already-global
   mixing residual and so need no communication at all, just redundant
   per-rank application of the same operator — the Stoner preconditioner's
   ingredients (Fermi-surface band codensities) are inherently per-k. Two
   collectives keep every rank's operator identical: an ``all_gather`` of
   the cheap scalar pick metadata (which (spin, k, band) carries
   Fermi-surface weight, and how much) to agree on the same global
   top-``max_bands`` selection everywhere, then an ``all_reduce`` to
   assemble the selected bands' (u, w) operator rows — each row is built
   by the one rank that owns that k-point and left zero on every other
   rank, so SUM-reducing reconstructs the full operator on every rank. See
   ``scf.spin_precond.build_stoner_precond``/``build_stoner_precond_nc``.

Opt in via ``Input.distributed: true`` (``api.run_scf``/``api.run``) or by
calling :func:`init_from_env` and :func:`shard_system` directly and passing
the resulting context to ``scf.loop.scf(..., dist_ctx=...)``.

Launch with ``torchrun`` (see ``docs/manual/distributed.md`` and
``scripts/gradwave_distributed.sh``) — RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
are read from the environment by :func:`init_from_env`, same as any other
torchrun job. The backend is Gloo (CPU collectives); a cross-machine launch
over Tailscale additionally needs ``GLOO_SOCKET_IFNAME=tailscale0`` set in the
environment of every rank (see docs).

Scope: the norm-conserving collinear SCF (``scf.loop.scf``) and the
USPP/PAW collinear SCF (``scf.uspp_loop.scf_uspp``, including DFT+U — see
:func:`shard_uspp_system`), both reachable from an input file via
``Input.distributed: true`` (``api.run_scf`` shards either formalism) — no IBZ
symmetry reduction (``use_symmetry=False``
is required on either path), no fully relativistic (SOC) pseudopotentials (NC
only; USPP/PAW has no SOC representation at all in this codebase — see
:func:`shard_uspp_system`'s docstring), no hybrid Fock exchange, and no warm
start across a shard boundary. The noncollinear/spinor SCF and
hybrid-under-distribution are documented follow-ups (see
``docs/manual/distributed.md``), not implemented here.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch

if TYPE_CHECKING:
    from gradwave.scf.loop import System
    from gradwave.scf.uspp_setup import USPPSystem

_T = TypeVar("_T")


@dataclass
class DistKContext:
    """One rank's view of a k-point-sharded distributed SCF.

    Built by :func:`shard_system` (NC) or :func:`shard_uspp_system`
    (USPP/PAW); threaded through ``scf.loop.scf`` / ``scf.uspp_loop.scf_uspp``
    as the ``dist_ctx`` argument. ``full_system`` is the ORIGINAL, unsharded
    system this rank was given — kept around so the converged
    ``SCFResult``/``USPPResult`` can be reassembled with a normal, full-mesh
    ``system``/``eigenvalues``/``occupations``/``coeffs`` (as if it had been
    an ordinary, single-process run), rather than leaking the local shard to
    callers.
    """

    rank: int
    world_size: int
    group: Any  # torch.distributed.ProcessGroup, or None for the default group
    k_start: int  # this rank's slice into the GLOBAL k-ordered arrays
    k_end: int
    nk_global: int
    full_system: "System | USPPSystem"


def current_rank() -> int:
    """This process's rank under torchrun (``0`` outside any distributed
    launch, so file-writing code can gate on it unconditionally)."""
    return int(os.environ.get("RANK", "0"))


def is_distributed_env() -> bool:
    """True when torchrun-style launch env vars indicate a >1-rank job.
    ``WORLD_SIZE`` absent (or ``"1"``) means an ordinary single-process run."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_from_env(backend: str = "gloo") -> tuple[int, int, Any] | None:
    """Initialize the default process group from torchrun's environment
    (``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT`` — the
    ``env://`` rendezvous ``torch.distributed`` reads by default).

    Returns ``(rank, world_size, group)``, or ``None`` when ``WORLD_SIZE`` is
    absent or ``1`` (an ordinary single-process run — callers should fall back
    to the non-distributed path, not fail). Idempotent: a process group
    already initialized (e.g. by a test harness) is reused rather than
    re-initialized.
    """
    if not is_distributed_env():
        return None
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), dist.group.WORLD


def shard_range(n: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous ``[start, end)`` block of ``n`` k-points for ``rank`` — an
    as-even-as-possible split (the first ``n % world_size`` ranks get one
    extra k-point). Contiguous, not round-robin, so concatenating every
    rank's gathered arrays IN RANK ORDER reconstructs the original global
    k order exactly."""
    base, rem = divmod(n, world_size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def shard_system(
    system: "System", rank: int, world_size: int, group: Any
) -> tuple["System", DistKContext]:
    """Slice a fully-built ``System`` to this rank's contiguous k-shard.

    Only the per-k fields move (``spheres``, ``kweights``, ``proj_data``, and
    the batched ``BatchedK`` rebuilt from them); everything geometry-global
    (grid, positions, species, charges, ``vloc_tables``, ``n_electrons``,
    ``nbands``) is untouched and identical on every rank.

    Requires an unsymmetrized system (``use_symmetry=False``): IBZ folding and
    the ``rho_symmetrizer`` it builds couple k-points across the whole mesh in
    a way this k-sharding does not yet account for (v1 scope; see module
    docstring). Fully relativistic (SOC) pseudopotentials are excluded for the
    same reason (the SCF driver that consumes them, ``scf_noncollinear``, is
    out of scope here too).
    """
    if system.sym is not None or system.rho_symmetrizer is not None:
        raise NotImplementedError(
            "distributed k-point parallelism does not yet support "
            "use_symmetry=True (IBZ reduction) — build the system with "
            "use_symmetry=False for a distributed run"
        )
    if system.is_fr:
        raise NotImplementedError(
            "distributed k-point parallelism does not yet support "
            "fully relativistic (SOC) pseudopotentials"
        )
    nk = len(system.kweights)
    start, end = shard_range(nk, rank, world_size)
    if start == end:
        raise ValueError(
            f"rank {rank} would get zero k-points out of {nk} total — "
            f"world_size ({world_size}) exceeds the k-mesh size"
        )

    from gradwave.core.batch import build_batched

    spheres = system.spheres[start:end]
    proj_data = system.proj_data[start:end]
    local = dataclasses.replace(
        system,
        spheres=spheres,
        kweights=system.kweights[start:end],
        proj_data=proj_data,
        batch=build_batched(spheres, proj_data, device=system.positions.device),
    )
    ctx = DistKContext(
        rank=rank,
        world_size=world_size,
        group=group,
        k_start=start,
        k_end=end,
        nk_global=nk,
        full_system=system,
    )
    return local, ctx


def shard_uspp_system(
    system: "USPPSystem", rank: int, world_size: int, group: Any
) -> tuple["USPPSystem", DistKContext]:
    """Slice a fully-built ``USPPSystem`` to this rank's contiguous k-shard.

    Only the per-k fields move (``spheres``, ``kweights``, ``proj_data``, and
    ``smooth_flat_idx`` when the dual-grid H-apply is in use); everything
    geometry/species-global (grid, positions, ``paws``, augmentation tables
    ``aug``/``q_full``, ``atom_slices``, ``vloc_tables``, the density-sphere
    fields ``sphere_idx``/``g_sphere``, ``n_electrons``, ``nbands``) is
    untouched and identical on every rank. Unlike NC's ``System``,
    ``USPPSystem`` carries no batched ``BatchedK`` field to rebuild here —
    ``scf.uspp_loop._build_iter_ops`` builds it fresh from ``spheres``/
    ``proj_data`` every call, so slicing those two is enough.

    Requires an unsymmetrized system (``use_symmetry=False``): IBZ folding and
    the ``rho_symmetrizer``/``becsum_sym`` it builds couple k-points (and, for
    becsum, atoms) across the whole mesh in a way this k-sharding does not yet
    account for (same v1 scope as :func:`shard_system`).

    No fully-relativistic (SOC) guard here: ``USPPSystem`` has no ``is_fr`` /
    ``so_beta_tables`` fields at all — this codebase's only SOC pseudopotential
    path is the norm-conserving spinor driver (``scf_noncollinear``, consuming
    ``System``); the USPP/PAW noncollinear driver (``scf_uspp_noncollinear``,
    out of scope here too) takes a magnetic ``rho_symmetrizer``/``becsum_sym``
    instead, already excluded by the symmetry guard above.
    """
    if (
        system.sym is not None
        or system.rho_symmetrizer is not None
        or system.becsum_sym is not None
    ):
        raise NotImplementedError(
            "distributed k-point parallelism does not yet support "
            "use_symmetry=True (IBZ reduction) — build the system with "
            "use_symmetry=False for a distributed run"
        )
    nk = len(system.kweights)
    start, end = shard_range(nk, rank, world_size)
    if start == end:
        raise ValueError(
            f"rank {rank} would get zero k-points out of {nk} total — "
            f"world_size ({world_size}) exceeds the k-mesh size"
        )

    local_spheres = system.spheres[start:end]
    smooth_flat_idx = system.smooth_flat_idx
    if smooth_flat_idx is not None:
        # smooth_flat_idx's column width was built as
        # max(s.miller.shape[0] for s in <the FULL-mesh spheres>) (uspp_setup.
        # _build_smooth_grid) -- the same quantity scf.uspp_loop._build_iter_ops
        # independently recomputes as bk.npw_max (core.batch.build_batched's
        # `int(npw.max())`), but over just THIS RANK's local_spheres. The two
        # need not agree once sharded (a rank's shard can easily miss the
        # single k-point that carries the global-max plane-wave count), and
        # BatchedHamiltonian requires them to match exactly -- so re-truncate
        # the column width down to the LOCAL max here. Row-slicing already
        # keeps every real (nonzero-padded) entry of the kept rows; this only
        # drops always-zero padding columns beyond the local max.
        local_npw_max = max(s.npw for s in local_spheres)
        smooth_flat_idx = smooth_flat_idx[start:end, :local_npw_max]
    local = dataclasses.replace(
        system,
        spheres=local_spheres,
        kweights=system.kweights[start:end],
        proj_data=system.proj_data[start:end],
        smooth_flat_idx=smooth_flat_idx,
    )
    ctx = DistKContext(
        rank=rank,
        world_size=world_size,
        group=group,
        k_start=start,
        k_end=end,
        nk_global=nk,
        full_system=system,
    )
    return local, ctx


def all_reduce_(tensor: torch.Tensor, ctx: DistKContext) -> torch.Tensor:
    """SUM ``all_reduce`` — the density / k-extensive-energy reduction point.
    Works for any tensor shape shared identically across ranks (the dense
    density grid, a 0-d energy scalar, or a Hubbard occupation-matrix
    diagonal block sliced out of a bigger tensor).

    Non-contiguous input is made contiguous first: a Gloo ``all_reduce`` on a
    non-contiguous VIEW (e.g. ``n_full[start:start+dim, start:start+dim]``,
    a diagonal block of a bigger square matrix) does not raise — it silently
    reduces the wrong bytes, treating the view as ``numel()`` contiguous
    elements starting at its base pointer rather than respecting its actual
    strides (caught by a real distributed DFT+U correctness test; see
    ``scf.loop._hubbard_occ_update``). When a copy was needed, the returned
    tensor is a DIFFERENT object from the input — not truly in-place in that
    case — so callers should use the return value rather than rely on the
    input being mutated; every current call site already does
    (``x = all_reduce_(x, ctx)``)."""
    import torch.distributed as dist

    t = tensor if tensor.is_contiguous() else tensor.contiguous()
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=ctx.group)
    return t


def gather_cat(local: torch.Tensor, ctx: DistKContext, dim: int = 0) -> torch.Tensor:
    """``all_gather`` ``local`` from every rank and concatenate along ``dim``,
    IN RANK ORDER — reconstructs a global k-ordered array (eigenvalues,
    kweights) from contiguous per-rank shards. Uses ``all_gather_object`` so
    ranks need not share the exact same local shape and CUDA tensors round-trip
    through the same code path as CPU ones."""
    import torch.distributed as dist

    buf: list[torch.Tensor | None] = [None] * ctx.world_size
    dist.all_gather_object(buf, local.detach().cpu(), group=ctx.group)
    # every slot was just filled by all_gather_object (one entry per rank)
    tensors = cast("list[torch.Tensor]", buf)
    return torch.cat(tensors, dim=dim).to(local.device)


def gather_list_cat(local: list[_T], ctx: DistKContext) -> list[_T]:
    """``all_gather`` a per-rank Python list (e.g. the per-k coefficient
    tensors, ragged in shape across k) and concatenate in rank order into the
    global per-k list."""
    import torch.distributed as dist

    buf: list[list[_T] | None] = [None] * ctx.world_size
    dist.all_gather_object(buf, local, group=ctx.group)
    out: list[_T] = []
    for chunk in buf:
        assert chunk is not None
        out.extend(chunk)
    return out
