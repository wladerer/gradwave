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

IBZ symmetry reduction (``use_symmetry=True``) composes with the sharding:
the shard unit becomes the IBZ k-list with its orbit weights, and the
symmetry operators need nothing extra from this module. ``rho_symmetrizer``
(and USPP's ``becsum_sym``) are built from the ops and the FFT box / atoms,
not the k-set, and both SCF drivers apply them AFTER the density/becsum
``all_reduce`` — at that point every rank holds the identical global
quantity, so the (deterministic) symmetrization is redundant per-rank work,
not a fourth collective. The Fermi search is likewise untouched: the
eigenvalue gather + the once-gathered global IBZ weights make the same
weighted count a single-process symmetric run performs. The only new
constraint is ``world_size <= nk_IBZ`` (the zero-share ValueError below).

Scope: the norm-conserving collinear SCF (``scf.loop.scf``) and the
USPP/PAW collinear SCF (``scf.uspp_loop.scf_uspp``, including DFT+U — see
:func:`shard_uspp_system`), both reachable from an input file via
``Input.distributed: true`` (``api.run_scf`` shards either formalism), with
or without IBZ symmetry reduction — no fully relativistic (SOC)
pseudopotentials (NC only; USPP/PAW has no SOC representation at all in this
codebase — see :func:`shard_uspp_system`'s docstring), no hybrid Fock
exchange, and no warm start across a shard boundary. The noncollinear/spinor
SCF and hybrid-under-distribution are documented follow-ups (see
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

    Rendezvous override: if ``GRADWAVE_DIST_INIT_METHOD`` is set it is used as
    the ``init_method`` instead of ``env://`` (with ``rank``/``world_size`` read
    from the env). This selects a ``FileStore`` (``file:///shared/path``) on a
    shared-filesystem launch, or a fixed ``tcp://host:port`` endpoint, without a
    ``MASTER_ADDR``/``MASTER_PORT`` pair. It is also what the distributed test
    harness uses: a unique per-test ``file://`` store is collision-proof, where
    a "free" TCP port drawn by ``bind(0)``/close is not (that TOCTOU race let two
    concurrent tests reuse one port and deadlock the ``env://`` rendezvous).

    Returns ``(rank, world_size, group)``, or ``None`` when ``WORLD_SIZE`` is
    absent or ``1`` (an ordinary single-process run — callers should fall back
    to the non-distributed path, not fail). Idempotent: a process group
    already initialized (e.g. by a test harness) is reused rather than
    re-initialized.
    """
    if not is_distributed_env():
        return None
    import torch.distributed as dist

    dist = cast("Any", dist)  # members are gated behind is_available(), loosely typed

    if not dist.is_initialized():
        init_method = os.environ.get("GRADWAVE_DIST_INIT_METHOD")
        if init_method:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=current_rank(),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
        else:
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

    A symmetrized system (``use_symmetry=True``) shards the same way: the
    per-k fields sliced here are already the IBZ set with its orbit weights,
    and ``sym``/``rho_symmetrizer`` ride along via ``dataclasses.replace``.
    Neither depends on the k-set — the symmetrizer is built from
    ``(FFT shape, ops, dens_mask)`` and the SCF applies it to the density
    AFTER the cross-rank ``all_reduce`` has made it global, so every rank
    applies the same deterministic operator to identical input and needs no
    extra communication (see the module docstring). ``world_size`` is bounded
    by the IBZ k-count, not the full mesh. Fully relativistic (SOC)
    pseudopotentials are excluded (the SCF driver that consumes them,
    ``scf_noncollinear``, is out of scope here too).
    """
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
            f"world_size ({world_size}) exceeds the k-point count "
            f"(after IBZ reduction, if enabled)"
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

    A symmetrized system shards the same way as in :func:`shard_system`, with
    one addition: ``becsum_sym`` (the per-atom augmentation symmetrization)
    also rides along. Like the density symmetrizer it depends on the ops and
    atoms, not the k-set, and ``uspp_loop._build_output_density`` applies it
    to the becsum AFTER the cross-rank ``all_reduce`` — identical global
    input on every rank, identical output, no extra communication.

    No fully-relativistic (SOC) guard here: ``USPPSystem`` has no ``is_fr`` /
    ``so_beta_tables`` fields at all — this codebase's only SOC pseudopotential
    path is the norm-conserving spinor driver (``scf_noncollinear``, consuming
    ``System``); the USPP/PAW noncollinear driver (``scf_uspp_noncollinear``)
    is out of scope here too, rejected upstream (``inputs.py`` and
    ``api.run_scf`` gate ``distributed`` to the collinear drivers, and
    ``scf_uspp`` rejects a magnetic symmetrizer under ``dist_ctx``).
    """
    nk = len(system.kweights)
    start, end = shard_range(nk, rank, world_size)
    if start == end:
        raise ValueError(
            f"rank {rank} would get zero k-points out of {nk} total — "
            f"world_size ({world_size}) exceeds the k-point count "
            f"(after IBZ reduction, if enabled)"
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


def shard_start_from(start_from: _T, ctx: DistKContext) -> _T:
    """Slice a FULL-mesh warm-start's per-k orbital coefficients down to this
    rank's k-shard ``[k_start, k_end)``, so the orbital seed matches the local
    system :func:`shard_system`/:func:`shard_uspp_system` built.

    The relax calculator and the EOS volume chain both warm-start each SCF from
    the previous, already-reassembled FULL-mesh result. That result's ``coeffs``
    carry every k-point, but the local shard's ``scf(..., dist_ctx=)`` seeds
    orbitals against a local system whose ``spheres``/``BatchedK`` cover only
    ``[k_start, k_end)`` — so an unsliced full-mesh ``coeffs`` fails the seed's
    k-count compatibility check and silently cold-starts every orbital
    (``scf.loop._seed_orbitals`` / ``scf.uspp_loop._seed_orbitals_uspp``),
    inflating the per-step SCF iteration count away from the single-process run.
    Slicing the per-k coefficient list here restores the exact per-k reuse.

    Only the per-k ``coeffs`` are sharded. Every other warm-start field is
    already global: the real-space density (``rho``/``rho_spin``, ``all_reduce``
    -summed each SCF iteration), the per-atom USPP augmentation ``rho_ij_atoms``,
    and ``system.grid`` (geometry-global, used only for the volume-ratio
    rescale in ``scf.common.warm_start_densities``). ``coeffs`` is a flat per-k
    list for nspin=1 and a ``[spin][k]`` list-of-lists for nspin=2; both layouts
    are sliced along the k axis. A ``None`` ``start_from`` (or one carrying no
    ``coeffs``) passes through unchanged."""
    if start_from is None:
        return start_from
    lo, hi = ctx.k_start, ctx.k_end

    def _slice(coeffs: Any) -> Any:
        if coeffs is None:
            return None
        # nspin=2 is a [spin][k] list-of-lists; nspin=1 is a flat per-k list.
        if len(coeffs) > 0 and isinstance(coeffs[0], list):
            return [ch[lo:hi] for ch in coeffs]
        return coeffs[lo:hi]

    if isinstance(start_from, dict):
        d = cast("dict[str, Any]", start_from)
        if d.get("coeffs") is None:
            return start_from
        out = dict(d)
        out["coeffs"] = _slice(d["coeffs"])
        return cast("_T", out)
    coeffs = getattr(start_from, "coeffs", None)
    if coeffs is None:
        return start_from
    return cast("_T", dataclasses.replace(cast("Any", start_from), coeffs=_slice(coeffs)))


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

    dist = cast("Any", dist)  # members are gated behind is_available(), loosely typed

    t = tensor if tensor.is_contiguous() else tensor.contiguous()
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=ctx.group)
    return t


def _gather_var_tensor_lists(
    local: list[torch.Tensor], world_size: int, group: Any, device: torch.device
) -> list[list[torch.Tensor]]:
    """``all_gather`` a per-rank list of tensors (ragged in shape/count across
    ranks) as raw bytes staged through CPU, returning one reconstructed list
    per rank IN RANK ORDER, each tensor placed back on ``device``.

    This is the deadlock-free replacement for pickling large (CUDA) tensors
    through ``all_gather_object`` (#216): ``all_gather_object`` serializes each
    rank's payload via ``_object_to_tensor``, and on multi-tens-of-MB CUDA
    coefficient lists over Gloo both ranks block there indefinitely. Here only
    a small metadata list (per-tensor shape + dtype) rides the object path; the
    bulk moves as a single padded ``uint8`` buffer through the plain-tensor
    ``all_gather`` collective, which does not pickle.

    Variable sizes are handled by exchanging each rank's byte count via a tiny
    ``all_gather``, padding every rank's buffer up to the global maximum, then
    slicing each rank's contribution back out using its own metadata. Dtypes
    are preserved exactly by reinterpreting the raw bytes (``view(dtype)``), so
    complex coefficients and real becp/eigenvalue arrays round-trip losslessly.
    """
    import torch.distributed as dist

    dist = cast("Any", dist)  # members are gated behind is_available(), loosely typed

    # 1. Metadata (shape, dtype) per local tensor — small, so the object path
    #    is fine here (this is NOT the large-payload pathology).
    local_meta = [(tuple(t.shape), t.dtype) for t in local]
    metas: list[list[tuple[tuple[int, ...], torch.dtype]] | None] = [None] * world_size
    dist.all_gather_object(metas, local_meta, group=group)

    # 2. Pack this rank's tensors into one contiguous CPU uint8 byte buffer.
    parts = [
        t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
        for t in local
        if t.numel() > 0
    ]
    flat = torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)

    # 3. Exchange byte counts and pad every rank's buffer to the global max.
    nbytes = torch.tensor([flat.numel()], dtype=torch.int64)
    sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(sizes, nbytes, group=group)
    maxlen = max(int(s.item()) for s in sizes)

    per_rank: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
    if maxlen == 0:
        return per_rank  # every rank contributed an empty list

    padded = torch.zeros(maxlen, dtype=torch.uint8)
    padded[: flat.numel()] = flat
    gathered = [torch.zeros(maxlen, dtype=torch.uint8) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)

    # 4. Slice each rank's byte buffer back into tensors using its metadata.
    for r in range(world_size):
        rank_meta = metas[r]
        assert rank_meta is not None
        buf = gathered[r]
        off = 0
        for shape, dtype in rank_meta:
            numel = 1
            for d in shape:
                numel *= d
            nb = numel * torch.empty(0, dtype=dtype).element_size()
            chunk = buf[off : off + nb].clone().view(dtype).reshape(shape)
            per_rank[r].append(chunk.to(device))
            off += nb
    return per_rank


def gather_cat(local: torch.Tensor, ctx: DistKContext, dim: int = 0) -> torch.Tensor:
    """``all_gather`` ``local`` from every rank and concatenate along ``dim``,
    IN RANK ORDER — reconstructs a global k-ordered array (eigenvalues,
    kweights) from contiguous per-rank shards. Ranks need not share the exact
    same local shape, and CUDA tensors round-trip through the same code path as
    CPU ones: the payload is staged as raw CPU bytes (see
    :func:`_gather_var_tensor_lists`), not pickled through
    ``all_gather_object`` (#216), then moved back onto ``local``'s device."""
    per_rank = _gather_var_tensor_lists(
        [local.detach()], ctx.world_size, ctx.group, local.device
    )
    tensors = [chunk[0] for chunk in per_rank]
    return torch.cat(tensors, dim=dim)


def gather_list_cat(local: list[_T], ctx: DistKContext) -> list[_T]:
    """``all_gather`` a per-rank Python list and concatenate in rank order into
    the global list.

    Two payload kinds flow through here. Lists of tensors (the per-k
    coefficient / ⟨β|ψ⟩ arrays, ragged in shape across k) are the large,
    possibly-CUDA payloads that deadlocked ``all_gather_object`` (#216); they
    take the raw-byte, CPU-staged path in :func:`_gather_var_tensor_lists` and
    come back on the local tensors' device. Lists of small Python scalars (the
    Stoner spin-preconditioner ``picks`` metadata tuples) are tiny and stay on
    the object path, where pickling is harmless. A one-shot boolean
    ``all_gather_object`` agrees on the kind across ranks first, so every rank
    enters the same collective sequence even when a rank's local list is empty
    (avoiding a path-divergence deadlock)."""
    import torch.distributed as dist

    dist = cast("Any", dist)  # members are gated behind is_available(), loosely typed

    is_tensor_local = bool(local) and isinstance(local[0], torch.Tensor)
    flags: list[bool | None] = [None] * ctx.world_size
    dist.all_gather_object(flags, is_tensor_local, group=ctx.group)

    if any(flags):
        tensors = cast("list[torch.Tensor]", local)
        device = tensors[0].device if tensors else torch.device("cpu")
        per_rank = _gather_var_tensor_lists(tensors, ctx.world_size, ctx.group, device)
        out: list[_T] = []
        for chunk in per_rank:
            out.extend(cast("list[_T]", chunk))
        return out

    buf: list[list[_T] | None] = [None] * ctx.world_size
    dist.all_gather_object(buf, local, group=ctx.group)
    out_obj: list[_T] = []
    for chunk in buf:
        assert chunk is not None
        out_obj.extend(chunk)
    return out_obj


def maybe_destroy_process_group() -> None:
    """Tear down the default process group if one is initialized (a distributed
    torchrun launch); a no-op otherwise. Safe to call unconditionally at the
    end of a run — single-process paths never initialize a group, so they never
    touch it. Idempotent: a second call after teardown does nothing."""
    import torch.distributed as dist

    dist = cast("Any", dist)  # members are gated behind is_available(), loosely typed

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
