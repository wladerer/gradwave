"""gradwave.distributed's pure (no torch.distributed process group needed)
pieces: k-shard partitioning and the shard_system slicing it feeds.

The actual cross-rank collective calls (all_reduce/all_gather) need a real
process group and are exercised instead by the heavier, standard-tier 2-rank
subprocess test tests/integration/test_distributed_scf.py. This file checks
the two things that don't need one:

1. shard_range's partition is exact, contiguous, and covers every k-point
   exactly once (the property gather_cat/gather_list_cat's "concatenate in
   rank order" reconstruction relies on).
2. Summing core.batch.density_b over two SEPARATELY-BUILT shard BatchedK
   objects (built the same way shard_system builds one per rank) reproduces
   the density_b of the full, unsharded system exactly -- the actual physics
   claim behind the all_reduce at the mixing step, checked here without any
   process-group machinery at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.core.batch import density_b
from gradwave.distributed import shard_range, shard_system
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.loop import setup_system
from tests.helpers import RY, si_upf

_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _si_system(kmesh=(2, 2, 1), use_symmetry=False, a=5.43):
    cell = a / 2 * _FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return setup_system(
        cell, pos, [0, 0], [si_upf()], ecut=8 * RY, kmesh=kmesh, use_symmetry=use_symmetry
    )


@pytest.mark.parametrize(
    ("n", "world_size"),
    [(4, 2), (5, 2), (7, 3), (1, 1), (8, 4)],
)
def test_shard_range_partitions_exactly(n, world_size):
    seen = []
    for rank in range(world_size):
        start, end = shard_range(n, rank, world_size)
        seen.extend(range(start, end))
        # as-even-as-possible: no rank's share differs from another's by >1
        assert 0 <= end - start <= -(-n // world_size)
    # every index covered exactly once, in order (contiguous, rank-ordered)
    assert seen == list(range(n))


def test_shard_system_rejects_symmetry_and_zero_share():
    # FCC Si's full cubic point group needs an equal-dimension mesh to reduce
    # cleanly (see setup_common.coupled_axes) -- (2,2,2), not the (2,2,1) the
    # other tests below use just to keep nk small.
    sym_system = _si_system(kmesh=(2, 2, 2), use_symmetry=True)
    with pytest.raises(NotImplementedError):
        shard_system(sym_system, rank=0, world_size=2, group=None)

    plain_system = _si_system(kmesh=(2, 2, 1))  # nk=4
    with pytest.raises(ValueError):
        # more ranks than k-points: some rank gets zero
        shard_system(plain_system, rank=5, world_size=8, group=None)


def test_shard_system_splits_k_data_contiguously():
    system = _si_system(kmesh=(2, 2, 1))  # nk=4
    nk = len(system.kweights)
    local0, ctx0 = shard_system(system, rank=0, world_size=2, group=None)
    local1, ctx1 = shard_system(system, rank=1, world_size=2, group=None)

    assert (ctx0.k_start, ctx0.k_end) == (0, nk // 2)
    assert (ctx1.k_start, ctx1.k_end) == (nk // 2, nk)
    assert ctx0.nk_global == ctx1.nk_global == nk
    assert ctx0.full_system is system and ctx1.full_system is system

    assert torch.equal(
        torch.cat([local0.kweights, local1.kweights]), system.kweights
    )
    assert len(local0.spheres) + len(local1.spheres) == nk
    # geometry-global fields are untouched (same object/tensor, not copies)
    assert local0.grid is system.grid
    assert torch.equal(local0.positions, system.positions)
    assert local0.n_electrons == system.n_electrons == local1.n_electrons


def test_sharded_density_sums_to_full_density():
    """The actual reduction claim: density_b on two disjoint k-shards, summed,
    equals density_b on the full, unsharded system -- exactly (same floats,
    same operation order per k, just partitioned)."""
    system = _si_system(kmesh=(2, 2, 1))  # nk=4
    local0, _ = shard_system(system, rank=0, world_size=2, group=None)
    local1, _ = shard_system(system, rank=1, world_size=2, group=None)

    torch.manual_seed(0)
    nb = system.nbands
    grid_shape = tuple(system.grid.shape)
    vol = system.grid.volume

    def _rand_coeffs(bk):
        c = torch.randn(bk.nk, nb, bk.npw_max, dtype=CDTYPE)
        c = c * bk.mask[:, None, :]
        return c

    def _rand_occ(nk):
        return torch.rand(nk, nb, dtype=RDTYPE)

    assert system.batch is not None
    assert local0.batch is not None and local1.batch is not None
    full_bk = system.batch

    # Independent random coeffs/occ PER SHARD -- each shard's own npw_max may
    # be SMALLER than the full system's (build_batched pads to the widest
    # sphere in whatever subset it's given, so a smaller batch can pad
    # narrower). That's fine: padded slots are zero either way (see
    # core/batch.py's padded-slot invariants).
    c0, o0 = _rand_coeffs(local0.batch), _rand_occ(local0.batch.nk)
    c1, o1 = _rand_coeffs(local1.batch), _rand_occ(local1.batch.nk)

    rho0 = density_b(c0, o0, local0.kweights, local0.batch, grid_shape, vol)
    rho1 = density_b(c1, o1, local1.kweights, local1.batch, grid_shape, vol)

    # Reconstruct the equivalent full-batch (world_size=1) coefficients: each
    # shard's zero-padded slots extend harmlessly to the full system's wider
    # npw_max, since the extra columns are exactly where the full batch's own
    # mask is False too.
    c_full = torch.zeros(full_bk.nk, nb, full_bk.npw_max, dtype=CDTYPE)
    c_full[: local0.batch.nk, :, : local0.batch.npw_max] = c0
    c_full[local0.batch.nk :, :, : local1.batch.npw_max] = c1
    occ_full = torch.cat([o0, o1], dim=0)

    rho_full = density_b(c_full, occ_full, system.kweights, full_bk, grid_shape, vol)
    torch.testing.assert_close(rho0 + rho1, rho_full, atol=1e-12, rtol=1e-10)
