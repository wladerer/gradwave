"""Large-payload collective-gather correctness for gradwave.distributed.

Targets the #216 fix directly, without running a full SCF: two real Gloo ranks
spawned over torch.multiprocessing exchange ragged tensor lists sized in the
tens of MB -- the regime where the old ``all_gather_object`` path deadlocked in
``_object_to_tensor`` (pickling large tensors over Gloo). The rewritten
:func:`gradwave.distributed.gather_list_cat` / :func:`gather_cat` stage the raw
bytes through CPU and must (a) complete and (b) reassemble the exact,
rank-ordered global structure with dtypes and values preserved.

CPU-only by design: the fix routes every payload through CPU regardless of the
source device, so a GPU is not needed to exercise the new code path.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.standard


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Global, deterministic per-k coefficient block: shape and contents depend only
# on the GLOBAL k index, so every rank (and the parent's reference) agrees on
# what k should hold after the gather. Sized so nk of them total ~40 MB, big
# enough to have tripped the old pickle path in spirit while staying quick.
_NK = 8
_NBANDS = 32


def _expected_block(gk: int) -> torch.Tensor:
    npw = 8000 + 500 * gk  # ragged across k: (32, 8000..11500) complex128
    n = _NBANDS * npw
    real = torch.arange(n, dtype=torch.float64).reshape(_NBANDS, npw) + gk
    imag = -torch.arange(n, dtype=torch.float64).reshape(_NBANDS, npw)
    return torch.complex(real, imag)


def _shard_range(n: int, rank: int, world: int) -> tuple[int, int]:
    base, rem = divmod(n, world)
    start = rank * base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


def _make_ctx(rank: int, world: int, group, k_start: int, k_end: int):
    from gradwave.distributed import DistKContext

    # gather_cat / gather_list_cat only read world_size and group; full_system
    # is irrelevant to the collective, so a placeholder is fine here.
    return DistKContext(
        rank=rank,
        world_size=world,
        group=group,
        k_start=k_start,
        k_end=k_end,
        nk_global=_NK,
        full_system=None,  # type: ignore[arg-type]
    )


def _worker(rank: int, world: int, port: int, out: dict) -> None:
    import os

    os.environ["GRADWAVE_NUM_THREADS"] = "1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    import torch.distributed as dist

    from gradwave.distributed import gather_cat, gather_list_cat

    info = None
    try:
        from gradwave.distributed import init_from_env

        info = init_from_env()
        assert info is not None
        r, ws, group = info
        k_start, k_end = _shard_range(_NK, r, ws)
        ctx = _make_ctx(r, ws, group, k_start, k_end)

        # (1) ragged large tensor list -> rank-ordered global list.
        local_coeffs = [_expected_block(gk) for gk in range(k_start, k_end)]
        gathered = gather_list_cat(local_coeffs, ctx)

        # (2) variable-length-along-dim single tensor (eigenvalue-like shard).
        local_eigs = torch.arange(k_start, k_end, dtype=torch.float64).reshape(-1, 1) * (
            torch.arange(_NBANDS, dtype=torch.float64) + 1.0
        )
        eigs_global = gather_cat(local_eigs, ctx, dim=0)

        # (3) object path: non-tensor scalar-tuple picks, with rank 1 empty to
        #     exercise the empty-list branch of the kind agreement.
        local_picks = [] if r == 1 else [(r, gk, gk * 10) for gk in range(k_start, k_end)]
        picks_global = gather_list_cat(local_picks, ctx)

        if r == 0:
            ok_len = len(gathered) == _NK
            ok_vals = all(
                torch.equal(gathered[gk], _expected_block(gk)) for gk in range(_NK)
            )
            ok_dtype = all(t.dtype == torch.complex128 for t in gathered)
            ok_dev = all(t.device.type == "cpu" for t in gathered)

            ref_eigs = torch.cat(
                [
                    torch.tensor([gk], dtype=torch.float64).reshape(-1, 1)
                    * (torch.arange(_NBANDS, dtype=torch.float64) + 1.0)
                    for gk in range(_NK)
                ],
                dim=0,
            )
            ok_eigs = torch.equal(eigs_global, ref_eigs)

            # rank 0's picks only (rank 1 contributed an empty list)
            ref_picks = [(0, gk, gk * 10) for gk in range(*_shard_range(_NK, 0, ws))]
            ok_picks = picks_global == ref_picks

            out["ok_len"] = ok_len
            out["ok_vals"] = ok_vals
            out["ok_dtype"] = ok_dtype
            out["ok_dev"] = ok_dev
            out["ok_eigs"] = ok_eigs
            out["ok_picks"] = ok_picks
            out["nbytes"] = sum(t.numel() * t.element_size() for t in gathered)
    finally:
        if info is not None and dist.is_initialized():
            dist.destroy_process_group()


def test_large_payload_gather_two_ranks():
    port = _free_port()
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(2, port, out), nprocs=2, join=True)

    assert out.get("ok_len"), "gathered global list has wrong length"
    assert out.get("ok_vals"), "gathered values differ from the expected blocks"
    assert out.get("ok_dtype"), "complex128 dtype not preserved through the gather"
    assert out.get("ok_dev"), "gathered tensors not on the expected (cpu) device"
    assert out.get("ok_eigs"), "gather_cat variable-length concatenation is wrong"
    assert out.get("ok_picks"), "object-path (non-tensor) gather is wrong"
    # sanity: the payload really was in the tens-of-MB regime the fix targets
    assert out.get("nbytes", 0) > 30_000_000
