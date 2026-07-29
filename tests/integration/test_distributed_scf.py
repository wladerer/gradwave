"""Distributed (k-point-sharded) SCF: 2-rank single-box correctness check.

Validates gradwave.distributed's density/occupation/energy reduction against
a plain single-process SCF on the SAME system: spawns 2 local processes over
torch.multiprocessing (Gloo, TCP rendezvous on localhost) that each set the
RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT environment variables torchrun would
normally set, then call gradwave.distributed.init_from_env() -- the exact,
real production entry point a `torchrun ... gradwave run input.yaml` launch
exercises, just driven by the test harness instead of torchrun itself.

This only covers the 2-rank-on-one-box case. A real 2-machine launch (the
scenario scripts/gradwave_distributed.sh and docs/manual/distributed.md
document, with GLOO_SOCKET_IFNAME=tailscale0) is NOT exercised here -- it
needs a second physical machine reachable from CI, which this suite does not
have. See docs/manual/distributed.md for that follow-up.
"""

from __future__ import annotations

import os
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from tests.helpers import RY, si_fcc, si_upf

pytestmark = pytest.mark.standard


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_system():
    """2-atom Si, a 2x2x1 Monkhorst-Pack mesh (nk=4, no symmetry reduction --
    distributed v1 requires use_symmetry=False), small enough to run 3x
    (reference + 2 shards) quickly."""
    from gradwave.scf.loop import setup_system

    cell, pos = si_fcc()
    return setup_system(
        cell, pos, [0, 0], [si_upf()], ecut=8 * RY, kmesh=(2, 2, 1), use_symmetry=False
    )


_SCF_KWARGS = dict(
    smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-9, max_iter=80, verbose=False
)


def _worker(rank: int, world_size: int, port: int, out: dict) -> None:
    os.environ["GRADWAVE_NUM_THREADS"] = "1"  # 2 worker processes share this box
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    from gradwave.core.xc.pbe import PBE
    from gradwave.distributed import init_from_env, shard_system
    from gradwave.scf.loop import scf

    system = _build_system()
    info = init_from_env()
    assert info is not None
    r, ws, group = info
    local_system, ctx = shard_system(system, r, ws, group)
    res = scf(local_system, PBE(), dist_ctx=ctx, **_SCF_KWARGS)

    if rank == 0:
        out["free_energy"] = float(res.energies.free_energy)
        out["kinetic"] = float(res.energies.kinetic)
        out["nonlocal"] = float(res.energies.nonlocal_)
        out["rho"] = res.rho.numpy()
        out["converged"] = bool(res.converged)
        out["eigenvalues"] = res.eigenvalues.numpy()
        out["occupations"] = res.occupations.numpy()
        out["n_coeffs_k"] = len(res.coeffs)
        out["nk_full"] = len(res.system.kweights)

    import torch.distributed as dist

    dist.destroy_process_group()


def test_distributed_matches_single_rank():
    port = _free_port()
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(2, port, out), nprocs=2, join=True)

    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.loop import scf

    ref = scf(_build_system(), PBE(), **_SCF_KWARGS)

    assert out["converged"]
    assert ref.converged
    # the reassembled SCFResult looks like an ordinary full-mesh run: every
    # k-point's orbitals/system present, not just this rank's shard
    assert out["n_coeffs_k"] == out["nk_full"] == 4

    assert out["free_energy"] == pytest.approx(float(ref.energies.free_energy), abs=1e-7)
    assert out["kinetic"] == pytest.approx(float(ref.energies.kinetic), abs=1e-7)
    assert out["nonlocal"] == pytest.approx(float(ref.energies.nonlocal_), abs=1e-7)
    np.testing.assert_allclose(out["rho"], ref.rho.numpy(), atol=1e-8)
    np.testing.assert_allclose(out["eigenvalues"], ref.eigenvalues.numpy(), atol=1e-6)
    np.testing.assert_allclose(out["occupations"], ref.occupations.numpy(), atol=1e-6)
