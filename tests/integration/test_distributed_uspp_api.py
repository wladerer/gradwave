"""Distributed (k-point-sharded) USPP/PAW SCF reached from an input file.

Mirrors test_distributed_uspp_scf.py, but drives the production entry point a
`torchrun ... gradwave run input.yaml` launch exercises: an ``Input`` with
``distributed: true`` handed to ``api.run_scf``, which shards the built
``USPPSystem`` through ``distributed.shard_uspp_system`` and threads the
resulting ``dist_ctx`` into ``scf_uspp`` -- instead of the library test's hand
call to those two functions. A DFT+U (Dudarev) variant confirms the Hubbard
manifold the api builds reaches the sharded driver unchanged.

The reference is the SAME input run in an ordinary single process (no
WORLD_SIZE in the environment, so ``distributed: true`` is a no-op and
``init_from_env`` returns None). Same Si PAW fixture, 2x2x1 mesh (nk=4,
``symmetry: false`` as distributed v1 requires), and 2-rank-on-one-box caveat
as test_distributed_uspp_scf.py.
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch.multiprocessing as mp

from tests.helpers import RY

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
PSEUDO = FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF"
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS_DISP = np.array([[0.0, 0.0, 0.0], [1.4075, 1.3175, 1.3775]])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_input(hubbard: bool):
    """An ``Input`` for the displaced-Si PAW cell with distributed enabled.
    Written to a temp YAML so the whole load_input path is exercised, the same
    way a real launch reads its input file."""
    from gradwave.inputs import load_input

    hub_block = "hubbard:\n  - {species: Si, l: 1, u: 3.0, j: 0.0}\n" if hubbard else ""
    body = f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS_DISP.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDO.parent}
  map: {{Si: {PSEUDO.name}}}
ecut: {15 * RY}
ecutrho: {60 * RY}
xc: pbe
symmetry: false
distributed: true
kpoints:
  mesh: [2, 2, 1]
scf:
  etol: 1.0e-10
  rhotol: 1.0e-9
  max_iter: 80
{hub_block}"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body)
        path = f.name
    return load_input(path)


def _worker(rank: int, world_size: int, port: int, hubbard: bool, out: dict) -> None:
    os.environ["GRADWAVE_NUM_THREADS"] = "1"  # 2 worker processes share this box
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    from gradwave.api import run_scf

    res = run_scf(_make_input(hubbard), verbose=False)

    if rank == 0:
        out["free_energy"] = float(res.energies.free_energy)
        out["kinetic"] = float(res.energies.kinetic)
        out["nonlocal"] = float(res.energies.nonlocal_)
        out["rho"] = res.rho.numpy()
        out["converged"] = bool(res.converged)
        out["eigenvalues"] = res.eigenvalues.numpy()
        out["occupations"] = res.occupations.numpy()
        out["n_coeffs_k"] = len(res.coeffs)
        out["n_becps_k"] = len(res.becps)
        out["nk_full"] = len(res.system.kweights)
        if hubbard:
            out["hub_occ"] = [m.numpy() for m in res.hub_occ[0]]

    import torch.distributed as dist

    dist.destroy_process_group()


def _run(hubbard: bool):
    port = _free_port()
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(2, port, hubbard, out), nprocs=2, join=True)

    from gradwave.api import run_scf

    # same Input, single process: WORLD_SIZE is unset here, so distributed:
    # true is a no-op (init_from_env returns None) and this is the ordinary
    # full-mesh reference.
    ref = run_scf(_make_input(hubbard), verbose=False)

    assert out["converged"]
    assert ref.converged
    # the reassembled USPPResult looks like an ordinary full-mesh run: every
    # k-point's orbitals/⟨β|ψ⟩ present, not just this rank's shard
    assert out["n_coeffs_k"] == out["n_becps_k"] == out["nk_full"] == 4

    assert out["free_energy"] == pytest.approx(float(ref.energies.free_energy), abs=1e-7)
    assert out["kinetic"] == pytest.approx(float(ref.energies.kinetic), abs=1e-7)
    assert out["nonlocal"] == pytest.approx(float(ref.energies.nonlocal_), abs=1e-7)
    np.testing.assert_allclose(out["rho"], ref.rho.numpy(), atol=1e-8)
    np.testing.assert_allclose(out["eigenvalues"], ref.eigenvalues.numpy(), atol=1e-6)
    np.testing.assert_allclose(out["occupations"], ref.occupations.numpy(), atol=1e-6)
    if hubbard:
        ref_hub = ref.hub_occ[0]
        for m_dist, m_ref in zip(out["hub_occ"], ref_hub, strict=True):
            np.testing.assert_allclose(m_dist, m_ref.numpy(), atol=1e-7)


def test_distributed_uspp_api_matches_single_rank():
    _run(hubbard=False)


def test_distributed_uspp_api_hubbard_matches_single_rank():
    _run(hubbard=True)
