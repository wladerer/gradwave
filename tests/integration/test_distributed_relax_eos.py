"""Distributed (k-point-sharded) relax and EOS reached from an input file.

Drives the production entry points a `torchrun ... gradwave run input.yaml`
launch exercises: an ``Input`` with ``distributed: true`` handed to
``api.run_relax`` (the nested SCF+BFGS engine, whose per-step SCF runs on the
sharded calculator) and to ``api.run_eos`` (the warm-started volume chain,
whose per-volume ``run_scf`` shards). Neither adds a new k-sum reduction: both
drivers converge each SCF sharded and reassemble a full-mesh result, so forces,
stress, the optimizer step, and the E(V) fit come out identical on every rank.

The reference is the SAME input run in an ordinary single process (no
WORLD_SIZE in the environment, so ``distributed: true`` is a no-op and
``init_from_env`` returns None). Small Si NC cell, 2x2x1 mesh (nk=4,
``symmetry: false`` as distributed v1 requires), and the 2-rank-on-one-box
caveat as tests/integration/test_distributed_scf.py.
"""

from __future__ import annotations

import os
import socket
import tempfile

import numpy as np
import pytest
import torch.multiprocessing as mp

from tests.helpers import RY, pseudo

pytestmark = pytest.mark.standard

PSEUDO = pseudo("Si_ONCV_PBE-1.2.upf")
# diamond-Si primitive cell; a = 5.43 Å
_A = 5.43
SI_CELL = _A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
# ideal second atom sits at a/4 along the body diagonal (1.3575); displace it
# so the relax has real work to do (and the warm-start chain is exercised)
SI_POS_DISP = np.array([[0.0, 0.0, 0.0], [1.30, 1.42, 1.35]])
SI_POS_IDEAL = np.array([[0.0, 0.0, 0.0], [_A / 4, _A / 4, _A / 4]])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _relax_input():
    """An ``Input`` for the displaced-Si NC cell, distributed, task: relax."""
    from gradwave.inputs import load_input

    body = f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS_DISP.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {os.path.dirname(PSEUDO)}
  map: {{Si: {os.path.basename(PSEUDO)}}}
ecut: {12 * RY}
xc: pbe
symmetry: false
distributed: true
task: relax
kpoints:
  mesh: [2, 2, 1]
relax:
  optimizer: bfgs
  fmax: 0.02
  max_steps: 6
scf:
  etol: 1.0e-9
  rhotol: 1.0e-8
  max_iter: 80
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body)
        path = f.name
    return load_input(path)


def _eos_input():
    """An ``Input`` for a near-equilibrium Si NC cell, distributed, task: eos."""
    from gradwave.inputs import load_input

    body = f"""
structure:
  cell: {SI_CELL.tolist()}
  positions:
    cart: {SI_POS_IDEAL.tolist()}
  species: [Si, Si]
pseudopotentials:
  dir: {os.path.dirname(PSEUDO)}
  map: {{Si: {os.path.basename(PSEUDO)}}}
ecut: {12 * RY}
xc: pbe
symmetry: false
distributed: true
task: eos
kpoints:
  mesh: [2, 2, 1]
eos:
  scales: [0.96, 0.98, 1.00, 1.02]
scf:
  etol: 1.0e-9
  rhotol: 1.0e-8
  max_iter: 80
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body)
        path = f.name
    return load_input(path)


def _relax_worker(rank: int, world_size: int, port: int, out: dict) -> None:
    os.environ["GRADWAVE_NUM_THREADS"] = "1"  # 2 worker processes share this box
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    from gradwave.api import run_relax

    relax, atoms, _frames = run_relax(_relax_input(), verbose=False)
    if rank == 0:
        out["energy_eV"] = relax["energy_eV"]
        out["positions_ang"] = np.asarray(relax["positions_ang"])
        out["fmax_eV_ang"] = relax["fmax_eV_ang"]
        out["scf_iter_per_step"] = list(relax.get("scf_iter_per_step", []))
        out["n_steps"] = relax["n_steps"]

    from gradwave.distributed import maybe_destroy_process_group

    maybe_destroy_process_group()


def _eos_worker(rank: int, world_size: int, port: int, out: dict) -> None:
    os.environ["GRADWAVE_NUM_THREADS"] = "1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    from gradwave.api import run_eos

    block = run_eos(_eos_input(), verbose=False)
    if rank == 0:
        out["v0"] = block["v0_ang3_per_atom"]
        out["b0_GPa"] = block["b0_GPa"]
        out["b0_prime"] = block["b0_prime"]
        out["energies"] = list(block["energies_eV_per_atom"])

    from gradwave.distributed import maybe_destroy_process_group

    maybe_destroy_process_group()


def test_distributed_relax_matches_single_rank():
    port = _free_port()
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_relax_worker, args=(2, port, out), nprocs=2, join=True)

    from gradwave.api import run_relax

    # same Input, single process: WORLD_SIZE unset, so distributed: true is a
    # no-op (init_from_env returns None) and this is the full-mesh reference.
    ref_relax, _ref_atoms, _ref_frames = run_relax(_relax_input(), verbose=False)

    # identical final geometry and energy to tight tolerance (the sharded
    # density/energy reduction reproduces the full-mesh value to ~1e-8, so the
    # BFGS trajectory lands on the same geometry)
    assert out["energy_eV"] == pytest.approx(ref_relax["energy_eV"], abs=1e-7)
    np.testing.assert_allclose(
        out["positions_ang"], np.asarray(ref_relax["positions_ang"]), atol=1e-6
    )
    assert out["fmax_eV_ang"] == pytest.approx(ref_relax["fmax_eV_ang"], abs=1e-6)
    # identical per-step SCF iteration counts: the warm-start orbital seed is
    # sharded to each rank's k-slice, so the Davidson converges exactly as the
    # single-process run does every ionic step (shard_start_from)
    assert out["n_steps"] == ref_relax["n_steps"]
    assert out["scf_iter_per_step"] == list(ref_relax.get("scf_iter_per_step", []))
    assert out["scf_iter_per_step"]  # the run actually recorded SCF iterations


def test_distributed_eos_matches_single_rank():
    port = _free_port()
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_eos_worker, args=(2, port, out), nprocs=2, join=True)

    from gradwave.api import run_eos

    ref = run_eos(_eos_input(), verbose=False)

    # identical BM3 fit parameters and the underlying E(V) curve
    np.testing.assert_allclose(
        out["energies"], np.asarray(ref["energies_eV_per_atom"]), atol=1e-8
    )
    assert out["v0"] == pytest.approx(ref["v0_ang3_per_atom"], abs=1e-6)
    assert out["b0_GPa"] == pytest.approx(ref["b0_GPa"], abs=1e-4)
    assert out["b0_prime"] == pytest.approx(ref["b0_prime"], abs=1e-5)
