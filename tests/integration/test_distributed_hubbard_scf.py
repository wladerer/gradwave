"""Distributed (k-point-sharded) SCF with DFT+U: 2-rank correctness check.

Companion to test_distributed_scf.py, but exercising the Hubbard occupation
matrix's 4th collective point (gradwave.distributed / scf.loop._hubbard_occ_update):
n_hub is a k-extensive sum built the same way as the density (all_reduce-summed
across ranks), but e_hub is a NONLINEAR (Tr[n(1-n)]) function of n_hub, so it
must be recomputed from the reduced, full-mesh n_hub rather than summed per rank
like kinetic/nonlocal energy. Validates the distributed result's converged free
energy, e_hub, eigenvalues, and Hubbard occupation matrix against a
single-process reference on the identical system.

Reuses the small diamond-carbon-with-Hubbard-U-on-2p system pattern from
tests/integration/test_hubbard_vs_qe.py's `_diamond_c_system` (a real
correlated-atom NC system already exercised by the linear-response-U tests),
rather than inventing a new fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch.multiprocessing as mp

from tests.helpers import RY, dist_file_init_method, pseudo, set_dist_worker_env

pytestmark = pytest.mark.standard


def _diamond_c_system():
    """Two-atom diamond-carbon cell, a 2x2x2 Monkhorst-Pack mesh (nk=8, no
    symmetry reduction -- distributed v1 requires use_symmetry=False), small
    enough to run 3x (reference + 2 shards) quickly. Same cell as
    test_hubbard_vs_qe.py's `_diamond_c_system`, evaluated at a slightly
    coarser k-mesh here purely for test speed."""
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    a = 3.567
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    c = parse_upf(pseudo("PD_C_PBE_std.upf"))
    return setup_system(
        cell, pos, [0, 0], [c], ecut=25 * RY, kmesh=(2, 2, 2), nbands=8, use_symmetry=False
    )


def _hubbard():
    from gradwave.core.hubbard import HubbardManifold

    # A real (nonzero) U on the C 2p manifold -- not physically tuned, but a
    # genuine correlated-orbital correction exercising the same n_hub/e_hub
    # code path a physical +U run would.
    return [HubbardManifold(species=0, l=1, u=5.0, j=0.0)]


_SCF_KWARGS = dict(
    smearing="gaussian", width=0.05, etol=1e-9, rhotol=1e-9, max_iter=100, verbose=False, nspin=1
)


def _worker(rank: int, world_size: int, init_method: str, out: dict) -> None:
    set_dist_worker_env(rank, world_size, init_method)

    from gradwave.core.xc.pbe import PBE
    from gradwave.distributed import init_from_env, shard_system
    from gradwave.scf.loop import scf

    system = _diamond_c_system()
    info = init_from_env()
    assert info is not None
    r, ws, group = info
    local_system, ctx = shard_system(system, r, ws, group)
    res = scf(local_system, PBE(), dist_ctx=ctx, hubbard=_hubbard(), **_SCF_KWARGS)

    if rank == 0:
        out["free_energy"] = float(res.energies.free_energy)
        out["kinetic"] = float(res.energies.kinetic)
        out["nonlocal"] = float(res.energies.nonlocal_)
        out["hubbard"] = float(res.energies.hubbard)
        out["hub_occ_trace"] = [float(m.trace().real) for m in res.hub_occ[0]]
        out["rho"] = res.rho.numpy()
        out["converged"] = bool(res.converged)
        out["eigenvalues"] = res.eigenvalues.numpy()
        out["occupations"] = res.occupations.numpy()
        out["n_coeffs_k"] = len(res.coeffs)
        out["nk_full"] = len(res.system.kweights)

    import torch.distributed as dist

    dist.destroy_process_group()


def test_distributed_hubbard_matches_single_rank():
    init_method = dist_file_init_method()
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(2, init_method, out), nprocs=2, join=True)

    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.loop import scf

    ref = scf(_diamond_c_system(), PBE(), hubbard=_hubbard(), **_SCF_KWARGS)

    assert out["converged"]
    assert ref.converged
    assert out["n_coeffs_k"] == out["nk_full"] == 8

    assert out["free_energy"] == pytest.approx(float(ref.energies.free_energy), abs=1e-7)
    assert out["kinetic"] == pytest.approx(float(ref.energies.kinetic), abs=1e-7)
    assert out["nonlocal"] == pytest.approx(float(ref.energies.nonlocal_), abs=1e-7)
    assert out["hubbard"] == pytest.approx(float(ref.energies.hubbard), abs=1e-7)

    ref_hub_occ_trace = [float(m.trace().real) for m in ref.hub_occ[0]]
    np.testing.assert_allclose(out["hub_occ_trace"], ref_hub_occ_trace, atol=1e-6)

    np.testing.assert_allclose(out["rho"], ref.rho.numpy(), atol=1e-8)
    np.testing.assert_allclose(out["eigenvalues"], ref.eigenvalues.numpy(), atol=1e-6)
    np.testing.assert_allclose(out["occupations"], ref.occupations.numpy(), atol=1e-6)
