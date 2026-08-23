"""Distributed (k-point-sharded) USPP/PAW SCF: 2-rank single-box correctness
check, mirroring test_distributed_scf.py's pattern for the NC driver but for
scf_uspp -- including a DFT+U (Dudarev) variant, which the NC driver's
dist_ctx does not support but scf_uspp's does (see uspp_loop.py's
_hubbard_occ_update: the S-metric occupation matrices are all_reduce-summed
across ranks before the nonlinear Dudarev trace, mirroring the density/becsum
reduction).

Validates gradwave.distributed's shard_uspp_system + scf_uspp(dist_ctx=...)
against a plain single-process scf_uspp on the SAME system, using the exact
Si PAW fixture and cell test_uspp_batched_equality.py already establishes as
a small, fast-converging reference (2x2x1 mesh, nk=4, plus a use_symmetry
variant on the ideal cell exercising the distributed x IBZ composition).

Only the 2-rank-on-one-box case is covered here, same caveat as
test_distributed_scf.py: a real 2-machine launch is not exercised (see
docs/manual/distributed.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch.multiprocessing as mp

from tests.helpers import RY, dist_file_init_method, set_dist_worker_env

# nightly: distributed-vs-single-rank is a structural-equivalence guard on the
# k-sharding path, which changes rarely; a refactor that touches it runs the full
# suite anyway, so it needn't cost every CI shard.
pytestmark = pytest.mark.slow

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS_DISP = np.array([[0.0, 0.0, 0.0], [1.4075, 1.3175, 1.3775]])
SI_POS_IDEAL = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])


def _build_system(use_symmetry: bool = False):
    """Displaced Si on a 2x2x1 mesh (nk=4) for the unsymmetrized tests; the
    IDEAL diamond cell on a 2x2x2 mesh (8 points -> 3 IBZ k, uneven 2/1 shard)
    when use_symmetry -- the displaced cell would fold to near-P1 and exercise
    nothing."""
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import setup_uspp

    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    pos = SI_POS_IDEAL if use_symmetry else SI_POS_DISP
    kmesh = (2, 2, 2) if use_symmetry else (2, 2, 1)
    return setup_uspp(
        SI_CELL, pos, [0, 0], [paw], ecut=15 * RY, kmesh=kmesh, ecutrho=60 * RY,
        use_symmetry=use_symmetry,
    )


_SCF_KWARGS = dict(etol=1e-10, rhotol=1e-9, max_iter=80, verbose=False)


def _worker(rank: int, world_size: int, init_method: str, hubbard: bool, out: dict,
            use_symmetry: bool = False) -> None:
    set_dist_worker_env(rank, world_size, init_method)

    from gradwave.core.xc.pbe import PBE
    from gradwave.distributed import init_from_env, shard_uspp_system
    from gradwave.scf.uspp import scf_uspp
    from gradwave.scf.uspp_hubbard import HubbardManifold

    man = [HubbardManifold(species=0, l=1, u=3.0, j=0.0)] if hubbard else None
    system = _build_system(use_symmetry)
    info = init_from_env()
    assert info is not None
    r, ws, group = info
    local_system, ctx = shard_uspp_system(system, r, ws, group)
    res = scf_uspp(local_system, PBE(), dist_ctx=ctx, hubbard=man, **_SCF_KWARGS)

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


def _run(hubbard: bool, use_symmetry: bool = False):
    init_method = dist_file_init_method()
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(2, init_method, hubbard, out, use_symmetry), nprocs=2, join=True)

    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.uspp import scf_uspp
    from gradwave.scf.uspp_hubbard import HubbardManifold

    man = [HubbardManifold(species=0, l=1, u=3.0, j=0.0)] if hubbard else None
    ref = scf_uspp(_build_system(use_symmetry), PBE(), hubbard=man, **_SCF_KWARGS)

    assert out["converged"]
    assert ref.converged
    # the reassembled USPPResult looks like an ordinary full-mesh run: every
    # k-point's orbitals/⟨β|ψ⟩ present, not just this rank's shard (3 IBZ
    # k-points when symmetrized, the plain 4-point mesh otherwise)
    nk_expect = 3 if use_symmetry else 4
    assert out["n_coeffs_k"] == out["n_becps_k"] == out["nk_full"] == nk_expect
    assert len(ref.system.kweights) == nk_expect

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


def test_distributed_uspp_matches_single_rank():
    _run(hubbard=False)


def test_distributed_uspp_hubbard_matches_single_rank():
    _run(hubbard=True)


def test_distributed_uspp_symmetric_hubbard_matches_single_rank():
    # distributed x IBZ on the USPP/PAW driver, +U on: one test exercising all
    # three symmetry objects under sharding -- rho_symmetrizer and becsum_sym
    # applied to the post-all_reduce global density/becsum, and the n_hub
    # all_reduce on the IBZ-weighted occupation matrices.
    _run(hubbard=True, use_symmetry=True)
