"""Memoized symmetry machinery in the setup layer.

During a relaxation the system setup re-runs every ionic step; the spglib
search is cheap and must re-run (a move can break the group), but the
RhoSymmetrizer index/phase maps, the becsum D blocks, and the USPP aug
form-factor tables are expensive and depend only on (grid shape, op set),
(Cartesian rotations, species layout), and (pseudo, density-sphere G set)
respectively. setup_common / uspp_setup memoize them on those content keys.

Oracles here are identity + reuse: a rebuild with unchanged inputs shares the
cached tensors (reuse observable by object identity), a symmetry-preserving
strain still shares the maps while picking up the CURRENT grid's density
mask, and cached instances act identically to freshly constructed ones.
"""

import numpy as np
import pytest
import torch

from gradwave.scf.loop import setup_system
from gradwave.scf.paw_symmetry import BecsumSymmetrizer
from gradwave.scf.uspp import setup_uspp
from gradwave.symmetry import RhoSymmetrizer
from tests.helpers import RY, si_upf

_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.fixture(autouse=True)
def _hermetic_setup_caches():
    """Start each test from empty setup-layer memo caches.

    The identity/reuse oracles below assert object identity across cache hits.
    The RhoSymmetrizer memo is deliberately tiny (``_RHO_SYM_CACHE_MAX = 2``), so
    under pytest-xdist a preceding test in the same worker can leave it full and
    the in-test cache miss (the rattled, symmetry-broken geometry) can then evict
    the very entry a later ``is`` assertion expects — a sharding-order flake that
    never shows in isolation. Clearing the three caches first makes the test
    depend only on its own call sequence, which fits the limits deterministically.
    """
    from gradwave.scf import setup_common, uspp_setup

    for cache in (setup_common._RHO_SYM_CACHE, uspp_setup._AUG_CACHE,
                  uspp_setup._BECSUM_SYM_CACHE):
        cache.clear()
    yield


def _si_nc(scale=1.0, a=5.43):
    cell = a / 2 * _FCC * scale
    pos = np.array([[0.0, 0, 0], [a / 4] * 3]) * scale
    return setup_system(cell, pos, [0, 0], [si_upf()], ecut=8 * RY,
                        kmesh=(2, 2, 2), use_symmetry=True)


def _si_paw(paw, scale=1.0, a=5.43, rattle=0.0):
    cell = a / 2 * _FCC * scale
    pos = np.array([[0.0, 0, 0], [a / 4] * 3]) * scale
    pos[1, 0] += rattle
    return setup_uspp(cell, pos, [0, 0], [paw], ecut=8 * RY,
                      kmesh=(2, 2, 2), use_symmetry=True)


def test_nc_rho_symmetrizer_maps_reused_and_identical():
    s1 = _si_nc()
    s2 = _si_nc()
    # unchanged geometry: the maps are shared, not rebuilt
    assert s2.rho_symmetrizer.idx is s1.rho_symmetrizer.idx
    assert s2.rho_symmetrizer.phase is s1.rho_symmetrizer.phase

    # symmetry-preserving strain: maps still shared (same shape, same ops),
    # mask re-dressed from the CURRENT grid's density sphere
    s3 = _si_nc(scale=1.002)
    assert tuple(s3.grid.shape) == tuple(s1.grid.shape)
    assert s3.rho_symmetrizer.idx is s1.rho_symmetrizer.idx
    assert torch.equal(s3.rho_symmetrizer.mask,
                       s3.grid.dens_mask.reshape(-1))

    # cached instance ≡ fresh construction on the strained geometry
    fresh = RhoSymmetrizer(s3.grid.shape, s3.sym, dens_mask=s3.grid.dens_mask)
    rho = torch.randn(tuple(s3.grid.shape), dtype=torch.complex128,
                      generator=torch.Generator().manual_seed(0))
    diff = (fresh.apply(rho) - s3.rho_symmetrizer.apply(rho)).abs().max()
    assert float(diff) == 0.0


def test_uspp_symmetry_machinery_reused_and_identical():
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from tests.helpers import pseudo

    paw = parse_upf_paw(pseudo("Si.pbe-n-kjpaw_psl.1.0.0.UPF"))

    s1 = _si_paw(paw)
    s2 = _si_paw(paw)
    assert s2.rho_symmetrizer.idx is s1.rho_symmetrizer.idx
    assert s2.becsum_sym.d_full is s1.becsum_sym.d_full
    assert s2.aug[0] is s1.aug[0]

    # a move that changes the group misses the symmetrizer caches and gets
    # its own maps; the aug tables are position-independent and still reused
    # at the same cell (the per-pseudo memo holds the latest G set)
    s4 = _si_paw(paw, rattle=0.05)
    assert s4.sym.n_ops != s1.sym.n_ops
    assert s4.rho_symmetrizer.idx is not s1.rho_symmetrizer.idx
    assert s4.aug[0] is s1.aug[0]

    # symmetry-preserving strain: symmetrizer maps and D blocks are reused
    # (Cartesian rotations unchanged), the aug tables follow the new G set
    s3 = _si_paw(paw, scale=1.002)
    assert tuple(s3.grid.shape) == tuple(s1.grid.shape)
    assert s3.rho_symmetrizer.idx is s1.rho_symmetrizer.idx
    assert s3.becsum_sym.d_full is s1.becsum_sym.d_full
    assert s3.becsum_sym.sg is s3.sym  # carries the CURRENT group
    assert s3.aug[0] is not s1.aug[0]
    assert torch.equal(s3.rho_symmetrizer.mask,
                       s3.grid.dens_mask.reshape(-1))

    # cached becsum symmetrizer ≡ fresh construction on the strained cell
    fresh = BecsumSymmetrizer(s3.sym, 5.43 / 2 * _FCC * 1.002, [paw], [0, 0],
                              s3.atom_slices)
    nm = s3.atom_slices[0][1]
    gen = torch.Generator().manual_seed(0)
    mats = [torch.randn(nm, nm, dtype=torch.float64, generator=gen)
            .to(torch.complex128) for _ in range(2)]
    for got, want in zip(s3.becsum_sym.apply(mats), fresh.apply(mats),
                         strict=True):
        assert float((got - want).abs().max()) < 1e-15
