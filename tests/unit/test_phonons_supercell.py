"""Unit tests for the supercell-phonon numerics (postscf.phonons_supercell).

No SCF: these check the (μ,R) supercell map, the D(q) Fourier folding against
direct Γ diagonalization, Hermiticity, and the sum-rule / symmetrization
bookkeeping. The physics (Si dispersion, acoustic modes) is exercised in
tests/integration/test_phonons_supercell_task.py.
"""

import numpy as np
import pytest

from gradwave.postscf.phonons import gamma_frequencies
from gradwave.postscf.phonons_supercell import (
    SupercellDisplacementSymmetry,
    _site_lookup,
    apply_acoustic_sum_rule,
    build_supercell,
    dynamical_matrix,
    frequencies_at_q,
    symmetrize_force_constants,
)

CELL = np.eye(3) * 3.0
POS = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ CELL
MASSES = np.array([28.0, 28.0])

# fcc / diamond primitive cells for the displacement-symmetry tests
_FCC_A = 4.05
FCC_CELL = _FCC_A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
FCC_POS = np.array([[0.0, 0.0, 0.0]])
DIAMOND_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
DIAMOND_POS = np.array([[0.0, 0.0, 0.0], [5.43 / 4] * 3])


def _covariance_residual(sym, phi):
    """max block residual of Φ[g(a),·,recentered P(s),·] − S·Φ[a,·,s,·]·Sᵀ over
    every op and home atom — zero iff Φ is exactly point-group symmetric."""
    n = sym._nsuper
    worst = 0.0
    for op in range(len(sym.s_cart)):
        s_cart = sym.s_cart[op]
        amap = sym.atom_map[op]
        mu_img = sym.mu_image[op]
        r_img = sym.r_image[op]
        for a in range(sym.na):
            r_out = (r_img - r_img[a]) % n
            s_out = sym._site_of[mu_img, r_out[:, 0], r_out[:, 1], r_out[:, 2]]
            lhs = phi[amap[a]][:, s_out, :]
            rhs = np.einsum("ik,ksj->isj", s_cart,
                            np.einsum("ksj,jl->ksl", phi[a], s_cart.T))
            worst = max(worst, float(np.abs(lhs - rhs).max()))
    return worst


def test_fcc_displacement_symmetry_is_one_column():
    sc = build_supercell(FCC_CELL, FCC_POS, [0], (2, 2, 2))
    sym = SupercellDisplacementSymmetry(sc)
    assert sym.sg.international == "Fm-3m"
    assert len(sym.s_cart) == 48  # full Oh survives the 2×2×2 box
    assert sym.can_reduce
    assert len(sym.displacements) == 1  # 6→2 SCFs for cubic monatomic
    assert sym.displacements[0][0] == 0


def test_diamond_displacement_symmetry_is_one_column():
    sc = build_supercell(DIAMOND_CELL, DIAMOND_POS, [0, 0], (2, 2, 2))
    sym = SupercellDisplacementSymmetry(sc)
    assert sym.sg.international == "Fd-3m"
    assert sym.can_reduce
    # both basis atoms and all axes reconstruct from a single column
    assert len(sym.displacements) == 1


def test_anisotropic_supercell_drops_axis_mixing_ops():
    # a 2×1×1 box of simple cubic keeps only ops that respect unequal periods
    sc = build_supercell(CELL, np.array([[0.0, 0, 0]]), [0], (2, 1, 1))
    sym = SupercellDisplacementSymmetry(sc)
    assert len(sym.s_cart) < 48  # cubic ops mixing axis 0 with 1/2 are dropped
    assert sym.can_reduce


@pytest.mark.parametrize(
    ("cell", "pos", "spec", "n"),
    [(FCC_CELL, FCC_POS, [0], (2, 2, 2)),
     (DIAMOND_CELL, DIAMOND_POS, [0, 0], (2, 2, 2)),
     (CELL, np.array([[0.0, 0, 0]]), [0], (2, 1, 1))])
def test_reconstruct_projects_onto_group_symmetric(cell, pos, spec, n):
    # feeding arbitrary columns through reconstruct must yield a Φ that satisfies
    # the two-index point-group covariance exactly — the group-algebra oracle
    sc = build_supercell(cell, pos, spec, n)
    sym = SupercellDisplacementSymmetry(sc)
    rng = np.random.default_rng(0)
    phi0 = rng.standard_normal((sc.n_prim, 3, sc.n_sc, 3))
    phi = sym.reconstruct([phi0[a, i] for a, i in sym.displacements])
    assert _covariance_residual(sym, phi) < 1e-10


def test_reconstruct_is_idempotent():
    sc = build_supercell(FCC_CELL, FCC_POS, [0], (2, 2, 2))
    sym = SupercellDisplacementSymmetry(sc)
    rng = np.random.default_rng(1)
    phi0 = rng.standard_normal((sc.n_prim, 3, sc.n_sc, 3))
    phi = sym.reconstruct([phi0[a, i] for a, i in sym.displacements])
    phi2 = sym.reconstruct([phi[a, i] for a, i in sym.displacements])
    assert np.allclose(phi, phi2, atol=1e-12)


def test_reconstruct_reproduces_group_symmetric_phi_bitclose():
    # if Φ is already group-symmetric, the irreducible columns reconstruct it
    # bit-for-bit — proves the force-completion transform is exact
    sc = build_supercell(DIAMOND_CELL, DIAMOND_POS, [0, 0], (2, 2, 2))
    sym = SupercellDisplacementSymmetry(sc)
    rng = np.random.default_rng(2)
    phi_sym = sym.reconstruct(
        [rng.standard_normal((sc.n_sc, 3)) for _ in sym.displacements])
    phi_back = sym.reconstruct([phi_sym[a, i] for a, i in sym.displacements])
    assert np.abs(phi_sym - phi_back).max() < 1e-10


def test_supercell_map_ordering_and_home():
    sc = build_supercell(CELL, POS, [0, 0], (2, 1, 1))
    assert sc.n_sc == 4 and sc.n_prim == 2
    # home cell first: sites 0,1 are μ=0,1 at R=0
    assert sc.mu_of_site.tolist() == [0, 1, 0, 1]
    assert sc.rint_of_site.tolist() == [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert sc.home_sites.tolist() == [0, 1]
    # the R=(1,0,0) image sits one primitive lattice vector along a1
    assert np.allclose(sc.positions_super[2], POS[0] + CELL[0])


def test_supercell_cell_and_count():
    sc = build_supercell(CELL, POS, [0, 0], (2, 3, 4))
    assert sc.n_sc == 2 * 2 * 3 * 4
    assert np.allclose(sc.cell_super, np.diag([2, 3, 4]) @ CELL)


def test_fold_1x1x1_matches_direct_gamma():
    # a 1×1×1 "supercell" folded at Γ must reproduce direct mass-weighted
    # diagonalization of the same (symmetric) Hessian
    rng = np.random.default_rng(0)
    h = rng.standard_normal((2, 3, 2, 3))
    h = h + h.transpose(2, 3, 0, 1)  # symmetric
    sc = build_supercell(CELL, POS, [0, 0], (1, 1, 1))
    f_fold = frequencies_at_q(h, sc, MASSES, [0, 0, 0])  # phi_home == h here
    f_direct = gamma_frequencies(h, MASSES)
    assert np.allclose(np.sort(f_fold), np.sort(f_direct), atol=1e-8)


def test_dynamical_matrix_is_hermitian():
    rng = np.random.default_rng(1)
    sc = build_supercell(CELL, POS, [0, 0], (2, 2, 2))
    phi = rng.standard_normal((2, 3, sc.n_sc, 3))
    d = dynamical_matrix(phi, sc, MASSES, [0.3, -0.15, 0.2])
    assert np.allclose(d, d.conj().T)


def test_acoustic_sum_rule_zeroes_row_sum():
    rng = np.random.default_rng(2)
    sc = build_supercell(CELL, POS, [0, 0], (2, 2, 2))
    phi = apply_acoustic_sum_rule(rng.standard_normal((2, 3, sc.n_sc, 3)))
    assert np.abs(phi.sum(axis=2)).max() < 1e-12


def test_symmetrize_enforces_pair_symmetry():
    # after symmetrization every block must satisfy Φ_μν(R) = Φ_νμ(−R)ᵀ
    # (the near-zero acoustic modes this enables are checked on real Si in the
    # integration test — random force constants don't satisfy the extra physical
    # structure that zeroes them)
    rng = np.random.default_rng(3)
    sc = build_supercell(CELL, POS, [0, 0], (2, 2, 2))
    phi = symmetrize_force_constants(rng.standard_normal((2, 3, sc.n_sc, 3)), sc)
    look = _site_lookup(sc)
    n = np.array(sc.supercell)
    worst = 0.0
    for s in range(sc.n_sc):
        r = sc.rint_of_site[s]
        nu = int(sc.mu_of_site[s])
        neg_r = tuple((-r) % n)
        for mu in range(sc.n_prim):
            s2 = look[(neg_r, mu)]
            worst = max(worst, float(np.abs(phi[mu, :, s, :] - phi[nu, :, s2, :].T).max()))
    assert worst < 1e-12


def test_symmetrize_is_idempotent():
    rng = np.random.default_rng(4)
    sc = build_supercell(CELL, POS, [0, 0], (2, 2, 2))
    phi = rng.standard_normal((2, 3, sc.n_sc, 3))
    once = symmetrize_force_constants(phi, sc)
    twice = symmetrize_force_constants(once, sc)
    assert np.allclose(once, twice, atol=1e-12)


def test_build_supercell_rejects_bad_size():
    with pytest.raises(ValueError, match="positive"):
        build_supercell(CELL, POS, [0, 0], (2, 0, 1))
