"""Unit tests for the supercell-phonon numerics (postscf.phonons_supercell).

No SCF: these check the (μ,R) supercell map, the D(q) Fourier folding against
direct Γ diagonalization, Hermiticity, and the sum-rule / symmetrization
bookkeeping. The physics (Si dispersion, acoustic modes) is exercised in
tests/integration/test_phonons_supercell.py.
"""

import numpy as np
import pytest

from gradwave.postscf.phonons import gamma_frequencies
from gradwave.postscf.phonons_supercell import (
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
