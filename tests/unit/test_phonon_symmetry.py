"""HessianSymmetry: irreducible-displacement selection and reconstruction.

The decisive check is synthetic: project a random Hessian onto the
group-invariant subspace, hand HessianSymmetry only the columns it asks
for, and demand the full matrix back exactly. No SCF anywhere.
"""

import numpy as np
import pytest

from gradwave.postscf.phonons import HessianSymmetry, gamma_frequencies

A = 5.43
SI_CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0, 0], [A / 4] * 3])
GE_POS = SI_POS  # zincblende: same sites, different species


def _symmetrize_hessian(h, hs):
    """(1/N) Σ_g P_g H P_gᵀ with the block action H[g(b),g(a)] = S H Sᵀ,
    then transpose symmetrization — the exact invariance reconstruct()
    relies on."""
    na = hs.na
    acc = np.zeros_like(h)
    for s, amap in zip(hs.s_cart, hs.sg.atom_map, strict=True):
        for a in range(na):
            for b in range(na):
                acc[amap[b], :, amap[a], :] += s @ h[b, :, a, :] @ s.T
    acc /= hs.sg.n_ops
    h2 = acc.reshape(3 * na, 3 * na)
    return (0.5 * (h2 + h2.T)).reshape(na, 3, na, 3)


@pytest.mark.parametrize(
    ("species", "n_irr"),
    [([0, 0], 1), ([0, 1], 2)],  # diamond Si: 1 column; zincblende: 2
)
def test_reconstruct_synthetic(species, n_irr):
    hs = HessianSymmetry(SI_CELL, SI_POS, species)
    assert len(hs.displacements) == n_irr

    rng = np.random.default_rng(7)
    h = _symmetrize_hessian(rng.normal(size=(2, 3, 2, 3)), hs)
    cols = [h[:, :, a, alpha] for a, alpha in hs.displacements]
    h_rec = hs.reconstruct(cols)
    np.testing.assert_allclose(h_rec, h, atol=1e-12)


def test_reconstruct_low_symmetry_needs_all_columns():
    # displaced positions AND distinct species: two same-species atoms
    # always share an inversion center (3 columns would suffice), so P1
    # requires both
    pos = np.array([[0.0, 0, 0], [1.31, 1.42, 1.29]])
    hs = HessianSymmetry(SI_CELL, pos, [0, 1])
    assert len(hs.displacements) == 6  # no reduction, and still exact
    rng = np.random.default_rng(3)
    h = _symmetrize_hessian(rng.normal(size=(2, 3, 2, 3)), hs)
    cols = [h[:, :, a, alpha] for a, alpha in hs.displacements]
    np.testing.assert_allclose(hs.reconstruct(cols), h, atol=1e-12)


def test_gamma_frequencies_translation_modes():
    hs = HessianSymmetry(SI_CELL, SI_POS, [0, 0])
    rng = np.random.default_rng(11)
    h = _symmetrize_hessian(rng.normal(size=(2, 3, 2, 3)), hs)
    # impose the acoustic sum rule, then the three translations are exact
    # zero modes of the mass-weighted matrix
    for a in range(2):
        h[a, :, a, :] -= h[a].sum(axis=1)
    h2 = h.reshape(6, 6)
    h = (0.5 * (h2 + h2.T)).reshape(2, 3, 2, 3)
    freqs = gamma_frequencies(h, [28.0855, 28.0855])
    assert np.abs(freqs[:3]).max() < 1e-4 * max(1.0, np.abs(freqs).max())
