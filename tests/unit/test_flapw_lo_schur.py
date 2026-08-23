"""Unit tests for the LO overlap Schur-complement conditioning diagnostic."""
import numpy as np
import pytest

from gradwave.flapw.lo_schur import (
    lo_conditioning_report,
    lo_labels,
    lo_overlap_schur,
    lo_resid_fracs,
)


def _S_with_los(lo_vectors, napw=3):
    """Extended overlap S = Gram matrix of [e0..e_{napw-1}, *lo_vectors] in an orthonormal
    ambient space (APW = the first napw unit axes). Each lo_vector is a full ambient-space
    coordinate; its APW overlap and residual are then exact by construction."""
    dim = max(napw, max((len(v) for v in lo_vectors), default=napw))
    basis = [np.eye(1, dim, i).ravel().astype(complex) for i in range(napw)]
    for v in lo_vectors:
        w = np.zeros(dim, dtype=complex)
        w[: len(v)] = v
        basis.append(w)
    B = np.array(basis)                      # (napw+nlo, dim)
    return B @ B.conj().T                    # Gram = overlap


def test_resid_frac_redundant_and_distinct():
    # LO0 = e0 (entirely in the APW span → resid_frac 0); LO1 = e3 (orthogonal → resid_frac 1)
    S = _S_with_los([[1, 0, 0, 0], [0, 0, 0, 1]], napw=3)
    lam, _ = lo_resid_fracs(S, npw=3)
    assert np.allclose(np.sort(lam), [0.0, 1.0], atol=1e-10)


def test_resid_frac_partial_redundancy():
    # LO = (e0 + e3)/√2 → half its norm lies in the APW span → resid_frac 0.5
    r = 1.0 / np.sqrt(2)
    S = _S_with_los([[r, 0, 0, r]], napw=3)
    lam, _ = lo_resid_fracs(S, npw=3)
    assert lam.shape == (1,)
    assert lam[0] == pytest.approx(0.5, abs=1e-10)


def test_overlap_schur_hermitian_and_shape():
    S = _S_with_los([[1, 0, 0, 0], [0, 0, 0, 1]], napw=3)
    eff = lo_overlap_schur(S, npw=3)
    assert eff.shape == (2, 2)
    assert np.allclose(eff, eff.conj().T)


def test_conditioning_report_flags_and_attributes():
    # LO0 redundant (e0), LO1 distinct (e3); labels name them
    S = _S_with_los([[1, 0, 0, 0], [0, 0, 0, 1]], napw=3)
    labels = [(0, "O", 0, 0), (0, "O", 1, 0)]
    redundant, report = lo_conditioning_report(S, npw=3, labels=labels, tol=0.05)
    assert len(redundant) == 1
    frac, label = redundant[0]
    assert frac < 0.05
    assert label == (0, "O", 0, 0)                 # the redundant LO is attributed correctly
    assert "resid_frac" in report


def test_no_lo_block_raises():
    with pytest.raises(ValueError):
        lo_overlap_schur(np.eye(3, dtype=complex), npw=3)


def test_lo_labels_ordering():
    """Atoms outer, LOs inner, m = −l..l innermost — matching the secular build."""
    atoms_cart = [(None, "O"), (None, "Ti")]
    lodat = {"O": [{"l": 0}, {"l": 1}], "Ti": [{"l": 1}]}
    labels = lo_labels(atoms_cart, lodat)
    assert labels == [
        (0, "O", 0, 0),
        (0, "O", 1, -1), (0, "O", 1, 0), (0, "O", 1, 1),
        (1, "Ti", 1, -1), (1, "Ti", 1, 0), (1, "Ti", 1, 1),
    ]
