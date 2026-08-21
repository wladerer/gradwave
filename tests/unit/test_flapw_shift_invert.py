"""Unit tests for the OPT-IN shift-invert Lanczos secular solver (``solve_geneig_shift_invert``).

The two things that must hold: (1) when it returns a result it equals the dense generalized
eigensolve to solver tolerance — including an exact degenerate multiplet (the Γ-point case the
thick restart exists for); (2) its two-shift inertia completeness certificate REJECTS (returns
``None`` → the caller's loud dense fallback) a deliberately mis-placed σ, so a bad shift can never
yield a silently-incomplete window.
"""

from __future__ import annotations

import numpy as np

from gradwave.flapw.lapw import solve_geneig, solve_geneig_shift_invert


def _pencil(dim, eigs, rng):
    """A Hermitian generalized pencil ``H c = ε S c`` with SPD ``S`` and prescribed spectrum
    ``eigs`` (S-orthonormal eigenvectors), so exact degeneracies can be planted."""
    b = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    s = b @ b.conj().T + dim * np.eye(dim)
    lc = np.linalg.cholesky(s)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(m)                                   # unitary
    v = np.linalg.solve(lc.conj().T, q)                      # columns S-orthonormal: V^H S V = I
    h = s @ v @ np.diag(eigs) @ v.conj().T @ s
    return 0.5 * (h + h.conj().T), 0.5 * (s + s.conj().T)


def test_shift_invert_matches_dense():
    """Parity with the dense solve on a generic pencil (σ just below the window bottom)."""
    rng = np.random.default_rng(3)
    dim, nb = 90, 14
    eigs = np.sort(rng.uniform(-5.0, 20.0, dim))
    h, s = _pencil(dim, eigs, rng)
    ref = solve_geneig(h, s, nb)
    out = solve_geneig_shift_invert(h, s, nb, sigma=float(ref[0]) - 1.0)
    assert out is not None, "well-placed σ should engage"
    ev, _vecs = out
    assert np.abs(np.sort(ev) - ref).max() < 1e-8


def test_shift_invert_resolves_degenerate_multiplet():
    """A planted 3-fold degenerate level inside the window is recovered with the right
    multiplicity (the certificate would reject a missed copy) and matches dense to tol."""
    rng = np.random.default_rng(5)
    dim, nb = 80, 12
    eigs = np.sort(rng.uniform(-4.0, 15.0, dim))
    eigs[4] = eigs[5] = eigs[6] = 2.5                       # exact 3-fold multiplet in-window
    eigs = np.sort(eigs)
    h, s = _pencil(dim, eigs, rng)
    ref = solve_geneig(h, s, nb)
    out = solve_geneig_shift_invert(h, s, nb, sigma=float(ref[0]) - 1.0)
    assert out is not None
    ev = np.sort(out[0])
    assert np.abs(ev - ref).max() < 1e-7
    assert int(np.sum(np.abs(ev - 2.5) < 1e-6)) == 3        # all three copies present


def test_shift_invert_misplaced_sigma_falls_back():
    """A σ dropped in the MIDDLE of the spectrum cannot isolate the lowest window; the inertia
    certificate must reject (return None) rather than hand back a wrong/incomplete set."""
    rng = np.random.default_rng(7)
    dim, nb = 90, 12
    eigs = np.sort(rng.uniform(-5.0, 20.0, dim))
    h, s = _pencil(dim, eigs, rng)
    bad_sigma = float(eigs[dim // 2])                       # deep bands sit far below σ → missed
    assert solve_geneig_shift_invert(h, s, nb, sigma=bad_sigma) is None


def test_shift_invert_small_matrix_declines():
    """Below the headroom threshold the solver declines (returns None) so the caller uses dense."""
    rng = np.random.default_rng(9)
    dim, nb = 30, 8
    eigs = np.sort(rng.uniform(-5.0, 20.0, dim))
    h, s = _pencil(dim, eigs, rng)
    assert solve_geneig_shift_invert(h, s, nb, sigma=float(eigs[0]) - 1.0) is None
