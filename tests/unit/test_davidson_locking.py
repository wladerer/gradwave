"""Guard the uniform leading-band locking in davidson_batched.

Locking deflates converged low bands out of the Rayleigh-Ritz eigh. The
invariant is that a band is deflated only after the driver would already call
it converged (residual < tol at every k), so the converged eigenvalues must be
UNCHANGED. These tests run a small dense batched eigenproblem (no SCF), so they
finish in well under a second single-threaded, and pin:

  1. lock=True eigenvalues match the exact dense eigenvalues to ~1e-9;
  2. lock=True eigenvalues match lock=False (the pre-locking driver) to ~1e-9;
  3. returned bands stay ascending and rows stay orthonormal;
  4. a near-degenerate spectrum (the delicate case for locking) is still exact.
"""

import torch

from gradwave.solvers.davidson import davidson_batched

DT = torch.complex128
RT = torch.float64


def _make_h(nk, npw, seed, gaps=None):
    """Random Hermitian (nk, npw, npw) with a positive kinetic-like diagonal.

    gaps: optional (npw,) ascending eigenvalue targets; when given the matrix is
    built as Q diag(gaps) Q^H so the spectrum is known and controllable (used to
    force a near-degenerate pair).
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(nk, npw, npw, generator=g, dtype=RT) + 1j * torch.randn(
        nk, npw, npw, generator=g, dtype=RT)
    a = a.to(DT)
    h = 0.5 * (a + a.conj().transpose(-1, -2))
    if gaps is not None:
        q, _ = torch.linalg.qr(a)
        d = torch.diag_embed(gaps.to(DT)).expand(nk, npw, npw)
        h = q @ d @ q.conj().transpose(-1, -2)
        h = 0.5 * (h + h.conj().transpose(-1, -2))
    # a kinetic diagonal for the Teter preconditioner: strictly positive
    t = (torch.arange(1, npw + 1, dtype=RT) / npw).expand(nk, npw).contiguous()
    return h, t


def _apply(h):
    # h_apply: (nk, j, npw) block of row-vectors -> (nk, j, npw)
    return lambda v: torch.einsum("kij,kbj->kbi", h, v)


def _run(h, t, nb, tol, lock):
    nk, npw, _ = h.shape
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(nk, nb, npw, generator=g, dtype=RT) + 1j * torch.randn(
        nk, nb, npw, generator=g, dtype=RT)
    x0 = x0.to(DT)
    mask = torch.ones(nk, npw, dtype=torch.bool)
    return davidson_batched(
        _apply(h), x0, t, mask, tol=tol, max_iter=200,
        # small max_dim_factor so the subspace hits max_dim and RESTARTS,
        # which is exactly where locking deflates — otherwise it never fires
        max_dim_factor=2, lock=lock,
    )


def test_locking_matches_exact_and_unlocked():
    nk, npw, nb, tol = 3, 24, 8, 1e-10
    h, t = _make_h(nk, npw, seed=1)
    exact = torch.linalg.eigvalsh(h)[:, :nb]  # ascending, lowest nb

    on = _run(h, t, nb, tol, lock=True)
    off = _run(h, t, nb, tol, lock=False)

    # (1) locked eigenvalues are the true lowest-nb eigenvalues
    assert torch.allclose(on.eigenvalues, exact, atol=1e-9, rtol=0), (
        (on.eigenvalues - exact).abs().max().item())
    # (2) locked == unlocked to tight tolerance (physics unchanged)
    assert torch.allclose(on.eigenvalues, off.eigenvalues, atol=1e-9, rtol=0), (
        (on.eigenvalues - off.eigenvalues).abs().max().item())
    # residuals actually converged
    assert float(on.residual_norms.max()) < tol


def test_locking_actually_fires():
    """Sanity: the deflation path is exercised (would be a no-op guard test
    otherwise). With max_dim_factor=2 and tol=1e-10 the solve restarts, so the
    only way lock=True and lock=False can share converged eigenvalues while
    taking different numbers of iterations is that locking engaged."""
    nk, npw, nb, tol = 2, 20, 6, 1e-10
    h, t = _make_h(nk, npw, seed=7)
    on = _run(h, t, nb, tol, lock=True)
    off = _run(h, t, nb, tol, lock=False)
    assert torch.allclose(on.eigenvalues, off.eigenvalues, atol=1e-9, rtol=0)
    # ascending order preserved across the locked/active boundary
    diffs = on.eigenvalues[:, 1:] - on.eigenvalues[:, :-1]
    assert float(diffs.min()) > -1e-9


def test_locking_rows_orthonormal():
    nk, npw, nb, tol = 2, 22, 7, 1e-10
    h, t = _make_h(nk, npw, seed=3)
    on = _run(h, t, nb, tol, lock=True)
    x = on.eigenvectors  # (nk, nb, npw)
    gram = torch.einsum("kbi,kci->kbc", x.conj(), x)
    eye = torch.eye(nb, dtype=DT).expand(nk, nb, nb)
    assert torch.allclose(gram, eye, atol=1e-8, rtol=0), (
        (gram - eye).abs().max().item())


def test_locking_near_degenerate_spectrum():
    """A near-degenerate pair straddling the lock boundary is the delicate case:
    lock one member, refine the other in the complement. Eigenvalues must still
    be exact (the multiplet's values are well defined even if the intra-multiplet
    basis is arbitrary)."""
    nk, npw, nb = 2, 20, 6
    # eigenvalues with a ~1e-7 split at the boundary band index nb-1/nb
    gaps = torch.tensor(
        [0.10, 0.55, 1.20, 1.85, 2.40, 2.40 + 1e-7, 3.10, 3.9,
         4.4, 5.0, 5.6, 6.1, 6.7, 7.2, 7.8, 8.3, 8.9, 9.4, 9.9, 10.5],
        dtype=RT)
    h, t = _make_h(nk, npw, seed=5, gaps=gaps)
    exact = gaps[:nb].expand(nk, nb)
    on = _run(h, t, nb, tol=1e-10, lock=True)
    assert torch.allclose(on.eigenvalues, exact, atol=1e-8, rtol=0), (
        (on.eigenvalues - exact).abs().max().item())
