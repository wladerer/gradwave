"""CheFSI must recover the nb lowest eigenpairs of a fixed Hermitian operator,
matching a dense eigh and the block-Davidson solver it can stand in for.

The operator is a synthetic batched dense Hermitian matrix with a plane-wave-
like spectrum (a wide, mostly-flat-at-the-top kinetic tail plus a low cluster),
exposed through an h_apply with the exact (nk, nb, npw) contract the real
BatchedHamiltonian uses. No SCF, no pseudopotentials — this isolates the
eigensolver.
"""

import torch

from gradwave.solvers.chebyshev import (
    _lanczos_bounds,
    chebyshev_filtered_batched,
    chebyshev_filtered_batched_ms,
)
from gradwave.solvers.davidson import davidson_batched


def _make_operator(nk, npw, seed=0):
    """Random Hermitian H_k with a realistic wide spectrum; returns (H, apply, mask)."""
    torch.manual_seed(seed)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        herm = 0.5 * (a + a.conj().T)
        # add a wide diagonal kinetic-like tail so the spectrum spans ~0..100,
        # the regime where Chebyshev filtering earns its keep
        diag = torch.linspace(0.0, 100.0, npw, dtype=torch.float64)
        herm = herm + torch.diag(diag.to(torch.complex128))
        h[k] = 0.5 * (herm + herm.conj().T)
    mask = torch.ones(nk, npw, dtype=torch.bool)

    def apply(c):  # (nk, nb, npw) -> (nk, nb, npw)
        return torch.einsum("kij,kbj->kbi", h, c)

    return h, apply, mask


def _reference_eigs(h, nb):
    w = torch.linalg.eigvalsh(h)  # (nk, npw) ascending
    return w[:, :nb]


def test_lanczos_brackets_the_spectrum():
    nk, npw = 3, 60
    h, apply, mask = _make_operator(nk, npw, seed=1)
    w = torch.linalg.eigvalsh(h)
    lo, hi = _lanczos_bounds(apply, mask, steps=8)
    # rigorous bracket: lo below the smallest, hi above the largest, per k
    assert bool((lo <= w[:, 0] + 1e-6).all())
    assert bool((hi >= w[:, -1] - 1e-6).all())
    # and not absurdly loose
    assert bool((hi - w[:, -1] < 0.5 * (w[:, -1] - w[:, 0])).all())


def test_recovers_lowest_eigenvalues():
    nk, npw, nb = 4, 80, 8
    h, apply, mask = _make_operator(nk, npw, seed=2)
    ref = _reference_eigs(h, nb)

    torch.manual_seed(99)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    t = torch.zeros(nk, npw, dtype=torch.float64)  # unused by CheFSI
    res = chebyshev_filtered_batched(
        apply, x0, t, mask, tol=1e-9, max_iter=60, degree=12
    )
    assert float((res.eigenvalues - ref).abs().max()) < 1e-7
    assert float(res.residual_norms.max()) < 1e-8


def test_eigenvectors_are_orthonormal_and_eigen():
    nk, npw, nb = 2, 50, 6
    h, apply, mask = _make_operator(nk, npw, seed=3)
    torch.manual_seed(7)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    res = chebyshev_filtered_batched(apply, x0, t, mask, tol=1e-9, max_iter=60, degree=12)

    x = res.eigenvectors
    gram = torch.einsum("kag,kbg->kab", x.conj(), x)
    eye = torch.eye(nb, dtype=torch.complex128).expand(nk, nb, nb)
    assert float((gram - eye).abs().max()) < 1e-8
    # H x_a ≈ eig_a x_a
    hx = apply(x)
    resid = hx - res.eigenvalues[..., None] * x
    assert float(torch.linalg.norm(resid, dim=-1).max()) < 1e-7


def test_matches_davidson():
    nk, npw, nb = 3, 70, 7
    h, apply, mask = _make_operator(nk, npw, seed=4)
    torch.manual_seed(11)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    t = torch.zeros(nk, npw, dtype=torch.float64)

    che = chebyshev_filtered_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=80, degree=12)
    dav = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=80)
    assert float((che.eigenvalues - dav.eigenvalues).abs().max()) < 1e-7


def test_mixed_precision_polishes_to_full():
    nk, npw, nb = 2, 60, 6
    h, apply, mask = _make_operator(nk, npw, seed=5)
    ref = _reference_eigs(h, nb)
    torch.manual_seed(13)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    res = chebyshev_filtered_batched_ms(
        apply, x0, t, mask, tol=1e-9, max_iter=80, degree=12, crossover=1e-4
    )
    # the fp64 polish removes the draft error
    assert float((res.eigenvalues - ref).abs().max()) < 1e-7
