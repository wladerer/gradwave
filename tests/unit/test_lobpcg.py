"""Block LOBPCG must recover the nb lowest eigenpairs of a fixed Hermitian
operator, matching a dense eigh and the block-Davidson solver it stands in for.

Same synthetic batched Hermitian operator as the CheFSI test: a plane-wave-like
spectrum (wide kinetic tail + low cluster) exposed through an h_apply with the
exact (nk, nb, npw) contract the real BatchedHamiltonian uses. The diagonal
kinetic tail is passed as the preconditioner `t`, so the Teter path is exercised
(not the identity K=1 that t=0 collapses to). No SCF, no pseudopotentials — this
isolates the eigensolver.
"""

import torch

from gradwave.solvers.davidson import davidson_batched
from gradwave.solvers.lobpcg import lobpcg_batched, lobpcg_batched_ms


def _make_operator(nk, npw, seed=0):
    """Random Hermitian H_k with a wide kinetic-like spectrum.

    Returns (H, apply, mask, t) where t is the diagonal kinetic tail, usable
    directly as the Teter preconditioner's kinetic diagonal.
    """
    torch.manual_seed(seed)
    diag = torch.linspace(1.0, 100.0, npw, dtype=torch.float64)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        herm = 0.5 * (a + a.conj().T)
        herm = herm + torch.diag(diag.to(torch.complex128))
        h[k] = 0.5 * (herm + herm.conj().T)
    mask = torch.ones(nk, npw, dtype=torch.bool)
    t = diag.expand(nk, npw).contiguous()

    def apply(c):  # (nk, nb, npw) -> (nk, nb, npw)
        return torch.einsum("kij,kbj->kbi", h, c)

    return h, apply, mask, t


def _reference_eigs(h, nb):
    w = torch.linalg.eigvalsh(h)  # (nk, npw) ascending
    return w[:, :nb]


def test_recovers_lowest_eigenvalues():
    nk, npw, nb = 4, 80, 8
    h, apply, mask, t = _make_operator(nk, npw, seed=2)
    ref = _reference_eigs(h, nb)

    torch.manual_seed(99)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    res = lobpcg_batched(apply, x0, t, mask, tol=1e-9, max_iter=100)
    assert float((res.eigenvalues - ref).abs().max()) < 1e-7
    assert float(res.residual_norms.max()) < 1e-8


def test_eigenvectors_are_orthonormal_and_eigen():
    nk, npw, nb = 2, 50, 6
    h, apply, mask, t = _make_operator(nk, npw, seed=3)
    torch.manual_seed(7)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    res = lobpcg_batched(apply, x0, t, mask, tol=1e-9, max_iter=100)

    x = res.eigenvectors
    gram = torch.einsum("kag,kbg->kab", x.conj(), x)
    eye = torch.eye(nb, dtype=torch.complex128).expand(nk, nb, nb)
    assert float((gram - eye).abs().max()) < 1e-8
    hx = apply(x)
    resid = hx - res.eigenvalues[..., None] * x
    assert float(torch.linalg.norm(resid, dim=-1).max()) < 1e-7


def test_matches_davidson():
    """The non-negotiable: LOBPCG and Davidson agree on the lowest nb to ~1e-9."""
    nk, npw, nb = 3, 70, 7
    h, apply, mask, t = _make_operator(nk, npw, seed=4)
    torch.manual_seed(11)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    lob = lobpcg_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=100)
    dav = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=100)
    assert float((lob.eigenvalues - dav.eigenvalues).abs().max()) < 1e-9


def test_mixed_precision_polishes_to_full():
    nk, npw, nb = 2, 60, 6
    h, apply, mask, t = _make_operator(nk, npw, seed=5)
    ref = _reference_eigs(h, nb)
    torch.manual_seed(13)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    res = lobpcg_batched_ms(
        apply, x0, t, mask, tol=1e-9, max_iter=100, crossover=1e-4
    )
    assert float((res.eigenvalues - ref).abs().max()) < 1e-7
