"""Degeneracy-robust eigh backward for the metals joint descent (#129).

``RobustEigh`` keeps the forward exact (``torch.linalg.eigh``) but replaces the
eigenvector-response denominators ``1/(λ_i−λ_j)`` with the Lorentzian-broadened
``F_ij = (λ_j−λ_i)/((λ_i−λ_j)²+ε²)``. Two properties matter:

1. away from degeneracy (``ε`` tiny relative to the level spacing) the backward
   reproduces ``torch.linalg.eigh``'s own gradient / a finite difference, and
2. at *exact* degeneracy the backward is finite (no NaN), where the bare
   ``1/(λ_i−λ_j)`` diverges.
"""

import torch

from gradwave.opt._metals import robust_eigh

torch.set_default_dtype(torch.float64)


def _hermitian(n, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, dtype=dtype, generator=g)
    return (a + a.conj().transpose(-2, -1)) / 2


def _gauge_invariant_loss(w, v, m, s, a):
    """A scalar that is invariant to the eigenvectors' per-column phase.

    ``V diag(s(λ)) Vᴴ`` and ``λ`` are gauge invariant, so its finite-difference
    gradient is well defined even though ``eigh``'s eigenvectors are not.
    """
    recon = v @ torch.diag((s * w).to(v.dtype)) @ v.conj().transpose(-2, -1)
    return (m * recon.conj()).real.sum() + (a * w).sum()


def _run_match(dtype, eps=1e-8):
    n = 6
    a0 = _hermitian(n, dtype, seed=3)
    m = _hermitian(n, dtype, seed=4)
    s = torch.randn(n)
    lin = torch.randn(n)

    a_ref = a0.clone().requires_grad_(True)
    w, v = torch.linalg.eigh(a_ref)
    _gauge_invariant_loss(w, v, m, s, lin).backward()

    a_rob = a0.clone().requires_grad_(True)
    w2, v2 = robust_eigh(a_rob, eps)
    _gauge_invariant_loss(w2, v2, m, s, lin).backward()

    # forward is exact
    assert torch.allclose(w, w2)
    # backward matches torch's own eigh gradient away from degeneracy
    assert torch.allclose(a_ref.grad, a_rob.grad, atol=1e-9)


def test_robust_eigh_matches_torch_real():
    _run_match(torch.float64)


def test_robust_eigh_matches_torch_complex():
    _run_match(torch.complex128)


def _hermitian_from_real(x, n):
    """Build an n×n complex Hermitian matrix from a real leaf vector.

    Layout: first n entries the real diagonal, then the real and imaginary parts
    of the strict upper triangle. A real parametrization lets ``gradcheck`` probe
    the robust backward in real space, sidestepping the complex-Wirtinger
    conventions of a hand-rolled finite difference.
    """
    a = torch.zeros(n, n, dtype=torch.complex128)
    a[range(n), range(n)] = x[:n].to(torch.complex128)
    k = n
    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = x[k] + 1j * x[k + 1]
            a[j, i] = x[k] - 1j * x[k + 1]
            k += 2
    return a


def test_robust_eigh_gradcheck_away_from_degeneracy():
    """``torch.autograd.gradcheck`` (analytic backward vs central finite
    difference) on a non-degenerate Hermitian matrix, via a real parametrization
    and a gauge-invariant loss."""
    n = 4
    gen = torch.Generator().manual_seed(11)
    m = _hermitian(n, torch.complex128, seed=12)
    s = torch.randn(n, generator=torch.Generator().manual_seed(13))
    lin = torch.randn(n, generator=torch.Generator().manual_seed(14))
    x0 = torch.randn(n + n * (n - 1), dtype=torch.float64, generator=gen)

    def fn(x):
        a = _hermitian_from_real(x, n)
        w, v = robust_eigh(a, 1e-9)  # ε ≪ gaps: exact-eigh limit
        return _gauge_invariant_loss(w, v, m, s, lin)

    x = x0.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_robust_eigh_finite_at_exact_degeneracy():
    """The bare 1/(λ_i−λ_j) backward blows up (NaN/inf) when levels coincide; the
    broadened one must stay finite. Uses an exactly doubly-degenerate matrix and
    a loss with explicit eigenvector entries (so the degenerate off-diagonal
    response is genuinely exercised)."""
    a0 = torch.diag(torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64))
    # exact double degeneracy
    assert abs(float(torch.linalg.eigvalsh(a0)[:2].diff())) < 1e-14
    # a fixed weight matrix contracting the eigenvectors (NOT gauge invariant —
    # deliberately, to force a non-zero cotangent in the degenerate block)
    c = torch.arange(1.0, 10.0).reshape(3, 3)

    def loss(w, v):
        return (w ** 2).sum() + (c * v.real).sum()

    a = a0.clone().requires_grad_(True)
    w, v = robust_eigh(a, 0.3)
    loss(w, v).backward()
    assert torch.isfinite(a.grad).all()

    # sanity: the naive (unbroadened) eigh backward is NOT finite here
    a_naive = a0.clone().requires_grad_(True)
    wn, vn = torch.linalg.eigh(a_naive)
    loss(wn, vn).backward()
    assert not torch.isfinite(a_naive.grad).all()
