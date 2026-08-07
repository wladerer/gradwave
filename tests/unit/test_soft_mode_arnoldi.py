"""SoftModeDeflate P1: the Arnoldi soft-subspace extractor (synthetic, fast).

Exercises the non-symmetric Krylov linear algebra on a planted-spectrum operator
— a real, non-normal matrix whose eigenvalues are exactly known — so the extractor
is validated deterministically without paying for a DFT response solve. The
physical tie-in (the extractor on the real screening operator) lives in the
integration test.
"""

import pytest
import torch

from gradwave.scf.soft_mode import (
    arnoldi_factorization,
    soft_subspace_from_operator,
)


def _planted_operator(n=40, soft=0.96, hard=-2.20, noise=0.05, seed=7):
    """Upper-triangular (so eigenvalues == diagonal) with a random strictly-upper
    part (so the operator is non-normal, like M = K_Hxc·χ₀). The diagonal is a
    known real spectrum spanning [hard, soft]; the max-real eigenvalue is ``soft``
    (near +1, the soft mode) and the dominant-magnitude one is ``hard``.

    ``noise`` is kept mild: this is the benign, near-normal regime P1 targets
    (P0 measured near-real eigenpairs on Si). Strong non-normality — where a
    truncated Krylov Ritz value overshoots the true spectrum toward the numerical
    range — is the criticality/defective regime P2 handles via the ``max_imag`` and
    residual warnings, not something the P1 extractor is expected to nail."""
    g = torch.Generator().manual_seed(seed)
    diag = torch.linspace(hard, soft, n, dtype=torch.float64)
    a = torch.triu(noise * torch.randn(n, n, generator=g, dtype=torch.float64), 1)
    a = a + torch.diag(diag)
    return a, diag


def test_arnoldi_full_recovers_the_spectrum():
    a, diag = _planted_operator()
    n = a.shape[0]

    def apply(x):
        return a @ x

    g = torch.Generator().manual_seed(1)
    v0 = torch.randn(n, generator=g, dtype=torch.float64)
    v_list, h, reached = arnoldi_factorization(apply, v0, krylov=n)

    # V is orthonormal
    for i in range(reached):
        assert abs(float(torch.dot(v_list[i], v_list[i])) - 1.0) < 1e-9
        for j in range(i):
            assert abs(float(torch.dot(v_list[i], v_list[j]))) < 1e-8

    # every Ritz value is a true eigenvalue (== a diagonal entry)
    ritz = torch.linalg.eigvals(h[:reached, :reached])
    for lam in ritz:
        assert lam.imag.abs() < 1e-6
        assert float((diag - lam.real).abs().min()) < 1e-6, lam
    # and full Krylov spans the space (non-derogatory operator, generic start)
    assert reached >= n - 1


def test_soft_subspace_selects_the_soft_mode():
    a, _ = _planted_operator()
    n = a.shape[0]

    def apply(x):
        return a @ x

    ref = torch.zeros(n, dtype=torch.float64)  # only its shape/dtype are used
    sub = soft_subspace_from_operator(apply, ref, krylov=24, n_modes=1, seed=3)

    assert len(sub.values) == 1
    lam = sub.values[0].real
    # the extremal max-real eigenvalue converges fast in Arnoldi
    assert lam == pytest.approx(0.96, abs=0.01), sub.values
    # true operator residual ‖M q − θ q‖ is small: a genuine eigenpair
    assert sub.residuals[0] < 1e-3, sub.residuals
    # real spectrum → no non-normality flag on the selected mode
    assert sub.max_imag < 1e-8, sub.max_imag

    # the recovered Ritz vector is a real eigenvector of A
    q = sub.vectors[0]
    assert float(torch.linalg.vector_norm(a @ q - lam * q)) < 1e-3
