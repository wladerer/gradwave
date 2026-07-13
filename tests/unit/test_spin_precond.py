"""StonerSpinPrecond: the Woodbury inverse must be the exact inverse of
M = I − Ω·U^T diag(c) conj(W) (the rank-r model dielectric), and reduce to
the identity when the Fermi-surface weight vanishes."""

import torch

from gradwave.scf.spin_precond import StonerSpinPrecond


def test_woodbury_is_exact_inverse():
    torch.manual_seed(7)
    r, ng, vol = 5, 40, 123.4
    u = torch.randn(r, ng, dtype=torch.complex128)
    w = torch.randn(r, ng, dtype=torch.complex128)
    c = -torch.rand(r, dtype=torch.float64)  # w_k f' < 0
    pc = StonerSpinPrecond(u, w, c, vol)

    # dense M: M x = x − Σ_α (Ω c_α) u_α ⟨w_α, x⟩
    m = torch.eye(ng, dtype=torch.complex128)
    for a in range(r):
        m = m - vol * c[a] * torch.outer(u[a], w[a].conj())

    x = torch.randn(ng, dtype=torch.complex128)
    y = pc.apply(m @ x)
    assert float((y - x).abs().max()) < 1e-10


def test_small_weight_is_near_identity():
    torch.manual_seed(3)
    r, ng, vol = 3, 30, 50.0
    u = torch.randn(r, ng, dtype=torch.complex128)
    w = torch.randn(r, ng, dtype=torch.complex128)
    c = -1e-14 * torch.rand(r, dtype=torch.float64)
    pc = StonerSpinPrecond(u, w, c, vol)
    x = torch.randn(ng, dtype=torch.complex128)
    assert float((pc.apply(x) - x).abs().max()) < 1e-9
