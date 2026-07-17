"""The differentiable moment penalties (scf/moment_penalty.py). Autograd must
reproduce the closed-form constraining field and direction gradient for both the
direction-only 'perp' penalty and the magnitude-robust 'vector' penalty. Pure
tensor algebra — no SCF, so this runs in the fast gate and pins the AD machinery
that the constrained SCF and the config search both rely on."""

import torch

from gradwave.scf.moment_penalty import (
    direction_gradient,
    field_coeff,
    penalty_energy,
)


def _setup():
    torch.manual_seed(0)
    M = torch.randn(3, 3, dtype=torch.float64)
    dirs = torch.randn(3, 3, dtype=torch.float64)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    return M, dirs


def _transverse(v, e):
    return v - (v * e).sum(-1, keepdim=True) * e


def test_perp_field_and_gradient_match_closed_form():
    M, dirs = _setup()
    lam = 2.5
    Mperp = _transverse(M, dirs)
    # field ∂E_p/∂M = 2λ M^perp
    assert torch.allclose(field_coeff(M, dirs, lam, "perp"), 2 * lam * Mperp)
    # dW/dê ⟂ ê = -2λ (M·ê) M^perp
    m_dot_e = (M * dirs).sum(-1, keepdim=True)
    assert torch.allclose(direction_gradient(M, dirs, lam, "perp"),
                          -2 * lam * m_dot_e * Mperp)
    assert torch.allclose(penalty_energy(M, dirs, lam, "perp"),
                          lam * (Mperp ** 2).sum())


def test_vector_field_and_gradient_match_closed_form():
    M, dirs = _setup()
    lam = 1.7
    m0 = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    resid = M - m0.unsqueeze(-1) * dirs
    # field ∂E_p/∂M = 2λ (M - m0 ê)
    assert torch.allclose(field_coeff(M, dirs, lam, "vector", m0), 2 * lam * resid)
    # dW/dê ⟂ ê = -2λ m0 M^perp  (the m0 ê term is longitudinal, projected out)
    Mperp = _transverse(M, dirs)
    assert torch.allclose(direction_gradient(M, dirs, lam, "vector", m0),
                          -2 * lam * m0.unsqueeze(-1) * Mperp)
    assert torch.allclose(penalty_energy(M, dirs, lam, "vector", m0),
                          lam * (resid ** 2).sum())


def test_vector_penalty_costs_demagnetization():
    """The point of the magnitude-robust penalty: driving M → 0 is *not* free.
    'perp' is minimized at M = 0 (the demagnetization loophole); 'vector' charges
    λ Σ m0² for it, so a collapsed moment is uphill, not downhill."""
    _, dirs = _setup()
    lam = 3.0
    m0 = torch.full((3,), 1.4, dtype=torch.float64)
    zero = torch.zeros(3, 3, dtype=torch.float64)
    aligned = m0.unsqueeze(-1) * dirs                       # M = m0 ê (on target)
    assert float(penalty_energy(zero, dirs, lam, "perp")) == 0.0     # free
    assert float(penalty_energy(zero, dirs, lam, "vector", m0)) > 0.0  # costs
    assert torch.allclose(penalty_energy(aligned, dirs, lam, "vector", m0),
                          torch.zeros((), dtype=torch.float64), atol=1e-12)
