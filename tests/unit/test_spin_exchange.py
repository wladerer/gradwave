"""The exchange-tensor decomposition and transverse basis (postscf/spin_exchange).
Pure tensor algebra — no SCF, so this runs in the fast gate and pins the
Heisenberg/DMI/anisotropic split that the (expensive) DFT extraction feeds into."""

import torch

from gradwave.postscf.spin_exchange import _transverse_basis, decompose


def test_decompose_recovers_heisenberg_dmi_anisotropic():
    j_iso, d, g, s = 0.031, 0.004, 0.0012, -0.0007   # eV
    # J = J_iso·I + antisym(D) + sym-traceless(g, s)
    J = torch.tensor([[j_iso + g, s + d],
                      [s - d, j_iso - g]], dtype=torch.float64)
    J_out, D_out, gamma = decompose(J)
    assert abs(J_out - j_iso) < 1e-12
    assert abs(D_out - d) < 1e-12
    assert torch.allclose(gamma, torch.tensor([[g, s], [s, -g]], dtype=torch.float64),
                          atol=1e-12)
    # gamma is symmetric and traceless
    assert abs(float(gamma.trace())) < 1e-12
    assert torch.allclose(gamma, gamma.T)


def test_pure_heisenberg_has_no_dmi_or_anisotropy():
    J = 0.05 * torch.eye(2, dtype=torch.float64)
    J_out, D_out, gamma = decompose(J)
    assert abs(J_out - 0.05) < 1e-12
    assert abs(D_out) < 1e-15
    assert float(gamma.abs().max()) < 1e-15


def test_transverse_basis_orthonormal_and_right_handed():
    for ref in ([0, 0, 1.0], [1.0, 0, 0], [1.0, 1, 1], [0.3, -0.7, 0.2]):
        r = torch.tensor(ref, dtype=torch.float64)
        u, v = _transverse_basis(r)
        rhat = r / torch.linalg.norm(r)
        assert abs(float(torch.dot(u, rhat))) < 1e-12       # u ⟂ ref
        assert abs(float(torch.dot(v, rhat))) < 1e-12       # v ⟂ ref
        assert abs(float(torch.dot(u, v))) < 1e-12          # u ⟂ v
        assert abs(float(torch.linalg.norm(u)) - 1) < 1e-12
        # right-handed: u × v = ref
        assert torch.allclose(torch.linalg.cross(u, v), rhat, atol=1e-12)
