import numpy as np
import torch

from gradwave.core.ylm import C00, ylm_all


def sphere_quadrature(ntheta=12, nphi=24):
    """Gauss–Legendre × uniform-φ quadrature, exact for Ylm products to l=5."""
    x, wx = np.polynomial.legendre.leggauss(ntheta)  # x = cosθ
    phi = 2 * np.pi * np.arange(nphi) / nphi
    wphi = 2 * np.pi / nphi
    ct, ph = np.meshgrid(x, phi, indexing="ij")
    st = np.sqrt(1 - ct**2)
    pts = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=-1).reshape(-1, 3)
    w = (wx[:, None] * wphi * np.ones(nphi)).reshape(-1)
    return torch.as_tensor(pts, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)


def test_orthonormality():
    pts, w = sphere_quadrature()
    y = ylm_all(3, pts)  # (npts, 16)
    gram = torch.einsum("pi,p,pj->ij", y, w, y)
    assert torch.allclose(gram, torch.eye(16, dtype=torch.float64), atol=1e-12)


def test_parity():
    # Y_lm(-n) = (-1)^l Y_lm(n)
    gen = torch.Generator().manual_seed(5)
    n = torch.randn(50, 3, generator=gen, dtype=torch.float64)
    yp, ym = ylm_all(3, n), ylm_all(3, -n)
    parity = torch.tensor([(-1.0) ** l for l in range(4) for _ in range(2 * l + 1)])
    assert torch.allclose(ym, yp * parity, atol=1e-13)


def test_zero_vector():
    y = ylm_all(3, torch.zeros(2, 3, dtype=torch.float64))
    assert torch.allclose(y[:, 0], torch.full((2,), C00, dtype=torch.float64))
    assert torch.all(y[:, 1:] == 0)


def test_scale_invariance():
    gen = torch.Generator().manual_seed(6)
    n = torch.randn(20, 3, generator=gen, dtype=torch.float64)
    assert torch.allclose(ylm_all(3, n), ylm_all(3, 7.3 * n), atol=1e-13)


def test_gradcheck_wrt_directions():
    gen = torch.Generator().manual_seed(8)
    g = torch.randn(4, 3, generator=gen, dtype=torch.float64) * 2.0
    g.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda v: ylm_all(3, v), (g,), atol=1e-9)
