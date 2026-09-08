import numpy as np
import torch

from gradwave.core.ylm import ylm_all


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


def test_gradcheck_wrt_directions():
    gen = torch.Generator().manual_seed(8)
    g = torch.randn(4, 3, generator=gen, dtype=torch.float64) * 2.0
    g.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda v: ylm_all(3, v), (g,), atol=1e-9)
