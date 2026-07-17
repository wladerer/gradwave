"""The local Thomas–Fermi preconditioner must reduce to the bare Kerker filter
when the density is uniform, become the identity in vacuum, and never disturb
the pinned G=0 charge. These three limits pin the operator's correctness
without an SCF: the screened-Poisson solve, the unit conversion, and the G=0
handling are all exercised.
"""

import torch

from gradwave.scf.local_tf import LocalTFPrecond

torch.manual_seed(0)


def _setup(q0=1.1):
    """A small box with a plane-wave-like Laplacian and a density sphere that
    includes G=0 (flat index 0) and a scattered subset of the box."""
    shape = (10, 10, 10)
    npts = 10 ** 3
    g2 = torch.rand(npts, dtype=torch.float64) * 50.0  # Å⁻², wide spectrum
    g2[0] = 0.0                                          # G=0
    mask = torch.zeros(npts, dtype=torch.bool)
    mask[torch.arange(npts)[::4]] = True
    mask[0] = True
    # a tight solve so the limit checks probe the operator, not the CG cutoff
    pc = LocalTFPrecond(g2, shape, mask, q0_max=q0, cg_iters=500, cg_tol=1e-13)
    return pc, g2, mask, shape


def test_constant_density_limit_is_kerker():
    """Uniform (dense) density → q²(r) caps at q0² everywhere → the operator is
    exactly the bare Kerker filter G²/(G²+q0²)."""
    q0 = 1.1
    pc, g2, mask, shape = _setup(q0)
    pc.set_density(torch.full(shape, 5.0, dtype=torch.float64))  # dense → capped
    assert torch.allclose(pc.q2_r, torch.full(shape, q0 ** 2, dtype=torch.float64))
    r = torch.randn(int(mask.sum()), dtype=torch.complex128)
    r[0] = 0.0
    kerker = (g2[mask] / (g2[mask] + q0 ** 2)) * r
    assert (pc(r) - kerker).abs().max() < 1e-10


def test_vacuum_limit_is_identity():
    """n(r) → 0 → q²(r) → 0 → the screened-Poisson operator is the bare
    Laplacian and P is the identity (no screening in vacuum)."""
    pc, g2, mask, shape = _setup()
    pc.set_density(torch.zeros(shape, dtype=torch.float64))
    assert float(pc.q2_r.max()) == 0.0
    r = torch.randn(int(mask.sum()), dtype=torch.complex128)
    r[0] = 0.0
    assert (pc(r) - r).abs().max() < 1e-12


def test_g0_is_preserved():
    """The pinned charge (G=0) is untouched for any density, uniform or not —
    the mixer asserts this every iteration."""
    pc, g2, mask, shape = _setup()
    n = torch.rand(shape, dtype=torch.float64) * 2.0  # arbitrary inhomogeneous
    pc.set_density(n)
    r = torch.randn(int(mask.sum()), dtype=torch.complex128)
    r[0] = 0.0
    assert pc(r)[0].abs() < 1e-14


def test_inhomogeneous_is_neither_limit():
    """A density that spans vacuum and bulk gives a preconditioner distinct
    from both the identity and the bare Kerker filter — the position-dependent
    screening is genuinely doing something. (P mixes G-components, so it is not
    bounded component-wise; the diagonal Kerker bound only holds for uniform
    density.)"""
    q0 = 1.1
    pc, g2, mask, shape = _setup(q0)
    n = torch.zeros(shape, dtype=torch.float64)
    n[:5] = 5.0  # half dense (capped screening), half vacuum (no screening)
    pc.set_density(n)
    r = torch.randn(int(mask.sum()), dtype=torch.complex128)
    r[0] = 0.0
    pr = pc(r)
    kerker = (g2[mask] / (g2[mask] + q0 ** 2)) * r
    assert torch.isfinite(pr).all()
    assert (pr - r).abs().max() > 1e-6       # not the identity (screening acts)
    assert (pr - kerker).abs().max() > 1e-6  # not the constant Kerker filter
    # low-G (long-wavelength) residual is damped relative to the identity, which
    # is the whole point of a charge-sloshing preconditioner
    low_g = g2[mask] < 1.0
    assert (pr[low_g].abs().sum() < r[low_g].abs().sum())
