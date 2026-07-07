"""Non-collinear XC: collinear limit and rotational invariance."""

import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92


def fields():
    gen = torch.Generator().manual_seed(2)
    rho = 0.05 + 0.3 * torch.rand(6, generator=gen, dtype=torch.float64)
    mz = 0.6 * rho * (torch.rand(6, generator=gen, dtype=torch.float64) - 0.2)
    return rho, mz


def test_collinear_limit():
    rho, mz = fields()
    nc = NoncollinearXC(LSDA_PW92())
    m_vec = torch.stack([torch.zeros_like(mz), torch.zeros_like(mz), mz])
    e_nc = nc.energy(rho, m_vec, volume=1.0)
    e_col = LSDA_PW92().energy(0.5 * (rho + mz.abs()), 0.5 * (rho - mz.abs()), volume=1.0)
    assert torch.allclose(e_nc, e_col, rtol=1e-10)


def test_rotational_invariance():
    # E depends only on |m| — any global rotation leaves it invariant
    rho, mz = fields()
    nc = NoncollinearXC(LSDA_PW92())
    m_z = torch.stack([torch.zeros_like(mz), torch.zeros_like(mz), mz])
    theta = 0.7
    m_rot = torch.stack([mz * torch.sin(torch.tensor(theta)),
                         torch.zeros_like(mz),
                         mz * torch.cos(torch.tensor(theta))])
    e1 = nc.energy(rho, m_z, volume=1.0)
    e2 = nc.energy(rho, m_rot, volume=1.0)
    assert torch.allclose(e1, e2, rtol=1e-12)


def test_bxc_parallel_to_m():
    from gradwave.core.xc.noncollinear import vxc_and_bxc

    class FakeGrid:
        volume = 1.0
        n_points = 6

    rho, mz = fields()
    m_vec = torch.stack([0.3 * mz, -0.2 * mz, 0.9 * mz])
    _, bxc, _ = vxc_and_bxc(NoncollinearXC(LSDA_PW92()), rho, m_vec, FakeGrid())
    # B_xc ∥ m pointwise (locally collinear): cross product vanishes
    cross = torch.linalg.cross(bxc.T, m_vec.T)
    assert float(cross.abs().max()) < 1e-10 * float(bxc.abs().max())
