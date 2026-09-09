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


def _mgga_fields():
    gen = torch.Generator().manual_seed(3)
    rho = 0.05 + 0.3 * torch.rand(6, generator=gen, dtype=torch.float64)
    mz = 0.5 * rho * (torch.rand(6, generator=gen, dtype=torch.float64) - 0.1)
    # a KE-density matrix consistent with |m| ∥ ẑ: τ↑↑, τ↓↓ per channel, τ↑↓ = 0
    tau_uu = 0.2 + 0.4 * torch.rand(6, generator=gen, dtype=torch.float64)
    tau_dd = 0.2 + 0.4 * torch.rand(6, generator=gen, dtype=torch.float64)
    return rho, mz, tau_uu, tau_dd


def test_metagga_collinear_limit_functional():
    """Non-collinear r2SCAN with every moment along ẑ reproduces the collinear
    (nspin=2) r2SCAN energy density evaluated on (ρ±, τ±) — the same functional
    two ways. τ⃗ ∥ ẑ ⇒ local_frame_tau gives (τ↑↑, τ↓↓)."""
    from gradwave.core.xc.noncollinear import local_frame_tau
    from gradwave.core.xc.r2scan import SpinR2SCAN

    rho, mz, tau_uu, tau_dd = _mgga_fields()
    nc = NoncollinearXC(SpinR2SCAN())
    assert nc.needs_tau and nc.needs_gradient
    m_vec = torch.stack([torch.zeros_like(mz), torch.zeros_like(mz), mz])
    tau_scalar = tau_uu + tau_dd
    tau_vec = torch.stack([torch.zeros_like(mz), torch.zeros_like(mz),
                           tau_uu - tau_dd])
    tau_up, tau_dn = local_frame_tau(m_vec, tau_scalar, tau_vec)
    assert torch.allclose(tau_up, tau_uu, rtol=1e-12)
    assert torch.allclose(tau_dn, tau_dd, rtol=1e-12)

    r_up = 0.5 * (rho + mz.abs())
    r_dn = 0.5 * (rho - mz.abs())
    # sigmas: locally-collinear channels along ẑ (gradients parallel here just
    # to exercise the full signature — zero is a valid consistent choice)
    z = torch.zeros_like(rho)
    e_nc = nc.energy(rho, m_vec, 1.0, z, z, z, tau_up=tau_up, tau_dn=tau_dn)
    e_col = SpinR2SCAN().energy(r_up, r_dn, 1.0, z, z, z, tau_uu, tau_dd)
    assert torch.allclose(e_nc, e_col, rtol=1e-12)


def test_tau_operator_fields_collinear():
    """tau_operator_fields with m⃗ ∥ ẑ makes v_τ⃗ ∥ ẑ with v_τz = (v↑−v↓)/2 and
    v_τ0 = (v↑+v↓)/2 — a diagonal 2×2 τ operator."""
    from gradwave.core.xc.noncollinear import tau_operator_fields

    rho, mz, _, _ = _mgga_fields()
    m_vec = torch.stack([torch.zeros_like(mz), torch.zeros_like(mz), mz])
    vu = 0.3 + 0.2 * torch.rand(6, dtype=torch.float64)
    vd = 0.1 + 0.2 * torch.rand(6, dtype=torch.float64)
    v0, vvec = tau_operator_fields(vu, vd, m_vec)
    # sign(mz) folds into m̂, so compare against v_τz·sign(mz)
    sgn = torch.sign(mz)
    assert torch.allclose(v0, 0.5 * (vu + vd), rtol=1e-12)
    assert torch.allclose(vvec[2], 0.5 * (vu - vd) * sgn, rtol=1e-12)
    assert float(vvec[0].abs().max()) < 1e-13
    assert float(vvec[1].abs().max()) < 1e-13


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


def test_noncollinear_vxc_bxc_finite_difference():
    """(v_xc, B⃗_xc) = ∂E_xc/∂(ρ, m⃗) MAGNITUDE check by central finite
    difference — not just B⃗_xc's direction.

    ``test_bxc_parallel_to_m`` only checks that B⃗_xc ∥ m⃗ (cross product ≈ 0), so
    a wrong ½ or a wrong |m|-Jacobian magnitude would slip through. Here the
    analytic potentials from ``vxc_and_bxc`` are compared, entry by entry, to the
    central difference of the total E_xc w.r.t. ρ and each Cartesian component
    m_x, m_y, m_z of a genuinely NON-collinear density (all three moments
    nonzero, varying per point). LSDA (no gradient) keeps E a pure pointwise sum,
    so the FD is exact to O(h²) with no neighbour coupling.

    ``vxc_and_bxc`` scales the raw autograd gradient ∂E/∂(field-value) by
    n_points/volume to the pointwise potential; undo that (×volume/n_points =
    the volume element dV) to compare against ΔE/Δfield directly.
    """
    from gradwave.core.xc.noncollinear import vxc_and_bxc

    class FakeGrid:
        volume = 3.0
        n_points = 5

    gen = torch.Generator().manual_seed(11)
    npts = FakeGrid.n_points
    rho = 0.2 + 0.4 * torch.rand(npts, generator=gen, dtype=torch.float64)
    # |m| kept comfortably below ρ so ρ_dn = (ρ − |m|)/2 stays positive; all
    # three components nonzero and per-point varying (a real non-collinear field)
    m_vec = 0.25 * rho * (torch.rand(3, npts, generator=gen, dtype=torch.float64) - 0.4)

    nc = NoncollinearXC(LSDA_PW92())
    vxc, bxc, _ = vxc_and_bxc(nc, rho, m_vec, FakeGrid())
    dV = FakeGrid.volume / FakeGrid.n_points
    # raw energy-derivatives ∂E/∂field = pointwise-potential × dV
    dE_drho = vxc * dV
    dE_dm = bxc * dV

    def e_of(r, m):
        return float(nc.energy(r, m, FakeGrid.volume))

    h = 1e-6
    # ∂E/∂ρ_i
    for i in range(npts):
        rp = rho.clone()
        rp[i] += h
        rm = rho.clone()
        rm[i] -= h
        fd = (e_of(rp, m_vec) - e_of(rm, m_vec)) / (2 * h)
        assert abs(fd - float(dE_drho[i])) <= 1e-6 * max(1.0, abs(fd)), (
            f"v_xc FD mismatch at i={i}: fd={fd:.10e} analytic={float(dE_drho[i]):.10e}")
    # ∂E/∂m_{a,i}
    for a in range(3):
        for i in range(npts):
            mp = m_vec.clone()
            mp[a, i] += h
            mm = m_vec.clone()
            mm[a, i] -= h
            fd = (e_of(rho, mp) - e_of(rho, mm)) / (2 * h)
            assert abs(fd - float(dE_dm[a, i])) <= 1e-6 * max(1.0, abs(fd)), (
                f"B_xc FD mismatch at ({a},{i}): fd={fd:.10e} "
                f"analytic={float(dE_dm[a, i]):.10e}")


def test_learnable_functionals_gradcheck():
    """torch.autograd.gradcheck on the learnable / spin-adapted XC potentials.

    These functionals are validated by energy LIMITS only (PBE-recovery,
    ζ→0). The SCF potential is ∂E_xc/∂ρ, so a ζ²-term sign/scaling bug would
    shift the potential while every energy-limit test still passes. gradcheck
    on ``energy_density`` w.r.t. (ρ, σ) — and, for the ζ-adaptive functionals,
    with the ζ²-enhancement ACTIVE (nonzero κ₁/μ₁, ζ derived from ρ↑,ρ↓) —
    verifies the analytic potential matches the numeric Jacobian, catching
    exactly such a bug. gradgradcheck on the cheap charge-only case as well.
    """
    from gradwave.core.xc.learnable import (
        LearnableSpinX,
        LearnableSpinXZeta,
        LearnableX,
        SpinAdaptedPBE,
    )

    gen = torch.Generator().manual_seed(7)

    def _rho(n):
        return (0.2 + 0.5 * torch.rand(n, generator=gen, dtype=torch.float64)
                ).requires_grad_(True)

    def _sig(n):
        return (0.01 + 0.05 * torch.rand(n, generator=gen, dtype=torch.float64)
                ).requires_grad_(True)

    n = 4
    # --- charge-only LearnableX: ∂e/∂(ρ, σ) ---
    xc = LearnableX()
    rho, sigma = _rho(n), _sig(n)
    assert torch.autograd.gradcheck(
        lambda r, s: xc.energy_density(r, s), (rho, sigma), eps=1e-6,
        atol=1e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(
        lambda r, s: xc.energy_density(r, s), (rho, sigma), eps=1e-6,
        atol=1e-5, rtol=1e-3)

    # --- spin LearnableSpinX (ζ-independent κ,μ): ∂e/∂(ρ↑,ρ↓,σ's) ---
    xs = LearnableSpinX()
    ru, rd = _rho(n), _rho(n)
    suu, sdd, stt = _sig(n), _sig(n), _sig(n)
    assert torch.autograd.gradcheck(
        lambda a, b, p, q, t: xs.energy_density(a, b, p, q, t),
        (ru, rd, suu, sdd, stt), eps=1e-6, atol=1e-6, rtol=1e-4)

    # --- ζ-adaptive functionals with the ζ² term ACTIVE ---
    # explicit nonzero κ₁, μ₁ (kept inside the Lieb–Oxford / positivity bounds so
    # the clamps are inactive and the map is smooth for gradcheck)
    xz = LearnableSpinXZeta(kappa1=-0.05, mu1=-0.03)
    ru, rd = _rho(n), _rho(n)
    suu, sdd, stt = _sig(n), _sig(n), _sig(n)
    assert torch.autograd.gradcheck(
        lambda a, b, p, q, t: xz.energy_density(a, b, p, q, t),
        (ru, rd, suu, sdd, stt), eps=1e-6, atol=1e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(
        lambda a, b, p, q, t: xz.energy_density(a, b, p, q, t),
        (ru, rd, suu, sdd, stt), eps=1e-6, atol=1e-5, rtol=1e-3)

    # --- the shipped preset (μ₁ = −0.0475) ---
    xp = SpinAdaptedPBE()
    ru, rd = _rho(n), _rho(n)
    suu, sdd, stt = _sig(n), _sig(n), _sig(n)
    assert torch.autograd.gradcheck(
        lambda a, b, p, q, t: xp.energy_density(a, b, p, q, t),
        (ru, rd, suu, sdd, stt), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_complex_ylm_vs_scipy():
    import numpy as np
    from scipy.special import sph_harm_y

    from gradwave.core.spinor_proj import complex_ylm

    rng = np.random.default_rng(4)
    pts = rng.normal(size=(40, 3))
    theta = np.arccos(np.clip(pts[:, 2] / np.linalg.norm(pts, axis=1), -1, 1))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    ours = complex_ylm(3, torch.as_tensor(pts, dtype=torch.float64)).numpy()
    for l in range(4):
        for m in range(-l, l + 1):
            ref = sph_harm_y(l, m, theta, phi)
            got = ours[:, l * l + l + m]
            assert np.abs(got - ref).max() < 1e-7, (l, m)
