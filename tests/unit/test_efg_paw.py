"""Plane-wave/PAW EFG (``postscf.efg_paw``): laptop-safe, synthetic unit tests.

Covers the three additive pieces independently — the ionic Ewald lattice sum vs a direct
real-space point-charge sum, the smooth-density reciprocal term's symmetry, and the on-site
l=2 operator's Gaunt/selection algebra — plus the tensor-observable conventions. No SCF; the
end-to-end cross-validation against FLAPW/Elk runs on asus (see the PR body)."""

from __future__ import annotations

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.postscf.efg_paw import (
    EFGOnSite,
    _efg_angular_tensor,
    _tensor_observables,
    _traceless,
    ionic_efg,
    smooth_density_efg,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _direct_pointcharge_efg(positions, charges, cell, site, nshell=8):
    """Brute-force real-space EFG at ``site`` from a neutral point-charge lattice:
    V_ab = −e² Σ'_{R≠self} Z (3 d_a d_b − δ_ab d²)/d^5, summed over an (2nshell+1)³ image block.
    Convergent (conditionally) for a charge-neutral, roughly-isotropic lattice."""
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    ns = np.arange(-nshell, nshell + 1)
    shifts = np.stack(np.meshgrid(ns, ns, ns, indexing="ij"), -1).reshape(-1, 3) @ cell
    v = np.zeros((3, 3))
    r0 = positions[site]
    for j in range(len(positions)):
        d = r0[None, :] - positions[j][None, :] - shifts  # (nimg, 3)
        r2 = np.einsum("ia,ia->i", d, d)
        alive = r2 > 1e-10
        d, r2 = d[alive], r2[alive]
        r = np.sqrt(r2)
        t = (3.0 * d[:, :, None] * d[:, None, :] - np.eye(3)[None] * r2[:, None, None])
        v += charges[j] * np.einsum("iab,i->ab", t, 1.0 / r**5)
    return _traceless(-E2 * v)


# ---------------------------------------------------------------------------
# ionic Ewald lattice part
# ---------------------------------------------------------------------------
def test_ionic_efg_matches_direct_lattice_sum_tetragonal():
    """The Ewald ionic EFG reproduces a direct real-space point-charge lattice sum for a neutral
    tetragonal CsCl-like array (±1 charges): the task's headline lattice-part validation."""
    a, c = 3.0, 4.2  # anisotropic → nonzero EFG at each site
    cell = np.diag([a, a, c])
    positions = np.array([[0.0, 0.0, 0.0], [a / 2, a / 2, c / 2]])
    charges = np.array([+1.0, -1.0])
    v = ionic_efg(positions, charges, cell)
    for site in (0, 1):
        ref = _direct_pointcharge_efg(positions, charges, cell, site, nshell=14)
        # Ewald V_zz matches the direct diagonal to ~5e-4; the direct sum's ~2e-3 off-diagonal
        # is its own conditional-convergence (cubic-block) truncation noise, not an Ewald error.
        vzz_e, eta_e = _tensor_observables(v[site])
        vzz_d, eta_d = _tensor_observables(ref)
        assert abs(vzz_e - vzz_d) < 3e-3 * abs(vzz_d), (site, vzz_e, vzz_d)
        assert abs(eta_e - eta_d) < 3e-3
        assert np.allclose(v[site], ref, atol=5e-3), (site, v[site], ref)


def test_ionic_efg_cubic_site_is_zero():
    """A cubic-symmetry point-charge site (rock-salt ±1) has no l=2 invariant → zero EFG."""
    a = 4.0
    cell = np.diag([a, a, a])
    positions = np.array([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]])
    charges = np.array([+1.0, -1.0])
    v = ionic_efg(positions, charges, cell)
    assert np.abs(v).max() < 1e-6


def test_ionic_efg_is_traceless_and_symmetric():
    rng = np.random.default_rng(3)
    cell = np.diag([3.1, 3.7, 4.3])
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 1.9, 2.1], [2.0, 0.4, 3.0]])
    charges = np.array([2.0, -1.0, -1.0])
    v = ionic_efg(positions, charges, cell)
    for site in range(3):
        assert np.allclose(v[site], v[site].T, atol=1e-10)
        assert abs(np.trace(v[site])) < 1e-9 * (np.abs(v[site]).max() + 1e-30)
    _ = rng


def test_ionic_efg_eta_independent_of_splitting():
    """η (splitting parameter) is a numerical knob: the physical tensor is η-independent."""
    cell = np.diag([3.0, 3.0, 4.2])
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 2.1]])
    charges = np.array([1.0, -1.0])
    v1 = ionic_efg(positions, charges, cell, eta=0.4)
    v2 = ionic_efg(positions, charges, cell, eta=1.1)
    assert np.allclose(v1, v2, atol=1e-6)


# ---------------------------------------------------------------------------
# smooth (reciprocal) density part
# ---------------------------------------------------------------------------
def _fft_grid(n, L):
    """A cubic FFT grid: cartesian G-vectors and |G|² for an n³ box, side L (Å)."""
    freq = np.fft.fftfreq(n, d=L / n) * 2.0 * np.pi  # Å⁻¹
    gx, gy, gz = np.meshgrid(freq, freq, freq, indexing="ij")
    g_cart = np.stack([gx, gy, gz], axis=-1)
    g2 = gx**2 + gy**2 + gz**2
    return torch.tensor(g_cart), torch.tensor(g2)


def _rgrid(n, L):
    ax = np.arange(n) * (L / n)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    return xx, yy, zz


def test_smooth_efg_spherical_density_is_zero():
    """A spherically-symmetric smooth density gives zero EFG at the box centre (and everywhere)."""
    n, L = 24, 6.0
    g_cart, g2 = _fft_grid(n, L)
    xx, yy, zz = _rgrid(n, L)
    c = L / 2

    def mi(a):
        return (a - c) - L * np.round((a - c) / L)

    r2 = mi(xx) ** 2 + mi(yy) ** 2 + mi(zz) ** 2
    rho_r = np.exp(-1.5 * r2)  # isotropic
    rho_g = torch.fft.fftn(torch.tensor(rho_r)) / n**3
    pos = torch.tensor([[c, c, c]], dtype=torch.float64)
    v = smooth_density_efg(rho_g, g_cart, g2, pos)
    assert torch.abs(v[0]).max() < 1e-9


def test_smooth_efg_prolate_density_is_axial():
    """A prolate (z-stretched) Gaussian smooth density centred at the site gives an axial EFG
    (V_xx=V_yy, η≈0) with V_zz>0 — the same sign a p_z electron density gives on-site (the
    FLAPW ρ_20>0 convention this module is locked to; prolate-along-z electron density → +V_zz)."""
    n, L = 28, 6.0
    g_cart, g2 = _fft_grid(n, L)
    xx, yy, zz = _rgrid(n, L)
    c = L / 2

    def mi(a):
        return (a - c) - L * np.round((a - c) / L)

    x, y, z = mi(xx), mi(yy), mi(zz)
    rho_r = np.exp(-(x**2 + y**2) / 0.5 - z**2 / 1.5)  # elongated along z
    rho_g = torch.fft.fftn(torch.tensor(rho_r)) / n**3
    pos = torch.tensor([[c, c, c]], dtype=torch.float64)
    v = smooth_density_efg(rho_g, g_cart, g2, pos)[0].numpy()
    assert np.allclose(v, v.T, atol=1e-10)
    assert abs(np.trace(v)) < 1e-9 * np.abs(v).max()
    assert abs(v[0, 1]) < 1e-8 and abs(v[0, 2]) < 1e-8 and abs(v[1, 2]) < 1e-8
    assert abs(v[0, 0] - v[1, 1]) < 1e-8  # V_xx = V_yy
    v_zz, eta = _tensor_observables(v)
    assert eta < 1e-4
    assert v_zz > 0.0  # prolate-along-z electron density → +V_zz (FLAPW ρ_20>0 convention)


# ---------------------------------------------------------------------------
# on-site l=2 angular / radial algebra
# ---------------------------------------------------------------------------
def test_efg_angular_tensor_traceless():
    """M^ab_IJ = ∫Y_I(3r̂_ar̂_b−δ)Y_J is traceless in (a,b): Σ_a M[a,a,I,J] = 0."""
    m = _efg_angular_tensor(2)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    assert np.abs(tr).max() < 1e-10


def test_efg_angular_tensor_s_channel_is_zero():
    """The l=0 (s) diagonal block has no l=2 coupling: M[:,:,0,0] = 0."""
    m = _efg_angular_tensor(2)
    assert np.abs(m[:, :, 0, 0]).max() < 1e-10


class _FakePAW:
    """Minimal PAWData stand-in carrying only what ``EFGOnSite.from_paw`` / ``PAWOnSite.from_paw``
    read: radial mesh, per-channel l, and r·φ AE/PS partial waves. AE=PS radials give zero on-site
    difference; setting them apart drives a nonzero AE−PS term."""

    def __init__(self, r, rab, ls, rphi_ae, rphi_ps):
        from types import SimpleNamespace

        # PAWOnSite.from_paw reads only .l on betas and .l/.rphi on the partial waves.
        self.element = "X"
        self.r = r
        self.rab = rab
        self.is_paw = True
        self.betas = tuple(SimpleNamespace(l=l) for l in ls)
        self.aewfc = tuple(SimpleNamespace(l=l, rphi=rphi_ae[i]) for i, l in enumerate(ls))
        self.pswfc = tuple(SimpleNamespace(l=l, rphi=rphi_ps[i]) for i, l in enumerate(ls))
        self.aug_cutoff_idx = len(r)
        self.n_proj = len(ls)


def _pp_channels(ls, nr=64):
    """Two-p-channel radials for a synthetic on-site test (r·φ ∝ r² e^{-r} for l=1)."""
    r = np.linspace(1e-3, 4.0, nr)
    rab = np.full_like(r, float(r[1] - r[0]))
    rphi_ae = [r ** (l + 1) * np.exp(-r) for l in ls]
    rphi_ps = [0.6 * r ** (l + 1) * np.exp(-1.4 * r) for l in ls]  # different → nonzero AE−PS
    return r, rab, rphi_ae, rphi_ps


def test_onsite_spherical_becsum_is_zero():
    """A closed-shell (m-diagonal, equal-occupancy) p becsum is spherical → zero on-site EFG."""
    ls = [1]
    r, rab, rphi_ae, rphi_ps = _pp_channels(ls)
    onsite = EFGOnSite.from_paw(_FakePAW(r, rab, ls, rphi_ae, rphi_ps))
    n = onsite.n_mexp
    assert n == 3  # one p channel → m = −1,0,1
    becsum = torch.eye(n, dtype=torch.complex128)  # equal in each m
    v = onsite.tensor(becsum).numpy()
    assert np.abs(v).max() < 1e-10


def test_onsite_pz_becsum_is_axial():
    """A single p_z occupation (becsum on the m=0 slot) gives an axial on-site EFG, traceless and
    symmetric, matching the analytic V_zz = e²·(4/5)∫R²/r dr (the FLAPW-validated normalisation)."""
    ls = [1]
    r, rab, rphi_ae, rphi_ps = _pp_channels(ls)
    onsite = EFGOnSite.from_paw(_FakePAW(r, rab, ls, rphi_ae, rphi_ps))
    n = onsite.n_mexp  # 3, ordering (m0, m+1c, m-1s) per ylm_np l²+m indexing
    # ylm_np p-order at l=1: slot0=Y_10 (p_z), slot1=Y_11c (p_x), slot2=Y_11s (p_y)
    becsum = torch.zeros(n, n, dtype=torch.complex128)
    becsum[0, 0] = 1.0  # one electron in p_z
    v = onsite.tensor(becsum).numpy()
    assert np.allclose(v, v.T, atol=1e-12)
    assert abs(np.trace(v)) < 1e-12
    assert abs(v[0, 1]) < 1e-10 and abs(v[0, 2]) < 1e-10 and abs(v[1, 2]) < 1e-10
    assert abs(v[0, 0] - v[1, 1]) < 1e-10 and abs(v[2, 2] + 2 * v[0, 0]) < 1e-10
    # analytic: R = φ = (rφ)/r; ∫R²/r dr = ∫(rφ_ae)²/r³ dr, AE−PS
    w3 = rab / r**3
    r_int = float(np.sum((rphi_ae[0] ** 2 - rphi_ps[0] ** 2) * w3))
    v_zz_analytic = E2 * (4.0 / 5.0) * r_int
    assert abs(v[2, 2] - v_zz_analytic) < 1e-9 * abs(v_zz_analytic)


# ---------------------------------------------------------------------------
# tensor observables
# ---------------------------------------------------------------------------
def test_tensor_observables_conventions():
    """V_zz is the largest-magnitude eigenvalue; η = |V_xx−V_yy|/|V_zz| ∈ [0, 1]."""
    v = np.diag([1.0, -3.0, 2.0])  # traceless
    v_zz, eta = _tensor_observables(v)
    assert abs(v_zz + 3.0) < 1e-12  # largest |eig| is -3
    assert 0.0 <= eta <= 1.0
    assert abs(eta - abs((1.0 - 2.0) / -3.0)) < 1e-12


def test_tensor_observables_axial_eta_zero():
    v = np.diag([-1.0, -1.0, 2.0])
    v_zz, eta = _tensor_observables(v)
    assert abs(v_zz - 2.0) < 1e-12
    assert eta < 1e-12


# ---------------------------------------------------------------------------
# non-PAW dataset guard (measured: experiments/autoapw/efg_eta_anion.md Front A)
# ---------------------------------------------------------------------------
def test_efg_onsite_rejects_non_paw_dataset():
    """A bare ultrasoft/GBRV dataset (is_paw=False, no AE partial waves) carries no on-site l=2
    density: EFGOnSite.from_paw must raise a clear ValueError, not a deep IndexError."""
    from types import SimpleNamespace

    import pytest

    stub = SimpleNamespace(element="O", is_paw=False, aewfc=())
    with pytest.raises(ValueError, match="needs a PAW dataset"):
        EFGOnSite.from_paw(stub)  # type: ignore[arg-type]
