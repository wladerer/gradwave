import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.grids import build_fft_grid, build_gsphere, gmax_from_ecut, good_fft_size
from gradwave.kpoints import monkhorst_pack

SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
ECUT = 300.0  # eV, small test cutoff


def test_good_fft_size():
    assert good_fft_size(1) == 1
    assert good_fft_size(11) == 12
    assert good_fft_size(13) == 14
    assert good_fft_size(97) == 98  # 2·7²


def test_sphere_count_matches_volume_estimate():
    grid = build_fft_grid(SI_CELL, ECUT)
    sph = build_gsphere(grid, ECUT, [0.0, 0.0, 0.0])
    gmax = gmax_from_ecut(ECUT)
    vol_bz = (2 * np.pi) ** 3 / grid.volume
    est = 4 / 3 * np.pi * gmax**3 / vol_bz
    assert abs(sph.npw - est) / est < 0.05
    assert torch.all(HBAR2_2M * sph.kpg2 <= ECUT * (1 + 1e-10))


def test_offgamma_sphere_recentered():
    grid = build_fft_grid(SI_CELL, ECUT)
    s0 = build_gsphere(grid, ECUT, [0.0, 0.0, 0.0])
    sk = build_gsphere(grid, ECUT, [0.25, 0.1, -0.3])
    assert abs(sk.npw - s0.npw) / s0.npw < 0.05  # similar size, different set
    assert torch.all(HBAR2_2M * sk.kpg2 <= ECUT * (1 + 1e-10))


def test_fft_round_trip():
    grid = build_fft_grid(SI_CELL, ECUT)
    sph = build_gsphere(grid, ECUT, [0.0, 0.0, 0.0])
    gen = torch.Generator().manual_seed(7)
    c = torch.complex(
        torch.randn(sph.npw, generator=gen, dtype=torch.float64),
        torch.randn(sph.npw, generator=gen, dtype=torch.float64),
    )
    f = g_to_r(c, sph.flat_idx, grid.shape)
    c2 = box_to_sphere(r_to_g(f), sph.flat_idx)
    assert torch.allclose(c2, c, atol=1e-13)


def test_density_grid_holds_wavefunction_products():
    # |ψ|² of a sphere-limited ψ has support exactly ≤ 2·G_max; if the FFT box
    # is sized right, no component beyond the density sphere may appear.
    grid = build_fft_grid(SI_CELL, ECUT)
    sph = build_gsphere(grid, ECUT, [0.0, 0.0, 0.0])
    gen = torch.Generator().manual_seed(3)
    c = torch.complex(
        torch.randn(sph.npw, generator=gen, dtype=torch.float64),
        torch.randn(sph.npw, generator=gen, dtype=torch.float64),
    )
    psi = g_to_r(c, sph.flat_idx, grid.shape)
    rho_g = r_to_g((psi.conj() * psi))
    outside = rho_g[~grid.dens_mask]
    assert outside.abs().max() < 1e-12 * rho_g.abs().max()


def test_monkhorst_pack_weights_and_tr():
    k, w = monkhorst_pack([4, 4, 4])
    assert abs(w.sum() - 1.0) < 1e-14
    kfull, wfull = monkhorst_pack([4, 4, 4], time_reversal=False)
    assert len(kfull) == 64 and len(k) < 64
    # every full-mesh point must be represented by k or -k in the reduced set
    reduced = {tuple(np.round(ki, 9)) for ki in k}
    for kf in kfull:
        pos = tuple(np.round(kf, 9))
        neg = tuple(np.round(-((-(-kf) + 0.5) % 1.0 - 0.5), 9))
        assert pos in reduced or neg in reduced


def test_monkhorst_pack_gamma_centered_contains_gamma():
    k, w = monkhorst_pack([3, 3, 3])
    assert any(np.allclose(ki, 0) for ki in k)
    # shifted 2x2x2 must NOT contain Γ
    k2, _ = monkhorst_pack([2, 2, 2], shift=(1, 1, 1))
    assert not any(np.allclose(ki, 0) for ki in k2)


def test_empty_lattice_kinetic_spectrum():
    # Free electrons: eigenvalues are exactly HBAR2_2M|k+G|² sorted — the
    # kinetic operator is diagonal in the plane-wave basis.
    grid = build_fft_grid(SI_CELL, ECUT)
    sph = build_gsphere(grid, ECUT, [0.25, 0.0, 0.0])
    t = HBAR2_2M * sph.kpg2
    bands = torch.sort(t).values[:8]
    # reference from an independent brute-force enumeration
    from gradwave.grids import reciprocal_cell

    b = reciprocal_cell(SI_CELL)
    ms = np.stack(np.meshgrid(*[np.arange(-6, 7)] * 3, indexing="ij"), -1).reshape(-1, 3)
    kpg = (ms + np.array([0.25, 0.0, 0.0])) @ b
    ref = np.sort(HBAR2_2M * np.einsum("ij,ij->i", kpg, kpg))[:8]
    assert np.allclose(bands.numpy(), ref, atol=1e-10)
