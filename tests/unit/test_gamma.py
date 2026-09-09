"""Gamma-real path gated against the complex path to machine precision.

The Gamma specialization stores the half plane-wave sphere with real
wavefunctions and runs the local term on a real FFT. Every test here compares
it against the existing complex machinery restricted to the Gamma point, so a
regression that changes the physics fails immediately.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.core.gamma import (
    GammaHamiltonian,
    build_gamma_basis,
    davidson_gamma,
    embed_real,
    full_to_half,
    half_to_full,
    metric_inner,
    unembed_real,
)
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import RDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.davidson import davidson_batched
from tests.helpers import RY

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(scope="module")
def o2_gamma():
    """O2 molecule in a box, Gamma-only, converged with an NC ONCV pseudo."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "O_ONCV_PBE-1.2.upf")
    a = 8.0
    cell = np.diag([a, a, a])
    d = 1.21
    pos = np.array([[a / 2, a / 2, a / 2 - d / 2], [a / 2, a / 2, a / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [upf], ecut=25 * RY,
                          kmesh=(1, 1, 1), nbands=8)
    res = scf(system, PBE(), smearing="gaussian", width=0.2, etol=1e-9,
              rhotol=1e-8, verbose=False, max_iter=60)
    assert res.converged
    grid, sphere, bk = system.grid, system.spheres[0], system.batch
    gb = build_gamma_basis(sphere, grid.shape)
    p_full = projectors_b(bk, system.positions)[0]
    return dict(system=system, res=res, grid=grid, sphere=sphere, bk=bk,
                gb=gb, veff=res.v_eff, p_full=p_full, dij=bk.dij_full)


def _herm_partner(sphere, shape):
    """Map each sphere index to the sphere index of -G (for Hermitian checks)."""
    miller = sphere.miller.cpu().numpy()
    box_shape = np.array(shape)
    lut = {tuple(m % box_shape): g for g, m in enumerate(miller)}
    return np.array([lut[tuple((-m) % box_shape)] for m in miller])


def test_basis_closure_and_sizes(o2_gamma):
    gb, sphere = o2_gamma["gb"], o2_gamma["sphere"]
    assert gb.nhalf == (gb.npw + 1) // 2
    assert gb.npw == sphere.npw
    # metric weight: 1 on the G=0 slot, 2 elsewhere
    assert float(gb.metric_w[0]) == 1.0
    assert torch.all(gb.metric_w[1:] == 2.0)


def test_half_full_roundtrip(o2_gamma):
    gb = o2_gamma["gb"]
    dev = gb.src_half.device
    torch.manual_seed(1)
    chalf = torch.randn(5, gb.nhalf, dtype=torch.complex128).to(dev)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)  # G=0 real
    back = full_to_half(gb, half_to_full(gb, chalf))
    assert torch.allclose(back, chalf, atol=1e-15)


def test_full_sphere_is_hermitian(o2_gamma):
    gb, sphere, grid = o2_gamma["gb"], o2_gamma["sphere"], o2_gamma["grid"]
    dev = gb.src_half.device
    torch.manual_seed(2)
    chalf = torch.randn(3, gb.nhalf, dtype=torch.complex128).to(dev)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)
    partner = _herm_partner(sphere, grid.shape)
    err = (cfull - cfull[:, partner].conj()).abs().max()
    assert float(err) < 1e-15


def test_irfftn_matches_ifftn(o2_gamma):
    """The real half-box transform reproduces the complex box transform."""
    gb, sphere, grid = o2_gamma["gb"], o2_gamma["sphere"], o2_gamma["grid"]
    shape = grid.shape
    n = shape[0] * shape[1] * shape[2]
    dev = gb.src_half.device
    torch.manual_seed(3)
    chalf = torch.randn(4, gb.nhalf, dtype=torch.complex128).to(dev)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)
    box = torch.zeros(4, n, dtype=torch.complex128, device=dev)
    box.index_add_(1, sphere.flat_idx, cfull)
    box = box.reshape(4, *shape)
    psi_c = torch.fft.ifftn(box, dim=(-3, -2, -1))
    psi_r = torch.fft.irfftn(box[..., : gb.nh3], s=shape, dim=(-3, -2, -1))
    assert float(psi_c.imag.abs().max()) < 1e-14  # Hermitian => real
    assert float((psi_r - psi_c.real).abs().max()) < 1e-14


def test_apply_matches_complex(o2_gamma):
    """H-apply equivalence: the whole point of the specialization."""
    d = o2_gamma
    gb, bk, grid = d["gb"], d["bk"], d["grid"]
    dev = gb.src_half.device
    torch.manual_seed(4)
    chalf = torch.randn(8, gb.nhalf, dtype=torch.complex128).to(dev)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)

    gh = GammaHamiltonian(gb, d["veff"], d["p_full"], d["dij"])
    hb = BatchedHamiltonian(bk, grid.shape, d["veff"], d["p_full"][None])
    out_gamma = gh.apply(chalf)
    out_full = hb.apply(cfull[None])[0]
    # the complex result must itself be Hermitian-symmetric, else the
    # half-sphere projection would lose information
    partner = _herm_partner(d["sphere"], grid.shape)
    assert float((out_full - out_full[:, partner].conj()).abs().max()) < 1e-11
    assert float((out_gamma - full_to_half(gb, out_full)).abs().max()) < 1e-11


def test_half_box_psi_matches_ifftn(o2_gamma):
    """The direct full-sphere→rfft-half-box scatter reproduces ifftn of the full
    complex box (validates full_keep_idx/half_box_flat), at half the box memory."""
    from gradwave.core.gamma import half_box_psi

    gb, sphere, grid = o2_gamma["gb"], o2_gamma["sphere"], o2_gamma["grid"]
    shape = grid.shape
    n = shape[0] * shape[1] * shape[2]
    dev = gb.src_half.device
    torch.manual_seed(6)
    chalf = torch.randn(5, gb.nhalf, dtype=torch.complex128).to(dev)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)
    box = torch.zeros(5, n, dtype=torch.complex128, device=dev)
    box.index_add_(1, sphere.flat_idx, cfull)
    psi_c = torch.fft.ifftn(box.reshape(5, *shape), dim=(-3, -2, -1))
    psi_r = half_box_psi(gb, cfull)  # ifftn-scaled, real
    assert float((psi_r - psi_c.real).abs().max()) < 1e-14


def test_density_gamma_matches_density_b(o2_gamma):
    """The real Γ density (half-box, ρ=Σ w ψ²) reproduces the complex density_b
    (complex ψ box, ψ.real²+ψ.imag²) to machine precision."""
    from gradwave.core.batch import density_b
    from gradwave.core.gamma import density_gamma

    d = o2_gamma
    gb, bk, grid = d["gb"], d["bk"], d["grid"]
    dev = gb.src_half.device
    torch.manual_seed(7)
    chalf = torch.randn(6, gb.nhalf, dtype=torch.complex128).to(dev)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)
    w = torch.rand(6, dtype=torch.float64, device=dev)
    kw = torch.ones(1, dtype=torch.float64, device=dev)
    rho_c = density_b(cfull[None], w[None], kw, bk, grid.shape, 1.0)
    rho_g = density_gamma(gb, cfull, w, 1.0)
    assert float((rho_g - rho_c).abs().max()) < 1e-11 * float(rho_c.abs().max())


def test_embed_roundtrip_and_metric(o2_gamma):
    """The real embedding is invertible and turns the metric into a dot product."""
    gb = o2_gamma["gb"]
    dev = gb.src_half.device
    torch.manual_seed(5)
    a = torch.randn(6, gb.nhalf, dtype=torch.complex128).to(dev)
    b = torch.randn(6, gb.nhalf, dtype=torch.complex128).to(dev)
    a[:, 0] = a[:, 0].real.to(torch.complex128)
    b[:, 0] = b[:, 0].real.to(torch.complex128)
    assert torch.allclose(unembed_real(gb, embed_real(gb, a)), a, atol=1e-15)
    dot = embed_real(gb, a) @ embed_real(gb, b).T
    assert torch.allclose(dot, metric_inner(gb, a, b), atol=1e-13)


def test_eigenvalues_match_complex(o2_gamma):
    """Frozen-potential eigenvalues match the complex Davidson to ~machine eps."""
    d = o2_gamma
    gb, bk, grid = d["gb"], d["bk"], d["grid"]
    nb = d["system"].nbands
    dev = gb.src_half.device
    gh = GammaHamiltonian(gb, d["veff"], d["p_full"], d["dij"])
    hb = BatchedHamiltonian(bk, grid.shape, d["veff"], d["p_full"][None])

    x0h = torch.zeros(nb, gb.nhalf, dtype=torch.complex128, device=dev)
    order = torch.argsort(gb.t_half)
    for i in range(nb):
        x0h[i, order[i]] = 1.0
    gres = davidson_gamma(gh, x0h, tol=1e-9, max_iter=80)

    x0c = torch.zeros(1, nb, bk.npw_max, dtype=torch.complex128, device=dev)
    torder = torch.argsort(bk.t[0])
    for i in range(nb):
        x0c[0, i, torder[i]] = 1.0
    cres = davidson_batched(hb.apply, x0c, bk.t, bk.mask, tol=1e-9, max_iter=80)

    assert float((gres.eigenvalues - cres.eigenvalues[0]).abs().max()) < 1e-8
    g = metric_inner(gb, gres.eigenvectors, gres.eigenvectors)
    assert float((g - torch.eye(nb, dtype=g.dtype, device=g.device)).abs().max()) < 1e-10


# --- triclinic (non-orthogonal) frozen-potential equivalence ----------------
# The O2 fixture above uses a diagonal cubic box, where the sphere is closed
# under G -> -G trivially and the FFT axes are symmetric. A general triclinic
# cell with unequal, mixed-parity FFT dimensions is where the lexicographic
# G -> -G tie-break (gamma.py rep selection) and the rfft half-box gather
# (back_flat / back_conj, keyed on n3//2+1) actually have to be right. This
# builds the geometry only (no SCF) and freezes a smooth synthetic real
# potential, so it stays light while still exercising the full H-apply path.


def _smooth_real_field(shape, seed, scale=3.0):
    """A frozen, smooth, strictly real V_eff(r) on the FFT box (shape n1,n2,n3).

    irfftn always returns a real tensor; damping the high-|G| spectrum keeps the
    field smooth so the frozen-potential Davidson converges cleanly. The apply
    equivalence itself holds for ANY real field — smoothness is only for the
    eigenvalue test's convergence."""
    n1, n2, n3 = shape
    gen = torch.Generator().manual_seed(seed)
    nh3 = n3 // 2 + 1
    spec = torch.randn(n1, n2, nh3, dtype=torch.complex128, generator=gen)
    f1 = torch.fft.fftfreq(n1)[:, None, None]
    f2 = torch.fft.fftfreq(n2)[None, :, None]
    f3 = torch.fft.rfftfreq(n3)[None, None, :]
    damp = torch.exp(-8.0 * (f1**2 + f2**2 + f3**2))
    field = torch.fft.irfftn(spec * damp, s=shape, dim=(-3, -2, -1))
    field = field - field.mean()
    field = scale * field / field.abs().max()
    return field.to(RDTYPE)


@pytest.fixture(scope="module")
def tri_gamma():
    """Single O atom in a triclinic cell, Gamma-only, geometry-only (no SCF)."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "O_ONCV_PBE-1.2.upf")
    # genuinely non-orthogonal, unequal edge lengths -> unequal FFT dims of
    # mixed parity (the rfft half-box gather boundary is n3//2+1)
    cell = np.array([[4.3, 0.0, 0.0], [1.1, 4.9, 0.0], [0.7, 1.3, 5.6]])
    pos = np.array([[0.9, 1.2, 1.5]])  # off-origin -> nontrivial projector phases
    system = setup_system(cell, pos, [0], [upf], ecut=20 * RY,
                          kmesh=(1, 1, 1), nbands=6)
    grid, sphere, bk = system.grid, system.spheres[0], system.batch
    # the FFT box must be genuinely anisotropic for this to test anything
    assert len(set(grid.shape)) > 1, f"want anisotropic box, got {grid.shape}"
    gb = build_gamma_basis(sphere, grid.shape)
    p_full = projectors_b(bk, system.positions)[0]
    veff = _smooth_real_field(grid.shape, seed=7)
    return dict(system=system, grid=grid, sphere=sphere, bk=bk, gb=gb,
                veff=veff, p_full=p_full, dij=bk.dij_full)


def test_triclinic_apply_matches_complex(tri_gamma):
    """Frozen-potential H-apply equivalence on a non-orthogonal box."""
    d = tri_gamma
    gb, bk, grid = d["gb"], d["bk"], d["grid"]
    dev = gb.src_half.device
    torch.manual_seed(11)
    chalf = torch.randn(8, gb.nhalf, dtype=torch.complex128).to(dev)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)

    gh = GammaHamiltonian(gb, d["veff"], d["p_full"], d["dij"])
    hb = BatchedHamiltonian(bk, grid.shape, d["veff"], d["p_full"][None])
    out_gamma = gh.apply(chalf)
    out_full = hb.apply(cfull[None])[0]
    partner = _herm_partner(d["sphere"], grid.shape)
    assert float((out_full - out_full[:, partner].conj()).abs().max()) < 1e-11
    assert float((out_gamma - full_to_half(gb, out_full)).abs().max()) < 1e-11


def test_triclinic_eigenvalues_match_complex(tri_gamma):
    """Frozen-potential eigenvalues match the complex Davidson on a triclinic box.

    The apply test above already proves H_gamma == H_complex exactly (<1e-11), so
    the two eigenproblems are the SAME operator and their *converged* eigenvalues
    must coincide to ~machine eps. A weak, smooth potential in a triclinic box is
    nearly free-electron, so the top of a finite block stays clustered and never
    reaches tol; comparing an unconverged Ritz value is meaningless. We solve with
    headroom (nb_solve) and compare the lowest `ncmp` bands, gating on the complex
    Davidson residual as the converged reference (the real-embedding gamma solve
    uses a different residual-norm convention, so only its eigenvalues — which do
    match the reference to <1e-8 — are compared, not its residuals)."""
    d = tri_gamma
    gb, bk, grid = d["gb"], d["bk"], d["grid"]
    ncmp = d["system"].nbands  # bands we assert equivalence on
    nb_solve = ncmp + 6  # block, with headroom above the compared window
    dev = gb.src_half.device
    gh = GammaHamiltonian(gb, d["veff"], d["p_full"], d["dij"])
    hb = BatchedHamiltonian(bk, grid.shape, d["veff"], d["p_full"][None])

    x0h = torch.zeros(nb_solve, gb.nhalf, dtype=torch.complex128, device=dev)
    order = torch.argsort(gb.t_half)
    for i in range(nb_solve):
        x0h[i, order[i]] = 1.0
    gres = davidson_gamma(gh, x0h, tol=1e-9, max_iter=120)

    x0c = torch.zeros(1, nb_solve, bk.npw_max, dtype=torch.complex128, device=dev)
    torder = torch.argsort(bk.t[0])
    for i in range(nb_solve):
        x0c[0, i, torder[i]] = 1.0
    cres = davidson_batched(hb.apply, x0c, bk.t, bk.mask, tol=1e-9, max_iter=120)

    # the reference (complex) solve must be genuinely converged on the window
    assert float(cres.residual_norms[0, :ncmp].max()) < 1e-8
    diff = (gres.eigenvalues[:ncmp] - cres.eigenvalues[0, :ncmp]).abs().max()
    assert float(diff) < 1e-8
    g = metric_inner(gb, gres.eigenvectors, gres.eigenvectors)
    eye = torch.eye(nb_solve, dtype=g.dtype, device=g.device)
    assert float((g - eye).abs().max()) < 1e-10


# --- eligibility-gate rejection guards --------------------------------------
# build_gamma_basis is only correct at a time-reversal-invariant k-point where
# the sphere is closed under G -> -G. It must refuse anything else rather than
# silently build a wrong half-sphere map. The nspin=2 / forces / USPP-on-Gamma
# gates live in the heavy-SCF wave (they need a full solve) and are not built here.


def test_gate_rejects_shifted_kpoint():
    """A shifted (non-Gamma) k-point sphere is not G -> -G closed: must raise."""
    cell = np.diag([6.0, 6.0, 6.0])
    grid = build_fft_grid(cell, 20 * RY)
    sphere = build_gsphere(grid, 20 * RY, (0.25, 0.0, 0.0))
    with pytest.raises(ValueError, match="closed under G"):
        build_gamma_basis(sphere, grid.shape)


def test_gate_rejects_non_closed_sphere():
    """Dropping one member of a {G,-G} pair from a Gamma sphere breaks closure;
    build_gamma_basis must detect the missing partner and raise."""
    cell = np.diag([6.0, 6.0, 6.0])
    grid = build_fft_grid(cell, 20 * RY)
    sphere = build_gsphere(grid, 20 * RY, (0.0, 0.0, 0.0))
    # index 0 is G=0 (kept); index 1 is the first nonzero G — dropping it leaves
    # its partner -G with no match, so the sphere is no longer G -> -G closed.
    keep = torch.ones(sphere.npw, dtype=torch.bool)
    keep[1] = False
    broken = dataclasses.replace(
        sphere,
        miller=sphere.miller[keep],
        kpg=sphere.kpg[keep],
        kpg2=sphere.kpg2[keep],
        flat_idx=sphere.flat_idx[keep],
    )
    with pytest.raises(ValueError, match="closed under G"):
        build_gamma_basis(broken, grid.shape)
