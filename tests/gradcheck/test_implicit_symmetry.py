"""IBZ-reduced implicit backward for totally-symmetric perturbations.

use_symmetry=True normally blocks the SCF backward — a perturbation breaks the
crystal group, so folded IBZ representatives no longer suffice. The exception is
a *totally symmetric* perturbation (isotropic strain / EOS, a symmetry-preserving
composition or XC-parameter change): its response χ₀ w is itself totally
symmetric, so the IBZ k-sum (weights = star multiplicities) folded by the scalar
RhoSymmetrizer reproduces the full-BZ response. These check that apply_chi0 and
solve_adjoint on the IBZ match the full (use_symmetry=False) mesh to solver
tolerance under the ``assume_totally_symmetric`` opt-in, and that a symmetric
system still raises without it (no existing caller silently changes).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.learnable import LearnableX
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.common import symmetrize_rho
from gradwave.scf.implicit import apply_chi0, density_loss_param_grads, solve_adjoint
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

pytestmark = pytest.mark.standard

_FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"


def _si(use_symmetry):
    cell, pos = si_fcc()
    upf = parse_upf(_FIX / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0, 0], [upf], ecut=25 * RY, kmesh=(4, 4, 4),
                          nbands=8, use_symmetry=use_symmetry)
    return scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
               max_iter=120, verbose=False)


def _sym_probe(res):
    torch.manual_seed(0)
    grid = res.system.grid
    return symmetrize_rho(res.system.rho_symmetrizer,
                          torch.randn(grid.shape, dtype=torch.float64), grid)


def test_symmetric_chi0_matches_full_mesh():
    res_full, res_ibz = _si(False), _si(True)
    assert len(res_ibz.system.kweights) < len(res_full.system.kweights)  # IBZ smaller
    w = _sym_probe(res_ibz)
    d_full = apply_chi0(res_full, w)
    d_ibz = apply_chi0(res_ibz, w, assume_totally_symmetric=True)
    rel = float((d_full - d_ibz).abs().max() / d_full.abs().max())
    assert rel < 1e-6, rel


def test_symmetric_adjoint_matches_full_mesh():
    res_full, res_ibz = _si(False), _si(True)
    vbar = _sym_probe(res_ibz)
    u_full = solve_adjoint(res_full, PBE(), vbar)
    u_ibz = solve_adjoint(res_ibz, PBE(), vbar, assume_totally_symmetric=True)
    rel = float((u_full - u_ibz).abs().max() / u_full.abs().max())
    assert rel < 1e-6, rel


def test_symmetric_system_raises_without_optin():
    res_ibz = _si(True)
    grid = res_ibz.system.grid
    with pytest.raises(NotImplementedError):
        apply_chi0(res_ibz, torch.zeros(grid.shape, dtype=torch.float64))


def _sym_loss(rho):
    return (rho**2).sum()          # symmetry-invariant functional of the density


def _si_learnable(use_symmetry):
    cell, pos = si_fcc()
    upf = parse_upf(_FIX / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0, 0], [upf], ecut=25 * RY, kmesh=(4, 4, 4),
                          nbands=8, use_symmetry=use_symmetry)
    xc = LearnableX(kappa=0.8, mu=0.2)
    return scf(system, xc, smearing="none", etol=1e-10, rhotol=1e-9,
               max_iter=120, verbose=False), xc


def test_symmetric_xc_param_gradient_matches_full_mesh():
    """End-to-end: dL/dθ (XC params) through the IBZ backward equals the
    full-mesh gradient for a symmetry-invariant loss."""
    res_f, xc_f = _si_learnable(False)
    res_i, xc_i = _si_learnable(True)
    _, g_full = density_loss_param_grads(res_f, xc_f, _sym_loss)
    _, g_ibz = density_loss_param_grads(res_i, xc_i, _sym_loss,
                                        assume_totally_symmetric=True)
    assert set(g_full) == set(g_ibz) and g_full
    for k in g_full:
        denom = float(g_full[k].abs().max()) or 1.0
        rel = float((g_full[k] - g_ibz[k]).abs().max()) / denom
        assert rel < 1e-5, (k, rel)


def _fe_bcc(use_symmetry):
    """Ferromagnetic bcc Fe (collinear nspin=2, smeared metal). The chemical
    point group is unbroken (uniform moment), so ``System.rho_symmetrizer`` is a
    scalar ``RhoSymmetrizer`` and each spin channel folds independently."""
    a = 2.87
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    pos = np.zeros((1, 3))
    upf = parse_upf(_FIX / "Fe_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0], [upf], ecut=45 * RY, kmesh=(4, 4, 4),
                          nbands=16, use_symmetry=use_symmetry)
    return scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
               start_mag=[2.2], etol=1e-9, rhotol=1e-8, max_iter=250, verbose=False)


def test_symmetric_chi0_nspin2_matches_full_mesh():
    """Collinear nspin=2: the per-spin χ₀ fold on the IBZ reproduces the full mesh.

    A ferromagnetic metal (bcc Fe, moment ≈2.6 μB — genuinely spin-split, δρ↑≠δρ↓)
    with the chemical point group unbroken folds each spin channel independently
    with the scalar RhoSymmetrizer, exactly as the forward collinear SCF
    symmetrizes (ρ↑, ρ↓). The totally-symmetric per-spin probe's IBZ response
    matches the use_symmetry=False full-BZ response to solver tolerance — a
    channel-confusion or unfolded bug would be O(1)."""
    res_full, res_ibz = _fe_bcc(False), _fe_bcc(True)
    assert res_full.nspin == 2 and res_ibz.nspin == 2
    assert res_ibz.mag_total > 1.0  # genuinely spin-split (ferromagnetic)
    assert len(res_ibz.system.kweights) < len(res_full.system.kweights)  # IBZ smaller
    grid = res_ibz.system.grid
    torch.manual_seed(0)
    # totally-symmetric per-spin probe field (2, *grid.shape)
    w = torch.stack([symmetrize_rho(res_ibz.system.rho_symmetrizer,
                                    torch.randn(grid.shape, dtype=torch.float64), grid)
                     for _ in range(2)])
    d_full = apply_chi0(res_full, w)
    d_ibz = apply_chi0(res_ibz, w, assume_totally_symmetric=True)
    rel = float((d_full - d_ibz).abs().max() / d_full.abs().max())
    assert rel < 1e-7, rel

    # a symmetric nspin=2 system still raises without the opt-in (safe default)
    with pytest.raises(NotImplementedError):
        apply_chi0(res_ibz, w)
