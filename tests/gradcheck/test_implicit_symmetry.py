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

import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.common import symmetrize_rho
from gradwave.scf.implicit import apply_chi0, solve_adjoint
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
