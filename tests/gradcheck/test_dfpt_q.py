"""q≠0 bare density response χ₀[δV_q] (Phase 2 of the little-group star-unfold).

Two oracles:
- q=Γ reduction: chi0_q at q=0 for a real field equals the q=0 bare χ₀
  (scf.implicit.apply_chi0) — the decisive check that the k+q plumbing (spheres,
  Hamiltonian, Sternheimer, density contraction) reduces correctly.
- q≠0 conjugate symmetry: δρ_{-q} = conj(δρ_q) for a conjugated perturbation, a
  fundamental property of χ₀ that exercises the genuine k↔k+q coupling and the
  umklapp bookkeeping (k+q vs k−q fold to different mesh points).
"""

from pathlib import Path

import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.dfpt_q import chi0_q
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import apply_chi0
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

pytestmark = pytest.mark.standard

_FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"


def _si_res():
    cell, pos = si_fcc()
    upf = parse_upf(_FIX / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0, 0], [upf], ecut=20 * RY, kmesh=(4, 4, 4),
                          nbands=8, use_symmetry=False, time_reversal=False)
    return scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
               max_iter=120, verbose=False)


def test_chi0_q_reduces_to_apply_chi0_at_gamma():
    """q=Γ: chi0_q returns the +q Fourier component δρ_{+q} = Σ f w ψ*δψ. The
    q=0 bare χ₀ (apply_chi0) is the full real response δρ_0 = δρ_{+0} + δρ_{-0} =
    2·Re δρ_{+q}|_{q=0} — the two conjugate terms coincide at q=0. So
    apply_chi0 == 2·Re(chi0_q), the decisive check that the k+q plumbing reduces
    correctly."""
    res = _si_res()
    torch.manual_seed(0)
    w = torch.randn(res.system.grid.shape, dtype=torch.float64)
    d_ref = apply_chi0(res, w)                       # real grid field
    d_q = chi0_q(res, [0.0, 0.0, 0.0], w.to(torch.complex128))
    # at q=0 with a real field the +q component is real (imag ~ 0)
    assert float(d_q.imag.abs().max()) < 1e-6 * float(d_q.real.abs().max())
    rel = float((2.0 * d_q.real - d_ref).abs().max() / d_ref.abs().max())
    assert rel < 1e-7, rel


@pytest.mark.parametrize("q", [[0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.25, 0.0]])
def test_chi0_q_conjugate_symmetry(q):
    """δρ_{-q} = conj(δρ_q) for a conjugated perturbation — the k↔k+q coupling
    and umklapp bookkeeping are internally consistent."""
    res = _si_res()
    grid = res.system.grid
    torch.manual_seed(1)
    v = torch.randn(grid.shape, dtype=torch.complex128)
    d_q = chi0_q(res, q, v)
    d_mq = chi0_q(res, [-x for x in q], v.conj())
    rel = float((d_mq - d_q.conj()).abs().max() / d_q.abs().max())
    assert rel < 1e-6, rel
