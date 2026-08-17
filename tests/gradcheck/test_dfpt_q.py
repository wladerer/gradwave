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


@pytest.mark.standard
def test_chi0_q_matches_supercell_gamma():
    """External oracle: the primitive χ₀ at q=[1/2,0,0] equals the q=0 χ₀
    (apply_chi0, independently verified) of a 2x1x1 supercell for the SAME
    physical plane-wave potential.

    A wavevector-q perturbation in the primitive cell is a Γ perturbation in the
    commensurate supercell, so δρ_prim,q folds into δρ_sc,Γ. This runs an entirely
    different code path (supercell Γ Sternheimer, no k+q, no umklapp) and so
    validates chi0_q's k↔k+q coupling and umklapp bookkeeping against ground
    truth. δV = e^{iq·r} (v=1), real supercell potential 2cos(q·r); commensurate
    FFT grids let the comparison be done directly in real space."""
    import numpy as np

    cell, pos = si_fcc()
    upf = parse_upf(_FIX / "Si_ONCV_PBE-1.2.upf")
    kw = dict(smearing="none", etol=1e-10, rhotol=1e-9, max_iter=150, verbose=False)
    ecut = 18 * RY

    # primitive: q=[1/2,0,0] needs k along a1 with a 2-point mesh (k, k+q on mesh)
    sys_p = setup_system(cell, pos, [0, 0], [upf], ecut, kmesh=(2, 2, 2),
                         nbands=8, use_symmetry=False, time_reversal=False)
    res_p = scf(sys_p, PBE(), **kw)
    n1, n2, n3 = res_p.system.grid.shape

    # 2x1x1 supercell, FFT grid forced to exactly 2x along a1 (commensurate)
    cell_sc = cell.copy()
    cell_sc[0] = 2.0 * cell[0]
    pos_sc = np.concatenate([pos, pos + cell[0]], axis=0)
    sys_s = setup_system(cell_sc, pos_sc, [0, 0, 0, 0], [upf], ecut, kmesh=(1, 2, 2),
                         nbands=16, use_symmetry=False, time_reversal=False,
                         fft_shape=(2 * n1, n2, n3))
    res_s = scf(sys_s, PBE(), **kw)
    assert res_p.converged and res_s.converged

    i1 = torch.arange(2 * n1).view(-1, 1, 1)

    # --- control: apply_chi0 is consistent across cells (factor 1) for a q=0,
    # cell-periodic potential — the supercell response is the primitive one tiled.
    # This pins the k-mesh/normalization match, isolating the q≠0 check below. ---
    torch.manual_seed(7)
    w0_p = torch.randn(res_p.system.grid.shape, dtype=torch.float64)
    w0_s = torch.cat([w0_p, w0_p], dim=0)                       # tiled, cell-periodic
    d0_p = apply_chi0(res_p, w0_p)
    d0_s = apply_chi0(res_s, w0_s)
    ctrl = float((torch.cat([d0_p, d0_p], 0) - d0_s).abs().max() / d0_s.abs().max())
    assert ctrl < 1e-3, ("q=0 cross-cell control", ctrl)

    # --- q≠0 check. The real supercell potential 2cos(q·r) = e^{iq·r}+e^{-iq·r}
    # drives the +q density response TWICE: directly (the +q perturbation) and via
    # the −q perturbation's complex conjugate. So the supercell response is
    # δρ_sc = 2 · (2 Re[e^{iq·r} chi0_q]) — the factor-2 c.c. doubling confirms
    # chi0_q is the response to a single e^{iq·r}. ---
    q = [0.5, 0.0, 0.0]
    v = torch.ones(res_p.system.grid.shape, dtype=torch.complex128)
    u = chi0_q(res_p, q, v)                     # primitive +q response, periodic part
    phase = torch.exp(1j * np.pi * i1 / n1)     # e^{iq·r}, q·a1 = pi over the supercell
    drho_from_prim = 2.0 * (phase * torch.cat([u, u], dim=0)).real

    w_sc = (2.0 * torch.cos(np.pi * i1 / n1)).expand(2 * n1, n2, n3).to(torch.float64)
    drho_sc = apply_chi0(res_s, w_sc) / 2.0     # undo the c.c. doubling

    rel = float((drho_from_prim - drho_sc).abs().max() / drho_sc.abs().max())
    assert rel < 1e-3, rel


def test_chi0_q_star_unfold_matches_full_mesh():
    """q≠0 star-unfold (Phase 3): chi0_q summed over the little-group IBZ of q and
    folded by QFieldSymmetrizer equals the full-mesh chi0_q, for a perturbation
    invariant under the little co-group of q. Fewer k-points, identical result."""
    import numpy as np

    from gradwave.core.fftbox import g_to_r_box, r_to_g
    from gradwave.postscf.dfpt_q import chi0_q_reduced
    from gradwave.symmetry import (
        QFieldSymmetrizer,
        _k_ops,
        find_spacegroup,
        little_cogroup,
    )

    res = _si_res()                       # (4,4,4), use_symmetry=False, TR=False
    grid = res.system.grid
    cell, pos = si_fcc()
    sg = find_spacegroup(cell, np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]]), [0, 0])
    q = [0.25, 0.0, 0.0]                  # ordinary small rep (no glide in its little group)

    # a perturbation invariant under the little co-group of q
    qsym = QFieldSymmetrizer(grid.shape, q, sg, cell, grid.g2, grid.dens_mask)
    torch.manual_seed(0)
    v = g_to_r_box(qsym.apply(r_to_g(torch.randn(grid.shape, dtype=torch.complex128))))

    d_full = chi0_q(res, q, v)
    d_red = chi0_q_reduced(res, q, v, sg)
    rel = float((d_full - d_red).abs().max() / d_full.abs().max())
    assert rel < 1e-6, rel

    # the reduction is real: the little group of q shrinks the k-set
    ops_t = _k_ops(little_cogroup(q, sg)[0].rotations)
    assert len(ops_t) > 1


@pytest.mark.parametrize("q", [[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
def test_screened_response_q_conjugate_symmetry(q):
    """Screened (Dyson) +q response (Phase 4, link 1): δρ_{-q}=conj(δρ_q) for a
    conjugated perturbation, and screening materially changes the bare response."""
    from gradwave.postscf.dfpt_q import chi0_q, screened_response_q

    res = _si_res()
    grid = res.system.grid
    torch.manual_seed(4)
    v = torch.randn(grid.shape, dtype=torch.complex128)
    d_q = screened_response_q(res, PBE(), q, v)
    d_mq = screened_response_q(res, PBE(), [-x for x in q], v.conj())
    rel = float((d_mq - d_q.conj()).abs().max() / d_q.abs().max())
    assert rel < 1e-5, rel
    # screening is not a no-op: the self-consistent response differs from the bare
    bare = chi0_q(res, q, v)
    assert float((d_q - bare).abs().max() / bare.abs().max()) > 1e-2
