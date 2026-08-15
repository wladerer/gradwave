"""Alchemical composition channel: endpoint exactness and gradients.

Three things are pinned here, on small fast cells:

  * a heterogeneous substitution System reproduces the pure base cell at λ=0 and
    the pure target cell at λ=1 (the blend is not a virtual crystal — the
    endpoints are the real species);
  * the whole-cell binary alchemical energy gradient dE/dλ matches a central
    finite difference of re-converged SCF energies (Hellmann-Feynman, free by
    the envelope theorem);
  * the relaxed band-gap gradient d(E_gap)/dλ from the composition DFPT
    (alchemical_gap_gradient) matches a central finite difference of
    re-converged SCF gaps — the self-consistent density response included, not
    the frozen-orbital term (which here is a tiny fraction of the total).

isovalent pairs only (Si↔C, both 4 valence), so there is no charge-change
chemical-potential term to muddy the check.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import (
    alchemical_energy_gradient,
    alchemical_gap_gradient,
    setup_alchemical_substitution,
    setup_alchemical_system,
)
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard

SI = PSEUDOS / "Si_ONCV_PBE_sr.upf"
C = PSEUDOS / "C_ONCV_PBE_sr.upf"


def _sic_cell(a=4.36):
    """2-atom zinc-blende SiC primitive cell (Si at origin, C at the tetrahedral
    site). Insulating and isovalent throughout the Si→C substitution."""
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
    return cell, pos


def _gap(res):
    e, f = res.eigenvalues, res.occupations
    m = f > 1e-6
    return float(e[~m].min() - e[m].max())


def test_substitution_reproduces_endpoints():
    """λ=0 is the pure base cell, λ=1 the pure target cell, to SCF noise."""
    torch.set_num_threads(4)
    cell, pos = _sic_cell()
    si, c = parse_upf(SI), parse_upf(C)
    ecut, km = 30 * RY, (2, 2, 2)
    kw = dict(smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)

    # base = SiC; substitute the Si site (0) toward C  -> λ=1 is 2-C (diamond C)
    def alch(lam):
        return setup_alchemical_substitution(
            cell, pos, [si, c], [0, 1], {0: c}, lam, ecut=ecut, kmesh=km,
            use_symmetry=False)

    e_sic = scf(setup_system(cell, pos, [0, 1], [si, c], ecut, km,
                             use_symmetry=False), PBE(), **kw)
    e_cc = scf(setup_system(cell, pos, [0, 0], [c], ecut, km,
                            use_symmetry=False), PBE(), **kw)
    e_a0 = scf(alch(0.0), PBE(), **kw)
    e_a1 = scf(alch(1.0), PBE(), **kw)
    assert e_sic.converged and e_cc.converged and e_a0.converged and e_a1.converged
    assert abs(float(e_a0.energies.total) - float(e_sic.energies.total)) < 1e-6
    assert abs(float(e_a1.energies.total) - float(e_cc.energies.total)) < 1e-6


def test_binary_energy_gradient_matches_fd():
    """Whole-cell Si→C alchemical dE/dλ == central FD of re-converged energies."""
    torch.set_num_threads(4)
    cell, pos = _sic_cell(a=5.0)  # roomier so both endpoints stay insulating
    si, c = parse_upf(SI), parse_upf(C)
    ecut, km = 30 * RY, (2, 2, 2)
    kw = dict(smearing="none", etol=1e-11, rhotol=1e-10, verbose=False)

    def run(lam):
        return scf(setup_alchemical_system(cell, pos, si, c, lam, ecut, km,
                                           use_symmetry=False), PBE(), **kw)

    lam = 0.4
    res = run(lam)
    assert res.converged
    dE = float(alchemical_energy_gradient(res, lam, xc=PBE()))
    h = 0.01
    rp, rm = run(lam + h), run(lam - h)
    fd = (float(rp.energies.total) - float(rm.energies.total)) / (2 * h)
    assert abs(dE - fd) < 5e-3, f"dE/dλ {dE} vs FD {fd}"


def test_gap_gradient_dfpt_matches_fd():
    """Relaxed d(E_gap)/dλ from the composition DFPT == FD of re-converged gaps,
    and the frozen (sudden) estimate is materially different — the density
    response carries the gradient."""
    torch.set_num_threads(4)
    cell, pos = _sic_cell()
    si, c = parse_upf(SI), parse_upf(C)
    ecut, km = 30 * RY, (2, 2, 2)
    kw = dict(smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)

    def run(lam):
        return scf(setup_alchemical_substitution(
            cell, pos, [si, c], [0, 1], {0: c}, lam, ecut=ecut, kmesh=km,
            use_symmetry=False), PBE(), **kw)

    lam = 0.5
    res = run(lam)
    assert res.converged
    g = alchemical_gap_gradient(res, PBE())

    h = 0.01
    rp, rm = run(lam + h), run(lam - h)
    assert rp.converged and rm.converged
    fd_gap = (_gap(rp) - _gap(rm)) / (2 * h)

    def fd_eps(k, b):
        return (float(rp.eigenvalues[k, b]) - float(rm.eigenvalues[k, b])) / (2 * h)

    vk, vb = g.vbm_index
    ck, cb = g.cbm_index
    assert abs(g.dgap - fd_gap) < 5e-3, f"dgap {g.dgap} vs FD {fd_gap}"
    assert abs(g.dvbm - fd_eps(vk, vb)) < 5e-3
    assert abs(g.dcbm - fd_eps(ck, cb)) < 5e-3
    # the relaxed gradient is not the frozen one: the SC density response is the
    # whole point of the nonlocal DFPT build.
    assert abs(g.dgap - g.dgap_frozen) > 0.1
