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
    alchemical_gap_gradient_per_site,
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


def test_persite_charge_conserving_cosubstitution():
    """Per-site vector-λ gap gradient on a charge-conserving CO-substitution,
    MgO -> NaCl (rocksalt): Mg(10)+O(6) = 16 = Na(9)+Cl(7), so each component is
    aliovalent but the total valence is conserved (N=16 fixed, insulating). The
    coupled d(E_gap)/dλ matches FD; the per-site Jacobian sums to it exactly; and
    the frozen estimate is materially wrong (here even the sign) -- the density
    response carries the whole gradient."""
    torch.set_num_threads(4)
    mg = parse_upf(PSEUDOS / "Mg_ONCV_PBE-1.2.upf")
    o = parse_upf(PSEUDOS / "O_ONCV_PBE-1.2.upf")
    na = parse_upf(PSEUDOS / "Na_ONCV_PBE_sr.upf")
    cl = parse_upf(PSEUDOS / "Cl_ONCV_PBE_sr.upf")
    a = 4.9
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    ecut, km = 45 * RY, (2, 2, 2)
    kw = dict(smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)

    def run(lam):
        return scf(setup_alchemical_substitution(
            cell, pos, [mg, o], [0, 1], {0: na, 1: cl}, lam, ecut=ecut, kmesh=km,
            use_symmetry=False), PBE(), **kw)

    lam = 0.5
    res = run(lam)
    assert res.converged
    assert abs(res.system.n_electrons - 16.0) < 1e-9  # valence conserved

    g = alchemical_gap_gradient(res, PBE())
    per_site = alchemical_gap_gradient_per_site(res, PBE())
    h = 0.02
    fd = (_gap(run(lam + h)) - _gap(run(lam - h))) / (2 * h)

    # coupled analytic == coupled FD
    assert abs(g.dgap - fd) < 5e-3, f"coupled dgap {g.dgap} vs FD {fd}"
    # per-site Jacobian sums to the coupled gradient exactly (linearity)
    site_sum = sum(v.dgap for v in per_site.values())
    assert abs(site_sum - g.dgap) < 1e-4, f"Σ per-site {site_sum} vs coupled {g.dgap}"
    assert set(per_site) == {0, 1}
    # frozen is materially wrong: the density response is the whole gradient
    assert abs(g.dgap - g.dgap_frozen) > 1.0


def test_aliovalent_energy_gradient_with_janak():
    """Single-site aliovalent transmutation (Si -> As, ΔZ=+1) in diamond Si: N
    follows the ionic charge (fractional, metallic), so dF/dλ carries the Janak
    term μ·dN/dλ on top of the bare Hellmann-Feynman ionic derivative. It matches
    a central FD of re-converged free energies, and the isovalent control
    (Si -> C, ΔZ=0) has the Janak term vanish."""
    torch.set_num_threads(4)
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")
    as_ = parse_upf(PSEUDOS / "As_ONCV_PBE-1.2.upf")
    cc = parse_upf(PSEUDOS / "C_ONCV_PBE-1.2.upf")
    a = 5.43
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
    ecut, km = 30 * RY, (2, 2, 2)
    kw = dict(smearing="fermi-dirac", width=0.15, etol=1e-10, rhotol=1e-9,
              max_iter=400, verbose=False)

    def check(target, dZ_expected):
        def run(lam):
            return scf(setup_alchemical_substitution(
                cell, pos, [si], [0, 0], {1: target}, lam, ecut=ecut, kmesh=km,
                use_symmetry=False), PBE(), **kw)

        lam = 0.5
        res = run(lam)
        assert res.converged
        spec = res.system.alchemical
        dN = float((spec["z_target"] - spec["z_base"]).sum())
        assert abs(dN - dZ_expected) < 1e-9
        janak = float(res.fermi) * dN

        dF = float(alchemical_energy_gradient(res, lam, xc=PBE()))
        h = 0.01
        fd = (float(run(lam + h).energies.free_energy)
              - float(run(lam - h).energies.free_energy)) / (2 * h)
        assert abs(dF - fd) < 5e-3, f"dF/dλ {dF} vs FD {fd} (ΔZ={dZ_expected})"
        return janak

    janak_alio = check(as_, 1)     # aliovalent: Janak term is real and large
    assert abs(janak_alio) > 1.0
    janak_iso = check(cc, 0)       # isovalent control: Janak vanishes
    assert abs(janak_iso) < 1e-9


def test_grand_canonical_energy_gradient():
    """Rung 4: the grand-canonical alchemical gradient dΩ/dλ (grand_canonical=True)
    drops the Janak term and equals the bare Hellmann-Feynman ionic derivative. It
    matches a central FD of the grand potential Ω = F − μN (μ held at the reference
    Fermi level), and the Legendre relation dF = dΩ + μ·dN holds. Si → As, N=8.5,
    metallic."""
    torch.set_num_threads(4)
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")
    as_ = parse_upf(PSEUDOS / "As_ONCV_PBE-1.2.upf")
    a = 5.43
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
    ecut, km = 30 * RY, (2, 2, 2)
    kw = dict(smearing="fermi-dirac", width=0.15, etol=1e-10, rhotol=1e-9,
              max_iter=400, verbose=False)

    def run(lam):
        return scf(setup_alchemical_substitution(
            cell, pos, [si], [0, 0], {1: as_}, lam, ecut=ecut, kmesh=km,
            use_symmetry=False), PBE(), **kw)

    lam = 0.5
    res = run(lam)
    assert res.converged
    mu = float(res.fermi)

    d_omega = float(alchemical_energy_gradient(res, lam, xc=PBE(),
                                               grand_canonical=True))
    d_f = float(alchemical_energy_gradient(res, lam, xc=PBE()))  # canonical

    # FD of the grand potential Ω = F − μN, μ fixed at the reference Fermi level
    h = 0.01
    def omega(r):
        return float(r.energies.free_energy) - mu * r.system.n_electrons
    fd_omega = (omega(run(lam + h)) - omega(run(lam - h))) / (2 * h)
    assert abs(d_omega - fd_omega) < 5e-3, f"dΩ/dλ {d_omega} vs FD {fd_omega}"

    # Legendre relation: dF = dΩ + μ·dN (dN=+1 for one Si→As site)
    spec = res.system.alchemical
    dN = float((spec["z_target"] - spec["z_base"]).sum())
    assert abs(d_f - (d_omega + mu * dN)) < 1e-6
    # the grand-canonical gradient really differs from the canonical one
    assert abs(d_f - d_omega) > 1.0
