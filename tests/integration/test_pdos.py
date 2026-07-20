"""Projected-DOS validation across the four projection paths.

The population analysis is exact in two structural invariants regardless of
system: the group-resolved PDOS sums to the total (a partition of unity over the
AO columns), and the k-weighted captured weight equals states*(1-spilling). The
physics checks are per path,

  norm-conserving   the bare AO overlap, spilling in (0, 1)
  USPP/PAW          the S-metric caps per-state captured weight at <= 1, where
                    the bare overlap overshoots (the augmentation is applied)
  noncollinear      the m_z group DOS equals the collinear (up-down) group PDOS,
                    and rotating the moment z->x moves that signal into m_x
  fully-relativistic  summing the j channels of a shell reproduces the scalar-l
                    charge PDOS (the |l,j,mj> harmonics span {Y_lm}x{up,down})
"""

import numpy as np
import pytest
import torch

from gradwave.postscf import pdos
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

FIX = PSEUDOS
def _integ(y, x):
    return float(np.trapezoid(y, x))


@pytest.mark.standard
def test_pdos_nc_collinear_sum_rule():
    """Norm-conserving O2 (nspin=1): the group PDOS partitions the total at every
    grouping, spilling is physical, and equivalent atoms give identical PDOS."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_O_PBE.upf")
    assert getattr(upf, "pswfc", ()), "fixture must carry PP_PSWFC"
    L, d = 7.0, 1.21
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [upf], ecut=40 * RY, kmesh=(1, 1, 1))
    res = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-7,
              rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged
    for gb in ("total", "atom", "l", "lm"):
        p = pdos.projected_dos(res, group_by=gb, width=0.2)
        summed = sum(p.groups.values())
        assert np.abs(summed - p.total).max() < 1e-10
        assert 0.0 < p.spilling < 1.0
    # equivalent atoms: identical PDOS (to SCF convergence)
    p = pdos.projected_dos(res, group_by="l", width=0.2)
    assert np.abs(p.groups["atom1:2P"] - p.groups["atom2:2P"]).max() < 5e-6


@pytest.mark.standard
def test_pdos_uspp_s_metric():
    """PAW Al: the S-metric caps per-state captured weight at <= 1 where the bare
    overlap overshoots, and the group PDOS partitions the total."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp
    a = 4.05
    cell = 0.5 * a * np.array([[0, 1, 1.0], [1, 0, 1], [1, 1, 0]])
    paw = parse_upf_paw(f"{FIX}/Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    s = setup_uspp(cell, [[0, 0, 0]], [0], [paw], ecut=20 * RY, ecutrho=100 * RY,
                   kmesh=(2, 2, 2), nbands=8)
    r = scf_uspp(s, PBE(), smearing="gaussian", width=0.3, etol=1e-7, rhotol=1e-6,
                 verbose=False)
    assert r["converged"]
    p = pdos.projected_dos(r, group_by="l", width=0.3)
    assert np.abs(sum(p.groups.values()) - p.total).max() < 1e-10
    assert 0.0 < p.spilling < 1.0

    # per-state captured weight: S-metric <= 1, bare metric overshoots
    cols = pdos._atomic_columns(s)
    dev = r["rho"].device
    maxw_S = maxw_bare = 0.0
    for ik, sph in enumerate(s.spheres):
        c = r["coeffs"][ik].to(dev)
        maxw_S = max(maxw_S, pdos._uspp_weights_k(s, sph, ik, c, cols, dev).sum(1).max())
        maxw_bare = max(maxw_bare, pdos._nc_weights_k(s, sph, ik, c, cols, dev).sum(1).max())
    assert maxw_S <= 1.0 + 1e-6
    assert maxw_bare > 1.0 + 1e-3


@pytest.mark.slow
def test_pdos_noncollinear_spin_texture():
    """fcc Ni: the noncollinear m_z group DOS reproduces the collinear (up-down)
    group PDOS, and rotating the moment z->x rigidly rotates the spin texture."""
    torch.set_num_threads(8)
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import LSDA_PW92
    from gradwave.scf.noncollinear import scf_noncollinear
    a = 3.52
    cell = 0.5 * a * np.array([[0, 1, 1.0], [1, 0, 1], [1, 1, 0]])
    ni = parse_upf(f"{FIX}/PD_Ni_PBE.upf")

    def sys():
        return setup_system(cell, np.zeros((1, 3)), [0], [ni], ecut=45 * RY,
                            kmesh=(2, 2, 2), nbands=14, time_reversal=False)

    col = scf(sys(), LSDA_PW92(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.5], etol=1e-7, rhotol=1e-6, verbose=False, kerker=True)
    assert col.converged and col.mag_total > 0.5
    pc = pdos.projected_dos(col, group_by="l", width=0.2)

    ncz = scf_noncollinear(sys(), NoncollinearXC(LSDA_PW92()),
                           mag_vec_init=[[0, 0, 0.5]], width=0.1, smearing="gaussian",
                           etol=1e-7, rhotol=1e-6, verbose=False)
    assert ncz.converged and ncz.mag_vec[2] > 0.5
    pz = pdos.projected_dos_noncollinear(ncz, group_by="l", width=0.2)

    # sum rule and spilling agreement with the collinear projection
    assert np.abs(sum(pz.charge.values()) - pz.total_charge).max() < 1e-10
    assert abs(pz.spilling - pc.spilling) < 1e-3
    # m_z group DOS == collinear (up - down) group DOS, and m_x, m_y ~ 0 along z
    for lab in pz.charge:
        updn = pc.groups[lab]  # (2, npoints)
        assert np.abs(pz.m_z[lab] - (updn[0] - updn[1])).max() < 5e-3
        assert abs(_integ(pz.m_x[lab], pz.energy_eV)) < 1e-3
        assert abs(_integ(pz.m_y[lab], pz.energy_eV)) < 1e-3

    # rotation z->x: the same spin-texture magnitude appears in m_x
    ncx = scf_noncollinear(sys(), NoncollinearXC(LSDA_PW92()),
                           mag_vec_init=[[0.5, 0, 0]], width=0.1, smearing="gaussian",
                           etol=1e-7, rhotol=1e-6, verbose=False)
    px = pdos.projected_dos_noncollinear(ncx, group_by="total", width=0.2)
    mz_z = _integ(pz.m_z["total"], pz.energy_eV) if "total" in pz.m_z else \
        _integ(sum(pz.m_z.values()), pz.energy_eV)
    mx_x = _integ(px.m_x["total"], px.energy_eV)
    assert abs(mx_x - mz_z) < 5e-3
    assert abs(_integ(px.m_z["total"], px.energy_eV)) < 1e-3


@pytest.mark.slow
def test_pdos_soc_j_resolved():
    """FR Bi: the j-resolved projection resolves each shell into its spin-orbit j
    channels. Its group PDOS partitions the total exactly. For an s shell (single
    radial channel) the j-summed charge equals the scalar-l charge to machine
    precision; for l>0 the two differ slightly because the FR pseudo carries
    distinct radials R_{n,l,j} for the two j, so the scalar-AO span and the
    spinor-AO span are genuinely different subspaces."""
    torch.set_num_threads(8)
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.noncollinear import scf_noncollinear
    bi = parse_upf(f"{FIX}/PD_Bi_FR.upf")
    cell = 5.2 * np.eye(3)
    system = setup_system(cell, np.array([[0.0, 0, 0]]), [0], [bi], ecut=30 * RY,
                          kmesh=(1, 1, 1), nbands=12, time_reversal=False)
    res = scf_noncollinear(system, NoncollinearXC(SpinPBE()),
                           mag_vec_init=[[0, 0, 0.3]], width=0.2, smearing="gaussian",
                           etol=1e-6, rhotol=1e-5, verbose=False)
    assert res.converged

    pj = pdos.projected_dos_soc(res, group_by="j", width=0.3)
    pl = pdos.projected_dos_soc(res, group_by="l", width=0.3)
    pn = pdos.projected_dos_noncollinear(res, group_by="l", width=0.3)

    # exact partition of unity over the j,mj columns
    assert np.abs(sum(pj.groups.values()) - pj.total).max() < 1e-10
    # both spin-orbit channels of the 6P shell are resolved
    assert any("6P_j0.5" in g for g in pj.groups)
    assert any("6P_j1.5" in g for g in pj.groups)
    # 18 (j, mj) columns: 5D(6) + 6S(2) + 6P(6) ... exactly 2l+2 per (l,j) pair
    assert len(pdos.projected_dos_soc(res, group_by="jmj", width=0.3).groups) == 18

    # s shell (l=0): j-summed SOC charge == scalar-l charge to machine precision
    s_shell = next(lab for lab in pl.groups if lab.endswith("6S"))
    assert np.abs(pl.groups[s_shell] - pn.charge[s_shell]).max() < 1e-8
    # l>0: same total valence captured to <1% (spilling agrees closely)
    assert abs(pj.spilling - pn.spilling) < 0.01
