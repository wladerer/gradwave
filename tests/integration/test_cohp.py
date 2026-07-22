"""COHP validation across the collinear and spinor projection paths.

There is no Quantum ESPRESSO COHP to compare against (QE carries no COHP), so
the check is the internal sum rule the band-limited projected Hamiltonian obeys
by construction: summing COHP over EVERY atom pair including the on-site blocks
and integrating to E_F reproduces the band-structure energy sum_n f_n eps_n, up
to the plane-wave spilling. Alongside it, the physical sign is fixed — a bound
dimer (O2, Bi2) must give a bonding (negative) ICOHP on its one bond.
"""

import numpy as np
import pytest
import torch

from gradwave.postscf import cohp
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

FIX = PSEUDOS


def _band_energy_step(res, system, g_spin):
    """sum_n f_n eps_n with a step occupation at E_F, matching the step
    occupation COHP._sumrule_icohp integrates with."""
    eig = res.eigenvalues.cpu().numpy()
    kw = system.kweights.cpu().numpy()
    # collapse a possible spin axis (nspin, nk, nb) -> (nk*..., nb) contribution
    if eig.ndim == 3:
        step = (eig < float(res.fermi)).astype(float) * g_spin
        return float((kw[None, :, None] * step * eig).sum())
    step = (eig < float(res.fermi)).astype(float) * g_spin
    return float((kw[:, None] * step * eig).sum())


@pytest.mark.standard
def test_cohp_collinear_o2_sum_rule():
    """Norm-conserving O2 (nspin=1): the single O-O bond is bonding (ICOHP < 0),
    and the all-pairs COHP integrated to E_F reproduces the band energy up to the
    spilling."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_O_PBE.upf")
    L, d = 7.0, 1.21
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [upf], ecut=40 * RY, kmesh=(1, 1, 1))
    res = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-7,
              rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged

    c = cohp.cohp(res, width=0.2)
    assert c.kind == "collinear"
    assert 0.0 < c.spilling < 1.0
    # exactly the one nearest-neighbour pair is picked within rcut
    assert [p[:2] for p in c.pairs] == [(0, 1)]
    # the O-O bond is bonding: negative ICOHP
    assert c.pair_icohp["1-2"] < 0.0
    # total matches the per-pair sum, and the broadened curve is finite
    assert abs(c.total_icohp - sum(c.pair_icohp.values())) < 1e-9
    assert np.isfinite(c.total).all()

    # sum rule: all pairs incl on-site integrate to the band energy up to spilling
    band_e = _band_energy_step(res, system, g_spin=2.0)
    ratio = c._sumrule_icohp / band_e
    assert 0.90 < ratio <= 1.001, (c._sumrule_icohp, band_e)


@pytest.mark.slow
def test_cohp_soc_bi2():
    """Fully-relativistic Bi2: the j-resolved (SOC) COHP and the scalar-charge
    noncollinear COHP both give a bonding bond on the same spinor states, and both
    satisfy the band-energy sum rule."""
    torch.set_num_threads(8)
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.noncollinear import scf_noncollinear
    bi = parse_upf(f"{FIX}/PD_Bi_FR.upf")
    L, d = 9.0, 2.7
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [bi], ecut=30 * RY, kmesh=(1, 1, 1),
                          nbands=24, time_reversal=False)
    res = scf_noncollinear(system, NoncollinearXC(SpinPBE()),
                           mag_vec_init=[[0, 0, 0.1], [0, 0, 0.1]], width=0.2,
                           smearing="gaussian", etol=1e-6, rhotol=1e-5,
                           verbose=False)
    assert res.converged

    cs = cohp.cohp_soc(res, width=0.3)
    cn = cohp.cohp_noncollinear(res, width=0.3)
    assert cs.kind == "soc" and cn.kind == "noncollinear"
    assert [p[:2] for p in cs.pairs] == [(0, 1)]
    # bonding on both projection bases
    assert cs.pair_icohp["1-2"] < 0.0
    assert cn.pair_icohp["1-2"] < 0.0
    # the two AO spans differ (spinor |l,j,mj> vs scalar l x spin), so the bond
    # strengths agree only to ~10%
    assert abs(cs.pair_icohp["1-2"] - cn.pair_icohp["1-2"]) < 0.1 * abs(cn.pair_icohp["1-2"])

    band_e = _band_energy_step(res, system, g_spin=1.0)
    assert 0.95 < cs._sumrule_icohp / band_e <= 1.001
    assert 0.95 < cn._sumrule_icohp / band_e <= 1.001
    assert 0.0 < cs.spilling < 1.0


@pytest.mark.standard
def test_cohp_explicit_pairs_and_rcut():
    """Pair selection: an explicit `pairs` list overrides rcut, and a tight rcut
    that excludes the bond yields no pairs."""
    torch.set_num_threads(8)
    from gradwave.core.xc.pbe import PBE
    upf = parse_upf(f"{FIX}/PD_O_PBE.upf")
    L, d = 7.0, 1.21
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [upf], ecut=40 * RY, kmesh=(1, 1, 1))
    res = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-7,
              rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged
    # explicit pair wins regardless of order
    c = cohp.cohp(res, pairs=[(1, 0)], width=0.2)
    assert [p[:2] for p in c.pairs] == [(0, 1)]
    # rcut below the bond length selects nothing
    c0 = cohp.cohp(res, rcut=1.0, width=0.2)
    assert c0.pairs == []
    assert c0.pair_icohp == {}
