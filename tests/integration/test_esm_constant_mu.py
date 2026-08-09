"""End-to-end: constant-potential (grand-canonical) SCF — the Level-2 feature.

At a fixed Fermi level µ (the electrode potential) the electron count N floats,
the cell is charged, and the ESM capacitor plates carry the counter-charge. Na is
a free-electron metal, so N(µ) is a smooth positive-capacitance curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo


@pytest.mark.standard
def test_constant_mu_scf_floats_charge():
    na = parse_upf(pseudo("Na_ONCV_PBE_sr.upf"))
    cell = np.diag([6.0, 6.0, 16.0])
    pos = np.array([[3.0, 3.0, 8.0]])  # a single Na (free-electron metal), between plates
    common = dict(smearing="fermi-dirac", width=0.3, etol=1e-6, rhotol=1e-5,
                  max_iter=150, verbose=False)

    def run(**kw):
        sysx = setup_system(cell, pos, [0], [na], ecut=26 * RY)
        return scf(sysx, LDA_PW92(), boundary="open_z_metal", **common, **kw)

    rf = run()  # fixed-N → the neutral Fermi level
    mu0, n0 = rf.fermi, rf.n_electrons
    r_at = run(target_mu=mu0)
    r_up = run(target_mu=mu0 + 0.5)
    r_dn = run(target_mu=mu0 - 0.5)
    assert rf.converged and r_at.converged and r_up.converged and r_dn.converged

    # at the neutral µ, constant-µ recovers the canonical electron count
    assert r_at.n_electrons == pytest.approx(n0, abs=0.02)
    # N floats monotonically with µ (positive differential capacitance / DOS>0)
    assert r_dn.n_electrons < n0 - 0.01 < n0 + 0.01 < r_up.n_electrons
    # the grand potential Ω = F − µN is well defined
    omega = float(r_up.energies.free_energy) - r_up.fermi * r_up.n_electrons
    assert np.isfinite(omega)

    # electrode-potential post-processing: a physical Na work function, U vs SHE
    # from the −4.44 V offset, and U − U_PZC that moves the right way (raising µ /
    # adding electrons makes the electrode more cathodic).
    from gradwave.postscf.work_function import electrode_potential, work_function

    phi_pzc = work_function(rf)  # neutral run = potential of zero charge
    assert 1.0 < phi_pzc < 5.0
    ep_up = electrode_potential(r_up, phi_pzc=phi_pzc)
    ep_dn = electrode_potential(r_dn, phi_pzc=phi_pzc)
    assert ep_up.potential_vs_she == pytest.approx(ep_up.work_function - 4.44, abs=1e-6)
    assert ep_up.potential_vs_pzc < 0.0 < ep_dn.potential_vs_pzc


def test_constant_mu_requires_metal_plates_and_smearing():
    """Guards: target_mu needs open_z_metal (charged cell → plates) and a smearing."""
    na = parse_upf(pseudo("Na_ONCV_PBE_sr.upf"))
    sysx = setup_system(np.diag([6.0, 6.0, 16.0]), np.array([[3.0, 3.0, 8.0]]),
                        [0], [na], ecut=20 * RY)
    with pytest.raises(ValueError, match="open_z_metal"):
        scf(sysx, LDA_PW92(), boundary="open_z", smearing="fermi-dirac",
            width=0.3, target_mu=-3.0, max_iter=1, verbose=False)
    with pytest.raises(ValueError, match="smearing"):
        scf(sysx, LDA_PW92(), boundary="open_z_metal", smearing="none",
            target_mu=-3.0, max_iter=1, verbose=False)
