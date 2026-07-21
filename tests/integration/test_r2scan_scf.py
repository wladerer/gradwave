"""r2SCAN self-consistent SCF through the meta-GGA generalized-KS machinery.

The functional itself is pinned to libxc pointwise (tests/unit/test_r2scan.py);
these gates exercise the whole stack — τ build, v_τ operator, energy assembly,
and the input/registry wiring — end to end:

  * the r2SCAN SCF converges and opens the Si gap relative to PBE (r2SCAN is
    known to give larger semiconductor gaps than PBE);
  * `xc: r2scan` resolves through the input registries;
  * meta-GGA is guarded off on the non-collinear spinor path.
"""

import numpy as np
import pytest

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_fcc


def _system():
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    return setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=(2, 2, 2),
                        nbands=8)


def _gap(res):  # Γ HOMO–LUMO, Si has 4 occupied bands
    ev = res.eigenvalues[0]
    return float(ev[4] - ev[3])


@pytest.mark.slow
def test_r2scan_scf_converges_and_opens_gap():
    pbe = scf(_system(), PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    r2 = scf(_system(), R2SCAN(), smearing="none", etol=1e-9, rhotol=1e-8,
             verbose=False, max_iter=120)
    assert pbe.converged and r2.converged
    # r2SCAN opens the gap relative to PBE (a genuine meta-GGA effect)
    assert _gap(r2) > _gap(pbe) + 0.1
    # and the XC energy actually moved (the τ term is live)
    assert abs(float(r2.energies.xc) - float(pbe.energies.xc)) > 1e-3


@pytest.mark.slow
def test_r2scan_forces_match_finite_difference():
    """r2SCAN forces via the standard Hellmann–Feynman forces() match FD — the
    meta-GGA needs NO extra τ force term (the τ operator affects the orbitals,
    not the explicit ionic force; at the SCF stationary point the HF theorem
    holds, exactly as for GGA). The residual is real-space XC-grid egg-box:
    larger than GGA (the known SCAN grid sensitivity — τ = ½Σf|∇ψ|² is exact in
    reciprocal space, but ∫e_xc(τ)dr on the grid is not), so a converged cutoff
    is used. Measured on asus: the gap falls 3.7e-3 (20 Ry) → 8.6e-5 (45 Ry),
    i.e. it vanishes with grid density rather than plateauing at a constant.
    """
    cell, pos0 = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    pos = pos0.copy()
    pos[1, 0] += 0.08  # break symmetry so the force is nonzero

    def run(p):
        s = setup_system(cell, p, [0, 0], [upf], ecut=44 * RY, kmesh=(2, 2, 2),
                         nbands=8)
        r = scf(s, R2SCAN(), smearing="none", etol=1e-12, rhotol=1e-11,
                verbose=False, max_iter=400)
        assert r.converged
        return r

    f = forces(run(pos), remove_net=True)
    h = 1e-4
    dx = np.zeros_like(pos)
    dx[1, 0] = h
    fd = -(float(run(pos + dx).energies.free_energy)
           - float(run(pos - dx).energies.free_energy)) / (2 * h)
    # egg-box floor at this cutoff (~1e-4); a missing τ-force term would show
    # ~4e-3 (the low-cutoff gap), so 1e-3 both passes and is a real gate
    assert abs(float(f[1, 0]) - fd) < 1e-3


def test_r2scan_resolves_through_registries():
    from gradwave.api import SPIN_XC_REGISTRY, XC_REGISTRY

    assert isinstance(XC_REGISTRY["r2scan"](), R2SCAN)
    assert isinstance(SPIN_XC_REGISTRY["r2scan"](), SpinR2SCAN)


def test_metagga_rejected_on_noncollinear():
    from gradwave.core.xc.noncollinear import NoncollinearXC

    with pytest.raises(NotImplementedError):
        NoncollinearXC(SpinR2SCAN())
