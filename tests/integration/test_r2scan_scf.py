"""r2SCAN self-consistent SCF through the meta-GGA generalized-KS machinery.

The functional itself is pinned to libxc pointwise (tests/unit/test_r2scan.py);
these gates exercise the whole stack — τ build, v_τ operator, energy assembly,
and the input/registry wiring — end to end:

  * the r2SCAN SCF converges and opens the Si gap relative to PBE (r2SCAN is
    known to give larger semiconductor gaps than PBE);
  * `xc: r2scan` resolves through the input registries;
  * meta-GGA is guarded off on the non-collinear spinor path.
"""

import pytest

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN
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


def test_r2scan_resolves_through_registries():
    from gradwave.api import SPIN_XC_REGISTRY, XC_REGISTRY

    assert isinstance(XC_REGISTRY["r2scan"](), R2SCAN)
    assert isinstance(SPIN_XC_REGISTRY["r2scan"](), SpinR2SCAN)


def test_metagga_rejected_on_noncollinear():
    from gradwave.core.xc.noncollinear import NoncollinearXC

    with pytest.raises(NotImplementedError):
        NoncollinearXC(SpinR2SCAN())
