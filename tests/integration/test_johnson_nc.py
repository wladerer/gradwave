"""The norm-conserving scf exposes mixing_scheme='johnson' (the QE-class mixer,
for FM metals near the Stoner instability). On a nonmagnetic metal it must reach
the same fixed point as the default pulay — same physics, different route."""
import numpy as np
import pytest

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo

FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _al(ecut_ry=24):
    upf = parse_upf(pseudo("Al_ONCV_PBE-1.2.upf"))
    cell = 4.05 / 2.0 * FCC
    return setup_system(cell, np.zeros((1, 3)), [0], [upf], ecut=ecut_ry * RY,
                        kmesh=(4, 4, 4), nbands=8, use_symmetry=True)


@pytest.mark.parametrize("scheme", ["johnson", "broyden"])
def test_nc_scheme_matches_pulay(scheme):
    kw = dict(smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
              verbose=False, max_iter=200)
    rp = scf(_al(), PBE(), mixing_scheme="pulay", **kw)
    rj = scf(_al(), PBE(), mixing_scheme=scheme, **kw)
    assert rp.converged and rj.converged
    # same stationary point to sub-µeV — different mixer, identical physics
    assert abs(float(rp.energies.total) - float(rj.energies.total)) < 1e-7


def test_nc_bad_scheme_raises():
    with pytest.raises(ValueError, match="mixing_scheme"):
        scf(_al(), PBE(), mixing_scheme="nope", verbose=False,
            max_iter=1)
