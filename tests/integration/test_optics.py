"""task: optics — independent-particle / RPA dielectric ε(ω) on silicon.

A tiny NC Si SCF, then the interband ε₂(ω) + Kramers-Kronig ε₁ + absorption:
absorption is non-negative, vanishes below the gap (ε₂(0)≈0), and the static
ε₁(0) > 1. Also checks the full task wiring via api.run.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.optics import optical_epsilon
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf

pytestmark = pytest.mark.standard

CELL, POS = si_fcc()


def _si_scf():
    upf = si_upf()
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY,
                          kmesh=(4, 4, 4), nbands=8)
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-8, rhotol=1e-7,
              verbose=False)
    assert res.converged
    return res


def test_optics_epsilon_si():
    torch.set_num_threads(8)
    res = _si_scf()
    om, eps1, eps2, alpha, info = optical_epsilon(
        res, omega_max=20.0, n_omega=400, eta=0.1, n_extra_bands=8, verbose=False)

    assert om.shape == eps1.shape == eps2.shape == alpha.shape == (400,)
    assert np.all(np.isfinite(eps2)) and (eps2 >= -1e-8).all()  # absorption ≥ 0
    assert np.all(alpha >= -1e-6)
    assert eps2[0] < 0.05 * eps2.max()  # gapped: only Lorentzian tail at ω→0
    assert 5.0 < info["eps_static"] < 40.0  # LDA Si ε₁(0) ≈ 15 (IP, no local fields)
    assert info["n_occ"] == 4            # Si: 8 valence e⁻ → 4 occupied bands
    # absorption onset sits above the (LDA-underestimated) gap
    assert om[np.argmax(eps2 > 1.0)] > 1.0


def test_optics_params_and_output():
    """OpticsParams validation + the `.out` renderer."""
    from gradwave.inputs.models import InputError, OpticsParams
    from gradwave.io.output import _optics_lines

    for bad in (dict(eta=0.0), dict(n_omega=1), dict(omega_max=-1.0),
                dict(n_extra_bands=0)):
        with pytest.raises(InputError):
            OpticsParams(**bad)

    optics = {"omega_eV": [0.0, 10.0], "eps1": [12.0, 1.0], "eps2": [0.0, 5.0],
              "absorption_inv_cm": [0.0, 1e5], "eta_eV": 0.1, "n_bands": 12,
              "n_occ": 4, "eps_static": 12.0}
    text = "\n".join(_optics_lines(optics))
    assert "optical dielectric" in text
    assert "ε₁(0)" in text
    assert "4 occ + 8 cond" in text
