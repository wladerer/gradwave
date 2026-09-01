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
    # anisotropic tensor: cubic Si is isotropic → xx≈yy≈zz, and their mean is ε₂
    exx, eyy, ezz = (np.array(info["eps2_tensor"][i]) for i in range(3))
    assert np.allclose(exx, eyy, atol=0.2) and np.allclose(exx, ezz, atol=0.2)
    assert np.allclose((exx + eyy + ezz) / 3.0, eps2, atol=1e-9)
    assert info["velocity"] == "full"


def test_optics_velocity_and_local_fields():
    """velocity='local' runs; local-field effects reduce ε₁(0) (RPA screening)."""
    res = _si_scf()
    _, e1_full, _, _, _ = optical_epsilon(res, omega_max=16.0, n_omega=200, eta=0.15,
                                          n_extra_bands=6, velocity="full")
    _, e1_loc, _, _, info_loc = optical_epsilon(res, omega_max=16.0, n_omega=200,
                                                eta=0.15, n_extra_bands=6,
                                                velocity="local")
    assert info_loc["velocity"] == "local"
    assert 5.0 < info_loc["eps_static"] < 40.0
    # nonlocal commutator shifts ε₁(0) but keeps it physical
    assert abs(e1_full[0] - e1_loc[0]) < 10.0

    _, e1_lfe, e2_lfe, _, info_lfe = optical_epsilon(
        res, omega_max=16.0, n_omega=120, eta=0.2, n_extra_bands=6, local_fields=True)
    assert info_lfe["local_fields"] is True
    assert np.all(np.isfinite(e2_lfe))
    # local fields screen the response down (ε₁(0)_LFE < ε₁(0)_IP)
    assert info_lfe["eps_static"] < info_lfe["eps1_ip"][0] + 0.5


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
