"""SCF, k-point, and smearing convergence-error estimators.

Covers the pieces of the numerical error budget beyond the basis-set (Ecut)
term: the second-order SCF self-consistency error against a loose-vs-tight run,
the (E+F)/2 smearing extrapolation and its scheme guards, and the k-point
mesh extrapolation on synthetic and real data. See
postscf/convergence_error.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.convergence_error import (
    estimate_kpoint_error,
    estimate_scf_error,
    estimate_smearing_error,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _si_cell(a=5.43):
    return a / 2 * FCC, np.array([[0.0, 0, 0], [a / 4] * 3])


# --------------------------------------------------------------------------- #
#  k-point extrapolation — pure arithmetic, no SCF                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p_true", [1.5, 2.0, 3.0])
def test_kpoint_extrapolation_recovers_synthetic(p_true):
    """Fitting E(N) = E_inf + c N^-p recovers E_inf and p to machine precision."""
    e_inf, c = -100.0, 5.0
    nks = [4, 8, 16, 32]
    energies = [e_inf + c * n ** (-p_true) for n in nks]
    kp = estimate_kpoint_error(nks, energies)
    assert abs(kp["exponent"] - p_true) < 1e-6
    assert abs(kp["e_infinity_eV"] - e_inf) < 1e-8
    # the reported error of the finest mesh is exactly its residual
    assert abs(kp["error_eV"] - (energies[-1] - e_inf)) < 1e-12


def test_kpoint_extrapolation_two_point():
    """With two meshes the exponent is supplied, not fit, and E_inf follows."""
    e_inf, c, p = -50.0, 3.0, 2.0
    kp = estimate_kpoint_error([6, 12], [e_inf + c * 6 ** -p, e_inf + c * 12 ** -p],
                               exponent=p)
    assert abs(kp["e_infinity_eV"] - e_inf) < 1e-9
    assert kp["n_meshes"] == 2


def test_kpoint_extrapolation_input_guards():
    with pytest.raises(ValueError):
        estimate_kpoint_error([8], [-1.0])            # need >= 2 meshes
    with pytest.raises(ValueError):
        estimate_kpoint_error([8, 8], [-1.0, -1.1])   # duplicate mesh
    with pytest.raises(ValueError):
        estimate_kpoint_error([8, 16], [-1.0])        # length mismatch


# --------------------------------------------------------------------------- #
#  Smearing error                                                             #
# --------------------------------------------------------------------------- #


def test_smearing_error_raises_for_fixed_occupations():
    """An insulator run with smearing='none' has no smearing error."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = _si_cell()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY, kmesh=(2, 2, 2))
    res = scf(system, PBE(), smearing="none", etol=1e-8, rhotol=1e-7, verbose=False)
    with pytest.raises(ValueError):
        estimate_smearing_error(res, scheme="none")
    # even asking for a scheme name raises, since the entropy term is exactly 0
    with pytest.raises(ValueError):
        estimate_smearing_error(res, scheme="gaussian")


@pytest.mark.slow
def test_smearing_error_matches_extrapolation():
    """E0 = (E+F)/2, dsmearing = E0 - F = -sigma*S/2, and E0 sits between E and F."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = _si_cell()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY, kmesh=(4, 4, 4))
    res = scf(system, PBE(), smearing="gaussian", width=0.4, etol=1e-9,
              rhotol=1e-8, verbose=False)
    se = estimate_smearing_error(res, scheme="gaussian", width=0.4)

    # entropy term is real and negative-of-sigma*S (F below E)
    assert se.entropy_term < 0.0
    assert se.free_energy < se.kohn_sham_energy       # F = E - sigma*S < E
    # E0 is the midpoint of E and F
    assert abs(se.energy_extrapolated - 0.5 * (se.kohn_sham_energy + se.free_energy)) < 1e-9
    # dsmearing carries F to E0, and equals -sigma*S/2 = -entropy_term/2 ... /2
    assert abs(se.dsmearing - (se.energy_extrapolated - se.free_energy)) < 1e-12
    assert abs(se.dsmearing - (-0.5 * se.entropy_term)) < 1e-9
    assert se.dsmearing > 0.0                          # E0 above the reported F
    assert abs(se.half_width - 0.5 * abs(se.entropy_term)) < 1e-12


# --------------------------------------------------------------------------- #
#  SCF convergence error                                                      #
# --------------------------------------------------------------------------- #


def _si_scf(rhotol=1e-8, etol=1e-9, diago_tol=1e-9, max_iter=100):
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = _si_cell()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY, kmesh=(2, 2, 2))
    return scf(system, PBE(), smearing="none", etol=etol, rhotol=rhotol,
               diago_tol=diago_tol, max_iter=max_iter, verbose=False)


@pytest.mark.slow
def test_scf_error_predicts_loose_tight_gap():
    """The second-order SCF-error estimate at a loosely-stopped density predicts
    the reported energy's distance above the fully converged energy.

    The loose runs are stopped inside the quadratic convergence basin (all three
    gates loosened together so the orbitals are still well solved), where the
    energy error is genuinely second order and the estimate is meaningful.
    """
    torch.set_num_threads(4)
    res_tight = _si_scf()
    assert res_tight.converged
    f_tight = float(res_tight.energies.free_energy)

    # loosen etol/rhotol/diago_tol together for an in-basin early stop
    for tol in (1e-4, 1e-5, 1e-6):
        res = _si_scf(rhotol=tol, etol=tol, diago_tol=tol)
        f = float(res.energies.free_energy)
        est = estimate_scf_error(res, PBE())
        true_err = f - f_tight
        assert est.screened                     # Si insulator, nspin=1, no sym
        assert est.denergy > 0.0                 # quadratic form, always >= 0
        assert true_err > 0.0                    # reported energy above converged
        # unscreened form is an upper bound on the screened one
        assert est.denergy_unscreened >= est.denergy * (1.0 - 1e-6)
        # right order of magnitude (second-order estimate, not exact)
        assert 0.3 < est.denergy / true_err < 3.0
        # converged-energy estimate lands closer to the tight reference than the
        # raw reported energy did
        assert abs(est.energy_converged_estimate - f_tight) < true_err


@pytest.mark.slow
def test_scf_error_unscreened_fallback():
    """screened=False forces the cheap unscreened overestimate, which matches
    denergy_unscreened and stays positive."""
    torch.set_num_threads(4)
    res = _si_scf(rhotol=1e-4, etol=1e-4, diago_tol=1e-4)
    est = estimate_scf_error(res, PBE(), screened=False)
    assert not est.screened
    assert abs(est.denergy - est.denergy_unscreened) < 1e-12
    assert est.denergy > 0.0


def test_scf_error_requires_residual():
    """A result without a stored residual raises a clear error."""
    import dataclasses
    res = _si_scf(rhotol=1e-4, etol=1e-4, diago_tol=1e-4)
    stripped = dataclasses.replace(res, drho_scf=None)
    with pytest.raises(ValueError, match="no SCF residual"):
        estimate_scf_error(stripped, PBE())
