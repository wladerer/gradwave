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
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.convergence_error import (
    _extrapolate_energy_tail,
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


@pytest.mark.parametrize("q", [0.05, 0.3, -0.4, 0.85])
def test_scf_error_extrapolation_recovers_synthetic(q):
    """A geometric energy tail E_i = E_inf + c q^i is extrapolated to E_inf and a
    non-negative denergy, for both monotone (q>0) and oscillatory (q<0) decay."""
    e_inf, c = -100.0, 0.5
    history = [{"free_energy": e_inf + c * q ** i, "dE": abs(c * q ** i),
                "res": abs(q) ** i} for i in range(9)]
    rem, den, qf, n_tail, reliable = _extrapolate_energy_tail(history)
    assert reliable
    assert den >= 0.0
    assert abs(qf - q) < 1e-6                       # ratio recovered
    e_last = history[-1]["free_energy"]
    assert abs((e_last + rem) - e_inf) < 1e-6       # E_inf recovered
    assert abs(den - abs(e_last - e_inf)) < 1e-6    # denergy == |E_last - E_inf|


def test_scf_error_extrapolation_flags_short_and_stalled():
    """Too few points or a non-contracting tail returns reliable=False with the
    last step as a crude proxy, never a negative denergy."""
    short = [{"free_energy": e, "dE": 0.0, "res": 0.0} for e in (-1.0, -1.5)]
    rem, den, q, n_tail, reliable = _extrapolate_energy_tail(short)
    assert not reliable and den >= 0.0
    # a stalled/growing tail (|q| ~ 1) is not trusted
    stalled = [{"free_energy": -1.0 - 0.1 * i, "dE": 0.1, "res": 0.1}
               for i in range(6)]
    rem, den, q, n_tail, reliable = _extrapolate_energy_tail(stalled)
    assert not reliable and den >= 0.0


@pytest.mark.slow
def test_scf_error_predicts_converged_energy_from_history():
    """Extrapolating a truncated prefix of a converged run's energy trajectory
    recovers the fully self-consistent energy and reports a positive, correctly
    scaled distance from a loosely-stopped energy.

    One SCF: the tight run's history is sliced to mimic an early stop, so the
    exact contract is tested against real data without a second run. E_inf is the
    tight run's final free energy, and each prefix's last recorded energy is the
    "reported" energy of a run stopped there.
    """
    torch.set_num_threads(4)
    res = _si_scf()
    assert res.converged
    e_inf = float(res.energies.free_energy)
    n = len(res.history)
    assert n >= 6                                    # enough tail to slice

    saw_reliable = False
    for k in range(4, n - 1):
        prefix = res.history[:k]
        reported = float(prefix[-1]["free_energy"])
        true_err = reported - e_inf
        remaining, denergy, q, n_tail, reliable = _extrapolate_energy_tail(prefix)
        assert denergy >= 0.0                        # always non-negative
        if abs(true_err) < 1e-9:
            continue                                 # already at the floor
        if reliable:
            saw_reliable = True
            # right order of magnitude relative to the true remaining error
            assert 0.2 < denergy / abs(true_err) < 6.0
            # extrapolated E_inf beats the reported energy when the error is
            # still well above the noise floor
            if abs(true_err) > 1e-6:
                assert abs((reported + remaining) - e_inf) < abs(true_err)
    assert saw_reliable                              # the basin yields a trusted estimate


@pytest.mark.slow
def test_scf_error_response_diagnostic_is_optional():
    """With xc, the response diagnostic is populated but never the headline;
    without a stored residual it is simply absent, and the robust estimate still
    works. screened=True without a residual raises a clear error."""
    import dataclasses
    torch.set_num_threads(4)
    res = _si_scf(rhotol=1e-5, etol=1e-5, diago_tol=1e-5)

    est = estimate_scf_error(res, PBE())
    assert est.denergy >= 0.0                        # headline is the robust value
    assert est.denergy_response is not None          # diagnostic computed
    assert est.denergy_unscreened is not None

    # robust path survives a missing residual; the diagnostic drops out
    stripped = dataclasses.replace(res, drho_scf=None)
    est2 = estimate_scf_error(stripped, PBE())
    assert est2.denergy >= 0.0
    assert est2.denergy_response is None
    with pytest.raises(ValueError, match="screened=True needs"):
        estimate_scf_error(stripped, PBE(), screened=True)


def test_scf_error_requires_history():
    """A result with no recorded history cannot be extrapolated."""
    import dataclasses
    res = _si_scf(rhotol=1e-4, etol=1e-4, diago_tol=1e-4)
    stripped = dataclasses.replace(res, history=[])
    with pytest.raises(ValueError, match="no SCF history"):
        estimate_scf_error(stripped)


# --------------------------------------------------------------------------- #
#  nspin=2 metal: the response diagnostic must degrade, not crash             #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fe_magnetic_metal():
    """One converged ferromagnetic bcc-Fe result (nspin=2, gaussian smearing,
    symmetry off) — the exact setting that surfaced the nspin=2 error-estimate
    IndexError. Genuinely spin-split (|m| > 1), so the two channels carry
    different occupied-band counts and drho_scf is the TOTAL residual only."""
    torch.set_num_threads(4)
    fe = parse_upf(FIX / "Fe_ONCV_PBE-1.2.upf")
    cell = 2.87 * np.eye(3)
    pos = np.zeros((1, 3))
    system = setup_system(cell, pos, [0], [fe], ecut=22 * RY, kmesh=(2, 2, 2),
                          nbands=12, use_symmetry=False)
    res = scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.5], etol=1e-6, rhotol=1e-5, max_iter=60, verbose=False)
    assert res.converged and abs(res.mag_total) > 1.0
    return res


def test_scf_error_nspin2_metal_skips_response_diagnostic(fe_magnetic_metal):
    """Regression (nspin=2 metal): the K_Hxc response diagnostic needs the
    per-spin residual, but nspin=2 stores only the TOTAL drho_scf (a 3-D grid,
    not the stacked (2, *grid) pair apply_k_hxc consumes). It must skip the
    diagnostic and keep the trajectory headline, instead of crashing with
    ``IndexError: tuple index out of range`` inside apply_k_hxc → r_to_g."""
    res = fe_magnetic_metal
    est = estimate_scf_error(res, SpinPBE())          # SpinXC + nspin=2: no crash
    assert np.isfinite(est.denergy) and est.denergy >= 0.0   # robust headline
    assert np.isfinite(est.residual_norm)             # total-residual L1 still set
    assert est.denergy_response is None               # diagnostic gated off
    assert est.denergy_unscreened is None
    # an explicit screened request names the nspin=1 restriction, not IndexError
    with pytest.raises(ValueError, match="nspin=1 only"):
        estimate_scf_error(res, SpinPBE(), screened=True)


def test_error_estimate_block_nspin2_metal(fe_magnetic_metal, tmp_path):
    """The full post-SCF error-estimate block returns a valid dict on the
    ferromagnetic bcc-Fe metal that used to raise IndexError: a finite basis-set
    energy error, a finite density error, and an SCF-convergence sub-block with
    no (nspin=1-only) response-diagnostic field."""
    from gradwave.api.summary import _error_estimate_block
    from gradwave.inputs import load_input

    res = fe_magnetic_metal
    (tmp_path / "fe.yaml").write_text(f"""
structure:
  cell: {(2.87 * np.eye(3)).tolist()}
  positions: {{cart: [[0.0, 0.0, 0.0]]}}
  species: [Fe]
pseudopotentials:
  dir: {FIX}
  map: {{Fe: Fe_ONCV_PBE-1.2.upf}}
ecut: {22 * RY}
xc: pbe
nspin: 2
start_mag: [0.5]
symmetry: false
smearing: {{type: gaussian, width: 0.1}}
kpoints: {{mesh: [2, 2, 2]}}
""")
    inp = load_input(tmp_path / "fe.yaml")
    block = _error_estimate_block(res, inp)
    assert block["available"] is True
    assert np.isfinite(block["denergy_eV"])
    assert np.isfinite(block["drho_L1_per_electron"])
    sc = block["scf_convergence"]
    assert np.isfinite(sc["denergy_eV"]) and sc["denergy_eV"] >= 0.0
    assert "denergy_response_eV" not in sc            # nspin=2: no diagnostic
