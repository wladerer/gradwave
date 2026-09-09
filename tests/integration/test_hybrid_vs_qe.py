"""PBE0 hybrid total energy vs Quantum ESPRESSO at IDENTICAL parameters.

This is the first cross-code anchor for the hybrid path: gradwave's own
``hybrid_scf`` (ScaledExchangePBE + ACE Fock exchange) validated only by the
α=0→PBE reduction and ACE-vs-direct self-consistency until now. The fixture is a
2-atom diamond Si cell, Γ-only, α=0.25 (PBE0), committed under
``tests/fixtures/qe/si_pbe0_ci`` (regenerate with ``regenerate.py``; CI never
runs QE).

Matching the divergence convention is what makes the comparison meaningful.
gradwave zeros the q+G=0 cell of the Fock kernel with no Gygi–Baldereschi
correction (``postscf/coulomb_kernel.py``), so the QE input pins
``exxdiv_treatment='none'`` and ``x_gamma_extrapolation=.false.`` and a Γ-only
q-mesh (``nqx*=1``) to use the same bare-excluded singularity. Under those
identical settings the converged PBE0 *total energy* agrees to well below the
1 meV/atom SCF acceptance bar (measured 2e-4 meV/atom on asus).

The direct gap needs the exact-Fock correction to be a cross-code quantity.
gradwave's SCF Fock operator is ACE-compressed: exact on the occupied subspace
(hence exact total energy and occupied eigenvalues) but only approximate for the
unoccupied states, so the *raw* ACE LUMO differs from the full-EXX LUMO by a
code-dependent amount (measured +785 meV on this fixture). ``postscf.hybrid.
exact_fock_corrected_eigenvalues`` rebuilds the un-compressed Fock operator on
the converged orbitals and re-diagonalizes the Ritz subspace; the corrected
direct gap matches QE's PBE0 gap to a few meV. QE's own ACE (default) is accurate
for the virtuals here — its ``ace=.false.`` exact-exchange eigenvalues are
identical to the committed reference — so ``reference.json``'s gap is the physical
exact-Fock anchor both codes agree on.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.postscf.hybrid import exact_fock_corrected_eigenvalues, hybrid_scf
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.mark.standard
def test_pbe0_total_energy_vs_qe():
    torch.set_num_threads(4)
    ref = json.loads((FIX / "si_pbe0_ci" / "reference.json").read_text())
    cell, pos = si_fcc()
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    # Γ-only, ecut/fft-box pinned to QE's; α=0.25 PBE0 with the same q+G=0
    # (bare-excluded) singularity as QE's exxdiv_treatment='none'.
    system = setup_system(cell, pos, [0, 0], [upf], ecut=20 * RY, kmesh=(1, 1, 1),
                          nbands=8, fft_shape=ref.get("fft_dims"))
    res = hybrid_scf(system, alpha=0.25, smearing="none", etol=1e-9, rhotol=1e-8,
                     verbose=False, max_iter=100)
    assert res.converged, "PBE0 SCF did not converge"
    ours = float(res.energies.free_energy)
    diff_mev_atom = abs(ours - ref["etot_eV"]) / 2 * 1000
    assert diff_mev_atom < 0.5, (
        f"PBE0: {ours:.8f} vs QE {ref['etot_eV']:.8f} -> {diff_mev_atom:.4f} meV/atom"
    )
    # the Fock term is live (negative, a physical fraction of the exchange)
    assert float(res.energies.fock) < 0.0


def _direct_gap(eig, occ, occ_tol=1e-6):
    """Direct gap at the (single) k-point: min unoccupied − max occupied [eV]."""
    e = np.asarray(eig, dtype=float)
    f = np.asarray(occ, dtype=float)
    return float(e[f <= occ_tol].min() - e[f > occ_tol].max())


@pytest.mark.standard
def test_pbe0_corrected_gap_vs_qe():
    """The exact-Fock-corrected PBE0 direct gap matches QE, closing the ACE artifact.

    The raw (ACE) conduction eigenvalues are a code-dependent artifact — the SCF
    Fock operator is exact only on the occupied subspace. ``exact_fock_corrected_
    eigenvalues`` re-diagonalizes the converged Ritz subspace with the
    un-compressed Fock operator; the corrected direct gap collapses onto QE's
    (measured 3.2 meV on asus, vs the +785 meV raw-ACE offset)."""
    torch.set_num_threads(4)
    ref = json.loads((FIX / "si_pbe0_ci" / "reference.json").read_text())
    cell, pos = si_fcc()
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0, 0], [upf], ecut=20 * RY, kmesh=(1, 1, 1),
                          nbands=8, fft_shape=ref.get("fft_dims"))
    res = hybrid_scf(system, alpha=0.25, smearing="none", etol=1e-9, rhotol=1e-8,
                     verbose=False, max_iter=100)
    assert res.converged, "PBE0 SCF did not converge"

    qe_gap = _direct_gap(ref["eigenvalues_eV"][0], ref["occupations"][0])
    occ = res.occupations[0].tolist()
    raw_gap = _direct_gap(res.eigenvalues[0].tolist(), occ)      # ACE (artifact)
    eig_corr = exact_fock_corrected_eigenvalues(res, alpha=0.25)
    corr_gap = _direct_gap(eig_corr[0].tolist(), occ)           # exact-Fock

    # the raw ACE gap carries the known virtual-state artifact (~0.79 eV here)
    assert raw_gap - qe_gap > 0.3, (
        f"expected the raw ACE gap to overshoot QE; raw {raw_gap:.4f} vs QE {qe_gap:.4f}")
    # the exact-Fock correction collapses it onto the QE anchor
    assert abs(corr_gap - qe_gap) < 0.02, (
        f"corrected PBE0 gap {corr_gap:.5f} vs QE {qe_gap:.5f} eV "
        f"({(corr_gap - qe_gap) * 1000:.1f} meV; raw ACE was {raw_gap:.5f})")
    # occupied eigenvalues are provably unchanged by the correction
    occ_mask = np.asarray(occ) > 1e-6
    ace_eig = np.asarray(res.eigenvalues[0].tolist())
    corr_eig = np.asarray(eig_corr[0].tolist())
    assert np.allclose(corr_eig[occ_mask], ace_eig[occ_mask], atol=1e-6)


def test_pbe0_fixture_is_gamma_only():
    """Guard the fixture convention: the anchor is Γ-only, α=0.25, and the
    QE input pins the bare-excluded singularity gradwave uses."""
    ref = json.loads((FIX / "si_pbe0_ci" / "reference.json").read_text())
    assert np.allclose(ref["k_points_tpiba"], [[0.0, 0.0, 0.0]])
    pw_in = (FIX / "si_pbe0_ci" / "pw.in").read_text()
    assert "input_dft = 'pbe0'" in pw_in
    assert "exxdiv_treatment = 'none'" in pw_in
    assert "x_gamma_extrapolation = .false." in pw_in
