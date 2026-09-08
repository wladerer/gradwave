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

Note the direct gap is deliberately NOT anchored: gradwave's Fock operator is
ACE-compressed, which is exact on the occupied subspace (hence exact total
energy and occupied eigenvalues) but only approximate for unoccupied states, so
the ACE LUMO differs from a full-EXX LUMO by a code-dependent amount — not a
robust cross-code quantity. The total energy is.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.postscf.hybrid import hybrid_scf
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


def test_pbe0_fixture_is_gamma_only():
    """Guard the fixture convention: the anchor is Γ-only, α=0.25, and the
    QE input pins the bare-excluded singularity gradwave uses."""
    ref = json.loads((FIX / "si_pbe0_ci" / "reference.json").read_text())
    assert np.allclose(ref["k_points_tpiba"], [[0.0, 0.0, 0.0]])
    pw_in = (FIX / "si_pbe0_ci" / "pw.in").read_text()
    assert "input_dft = 'pbe0'" in pw_in
    assert "exxdiv_treatment = 'none'" in pw_in
    assert "x_gamma_extrapolation = .false." in pw_in
