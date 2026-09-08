"""r2SCAN meta-GGA total energy vs Quantum ESPRESSO at IDENTICAL parameters.

The first cross-code anchor for the meta-GGA path: gradwave's r2SCAN was pinned
only pointwise to libxc (tests/unit/test_r2scan.py) and by FD-vs-analytic /
τ-flat→PBE plumbing gates until now. QE (v7.5) evaluates r2SCAN through the same
libxc functional (IDs 497/498), so the pointwise XC energy density is shared; the
SCF machinery (τ build, the −½∇·(v_τ∇ψ) operator, grid integration) is what this
gate independently validates. Fixture: 2-atom diamond Si, Γ-only, committed under
``tests/fixtures/qe/si_r2scan_ci`` (regenerate with ``regenerate.py``).

Grid convergence is the whole story for a meta-GGA. τ = ½Σf|∇ψ|² is exact in
reciprocal space, but ∫e_xc(ρ,∇ρ,τ)dr on a real-space grid is not, and the two
codes' discretizations differ off the converged grid — measured on asus the
gradwave−QE gap is ~27 meV/atom at the 15-Ry-equivalent fft=32 box, 0.0 meV/atom
at fft=36, and QE's own r2SCAN energy is grid-noisy at the ~10 meV level (it moves
+14 meV going to fft=48 while gradwave drifts only ~2 meV). So the anchor pins the
converged fft=36 box QE used (ecut=60 Ry): there the converged r2SCAN total energy
agrees to below 1 µeV/atom. gradwave's FFT box is pinned to QE's dense grid so the
comparison is on an identical real-space mesh.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.r2scan import R2SCAN
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.mark.standard
def test_r2scan_total_energy_vs_qe():
    torch.set_num_threads(4)
    ref = json.loads((FIX / "si_r2scan_ci" / "reference.json").read_text())
    cell, pos = si_fcc()
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    # Γ-only, ecut/dense-grid pinned to QE's converged fft=36 box.
    system = setup_system(cell, pos, [0, 0], [upf], ecut=60 * RY, kmesh=(1, 1, 1),
                          nbands=8, fft_shape=ref.get("fft_dims"))
    res = scf(system, R2SCAN(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False, max_iter=200)
    assert res.converged, "r2SCAN SCF did not converge"
    ours = float(res.energies.free_energy)
    diff_mev_atom = abs(ours - ref["etot_eV"]) / 2 * 1000
    # measured <1e-3 meV/atom at the committed grid; 0.5 meV/atom leaves margin
    # for cross-machine SCF reproducibility while still gating the meta-GGA stack
    assert diff_mev_atom < 0.5, (
        f"r2SCAN: {ours:.8f} vs QE {ref['etot_eV']:.8f} -> {diff_mev_atom:.4f} meV/atom"
    )


def test_r2scan_fixture_is_converged_grid():
    """Guard the fixture convention: r2scan, Γ-only, on the converged fft=36 box
    (coarser boxes carry the known meta-GGA grid noise)."""
    ref = json.loads((FIX / "si_r2scan_ci" / "reference.json").read_text())
    assert np.allclose(ref["k_points_tpiba"], [[0.0, 0.0, 0.0]])
    assert ref["fft_dims"] == [36, 36, 36]
    assert "input_dft = 'r2scan'" in (FIX / "si_r2scan_ci" / "pw.in").read_text()
