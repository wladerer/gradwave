"""FLAPW atomic KS SCF vs the NIST LDA atomic reference eigenvalues."""

from __future__ import annotations

from gradwave.flapw import atomic_scf, log_mesh
from gradwave.flapw.atom import NIST_LDA_EV


def test_atomic_scf_beryllium():
    """Be (1s²2s²) converges to the NIST LDA eigenvalues: 1s to <0.5 eV, valence 2s to <0.3 eV."""
    r, dx = log_mesh(1e-5, 28.0, 2500)
    eigs, _ = atomic_scf("Be", r, dx)
    assert abs(eigs["1s"] - NIST_LDA_EV["Be"]["1s"]) < 0.5
    assert abs(eigs["2s"] - NIST_LDA_EV["Be"]["2s"]) < 0.3


def test_atomic_scf_neon_valence():
    """Ne valence (2s, 2p) to <0.5 eV vs NIST LDA (deep 1s core is O(dx²) mesh-limited)."""
    r, dx = log_mesh(1e-5, 28.0, 2500)
    eigs, _ = atomic_scf("Ne", r, dx)
    assert abs(eigs["2s"] - NIST_LDA_EV["Ne"]["2s"]) < 0.5
    assert abs(eigs["2p"] - NIST_LDA_EV["Ne"]["2p"]) < 0.5
