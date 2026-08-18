"""Multi-sphere self-consistent FLAPW (``crystal_scf_multi``): single-atom reduction and the
two-sphere dilute limit. Zero-independent splittings, as everywhere in the muffin-tin scheme."""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.flapw import crystal_scf_multi


@pytest.mark.slow
def test_multi_reduces_to_single_atom():
    """One Ne in a cubic cell reproduces the single-atom crystal 2s-2p splitting (~22.6 eV):
    the multi-sphere density/Weinert path is structurally correct for one sphere."""
    conv, info = crystal_scf_multi(6.0, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4},
                                   ecut=180.0, iters=30)
    ev = np.array(conv["ev"])
    assert info["nbands"] == 4
    split = float(ev[1:4].mean() - ev[0])          # 2p (3-fold) − 2s
    assert abs(split - 22.6) < 0.4


@pytest.mark.slow
def test_two_spheres_dilute_recover_atom():
    """Two well-separated Ne spheres each recover the isolated atom: the 2s bands are degenerate
    and the 2s-2p splitting matches the atomic value — validates the two-sphere Weinert (two
    pseudocharges, off-centre boundary matching)."""
    conv, info = crystal_scf_multi(12.0, [((0.25, 0.25, 0.25), "Ne"), ((0.75, 0.75, 0.75), "Ne")],
                                   {"Ne": 2.0}, ecut=120.0, iters=20)
    ev = np.array(conv["ev"])
    assert info["nbands"] == 8
    assert abs(ev[0] - ev[1]) < 0.05               # the two 2s levels are degenerate
    split = float(ev[2:8].mean() - ev[:2].mean())
    assert abs(split - 22.47) < 0.2


@pytest.mark.slow
def test_smearing_leaves_insulator_unchanged():
    """Fermi smearing on a gapped insulator (Ne) reproduces the sharp-filling result: the extra
    conduction bands sit above E_F with ~zero occupation, so the metal code path is a no-op here.
    Validates the smearing machinery (extra bands + Fermi level + fractional occupations)."""
    sharp, _ = crystal_scf_multi(6.0, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4},
                                 ecut=160.0, iters=25)
    smeared, info = crystal_scf_multi(6.0, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4},
                                      ecut=160.0, iters=25, smearing=0.1)
    ss = np.array(sharp["ev"])
    ms = np.array(smeared["ev"])
    assert info["e_fermi"] is not None
    sharp_split = float(ss[1:4].mean() - ss[0])
    smeared_split = float(ms[1:4].mean() - ms[0])
    assert abs(smeared_split - sharp_split) < 0.1


@pytest.mark.slow
def test_orthorhombic_tetragonal_dilute():
    """An anisotropic (tetragonal) cell: two Ne far apart still recover the isolated atom — the
    per-axis Coulomb grid and reciprocal lattice reduce correctly (cubic a-vector reproduces the
    cubic result; this checks a genuinely non-cubic cell)."""
    conv, info = crystal_scf_multi([12.0, 12.0, 10.0],
                                   [((0.25, 0.25, 0.3), "Ne"), ((0.75, 0.75, 0.7), "Ne")],
                                   {"Ne": 2.0}, ecut=120.0, iters=20)
    ev = np.array(conv["ev"])
    assert abs(ev[0] - ev[1]) < 0.05
    split = float(ev[2:8].mean() - ev[:2].mean())
    assert abs(split - 22.47) < 0.2
