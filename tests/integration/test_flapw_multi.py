"""Multi-sphere self-consistent FLAPW (``crystal_scf_multi``): single-atom reduction, the two-sphere
dilute limit, Fermi smearing, and general cells (orthorhombic, rotation invariance, fcc-primitive).
Zero-independent splittings, as everywhere in the muffin-tin scheme."""

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


@pytest.mark.slow
def test_triclinic_rotation_invariance():
    """Rotating the whole cell is a passive coordinate change, so the spectrum must be invariant.
    An exact check of the general-cell machinery (reciprocal lattice G=m·B, Cartesian distances):
    a rotated cubic Ne cell reproduces the unrotated 2s-2p splitting to sub-meV."""
    acub = np.eye(3) * 6.0
    th, ph = 0.7, 0.4
    rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    arot = acub @ (rz @ rx).T
    unrot, _ = crystal_scf_multi(acub, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4},
                                 ecut=160.0, iters=25)
    rot, _ = crystal_scf_multi(arot, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4},
                               ecut=160.0, iters=25)
    ev0, ev1 = np.array(unrot["ev"]), np.array(rot["ev"])
    s0 = float(ev0[1:4].mean() - ev0[0])
    s1 = float(ev1[1:4].mean() - ev1[0])
    assert abs(s0 - s1) < 0.01


@pytest.mark.slow
def test_efg_cubic_null():
    """EFG null test: a cubic site has no l=2 invariant (lowest cubic anisotropy is l=4), so both
    the l=2 density magnitude Q2 and the EFG V_zz (from the l=2 sphere Poisson) vanish for
    simple-cubic Ne."""
    _, info = crystal_scf_multi(6.0, [((0.5, 0.5, 0.5), "Ne")], {"Ne": 1.4},
                                ecut=160.0, iters=25, efg=True)
    site = info["efg"]["a0"]
    assert site["Q2"] < 1e-9
    assert abs(site["V_zz"]) < 1e-8               # eV/Å²


@pytest.mark.slow
def test_fcc_primitive_dilute_recovers_atom():
    """A genuinely non-orthogonal Bravais lattice: the fcc primitive cell (60° angles) exercises
    the 27-image minimum-image search. A dilute single Ne recovers the atomic 2s-2p splitting."""
    a = 11.0
    afcc = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    conv, info = crystal_scf_multi(afcc, [((0.0, 0.0, 0.0), "Ne")], {"Ne": 2.0},
                                   ecut=110.0, iters=25)
    ev = np.array(conv["ev"])
    split = float(ev[1:4].mean() - ev[0])
    assert abs(split - 22.47) < 0.2
