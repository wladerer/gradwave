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
