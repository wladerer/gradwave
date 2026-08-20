"""The Fermi-level solver used by the metallic (smeared) FLAPW path."""

from __future__ import annotations

import numpy as np

from gradwave.flapw.scf import _fermi_level


def test_fermi_level_conserves_charge():
    """The bisected Fermi energy reproduces the requested electron count over a weighted k-mesh."""
    ev = np.array([[-5.0, -1.0, 3.0, 4.0], [-4.8, -0.9, 3.1, 4.2]])
    w = np.array([0.5, 0.5])
    for nelec in (2, 4, 6):
        ef = _fermi_level(ev, w, nelec, 0.05)
        f = 2.0 / (1.0 + np.exp((ev - ef) / 0.05))
        assert abs(float((w[:, None] * f).sum()) - nelec) < 1e-3


def test_fermi_level_zero_temperature_limit():
    """As kT→0 the Fermi level sits in the gap above the highest filled band (sharp filling)."""
    ev = np.array([[-5.0, -1.0, 3.0, 4.0], [-4.8, -0.9, 3.1, 4.2]])
    w = np.array([0.5, 0.5])
    ef = _fermi_level(ev, w, 2, 1e-3)         # 2 electrons -> fill the lowest band only
    assert -4.8 < ef < -1.0
