"""FLAPW crystal band structure and self-consistency, checked with zero-independent quantities.

The muffin-tin scheme references eigenvalues to the flat interstitial potential level, which is
only weakly determined (worst in dilute cells) — so *absolute* eigenvalues wander run-to-run while
energy *differences* (splittings, bandwidths) are stable and physical. Both codes in a cross-check
therefore compare splittings, never absolute levels (see ``experiments/autoapw/elk_ne_compare.py``).
These tests follow that rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.constants import BOHR_ANG
from gradwave.flapw import atomic_scf, build_matrices_multi, crystal_scf, log_mesh, solve_geneig

# Elk 11.0.2 (all-electron FLAPW), simple-cubic Ne a=6 Bohr, LSDA (PW92): 2s-2p splitting 22.77 eV.
ELK_NE_SPLIT_EV = 22.77


def _lapw_neon_gamma_split(a_bohr, R=1.4, ecut=300.0):
    """Single-shot LAPW 2s-2p splitting at Γ for simple-cubic Ne with the atomic potential.

    One generalized eigensolve — deterministic (no SCF, no wandering interstitial zero)."""
    L = a_bohr * BOHR_ANG
    r, dx = log_mesh(1e-5, 28.0, 2500)
    at, v_at = atomic_scf("Ne", r, dx)
    v0 = float(v_at.numpy()[np.argmin(np.abs(r.numpy() - R))])
    v_mt = torch.where(r <= R, v_at - v0, torch.zeros_like(r))
    el = {0: at["2s"] - v0, 1: at["2p"] - v0, 2: -5.0 - v0}
    species = {"Ne": {"R": R, "v": v_mt, "El": el}}
    ev = solve_geneig(*build_matrices_multi([0.0, 0.0, 0.0], L, [([0.0, 0.0, 0.0], "Ne")],
                                            2, ecut, r, dx, species)[:2], 6)
    return float(ev[1:4].mean() - ev[0])       # 2p (3-fold) minus 2s


def test_lapw_neon_splitting_vs_elk():
    """The a=6 Bohr Ne 2s-2p splitting from a single LAPW solve matches Elk 11 to <0.5 eV."""
    split = _lapw_neon_gamma_split(6.0)
    assert abs(split - ELK_NE_SPLIT_EV) < 0.5


@pytest.mark.slow
def test_crystal_scf_dilute_limit_recovers_atomic_splitting():
    """Simple-cubic Ne at a=10 Bohr (dilute): the self-consistent crystal 2s-2p splitting
    converges to the isolated-atom splitting. The splitting is the zero-independent quantity;
    absolute eigenvalues reference the wandering interstitial level and are not asserted."""
    bands, atomic = crystal_scf(10.0, "Ne", R=2.0, ecut=120.0, iters=25)
    crystal_split = bands["2p"] - bands["2s"]
    atomic_split = atomic["2p"] - atomic["2s"]
    assert abs(crystal_split - atomic_split) < 0.15


@pytest.mark.slow
def test_crystal_scf_multik_bz_integration():
    """A 2x2x2 Monkhorst-Pack k-mesh (folded to 4 irreducible points by time reversal) BZ-integrates
    the density; the dilute-limit 2s-2p splitting still recovers the atom — the k-loop is wired."""
    bands, atomic = crystal_scf(10.0, "Ne", R=2.0, ecut=120.0, iters=20, kmesh=(2, 2, 2))
    crystal_split = bands["2p"] - bands["2s"]
    atomic_split = atomic["2p"] - atomic["2s"]
    assert abs(crystal_split - atomic_split) < 0.2
