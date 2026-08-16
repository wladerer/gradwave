"""Classical Heisenberg Monte Carlo: reproduces the textbook nn-bcc T_c.

The load-bearing physics: for a nearest-neighbour bcc ferromagnet the classical
Heisenberg critical temperature is k_B T_c ≈ 2.054·K, well below the mean-field
2.667·K — the ~23% fluctuation correction is exactly what turns a mean-field
Curie-temperature overestimate into an experiment-matching number.
"""

import numpy as np
import pytest

from gradwave.postscf.heisenberg_mc import (
    bcc_lattice,
    curie_temperature,
    fcc_lattice,
    heisenberg_mc,
    mean_field_tc,
)

pytestmark = pytest.mark.standard


def test_bcc_lattice_is_bipartite_z8():
    nbr, sub, pos = bcc_lattice(4)
    assert len(pos) == 2 * 4 ** 3
    assert nbr.shape[1] == 8                       # bcc nn coordination
    # every nn is on the opposite sublattice (nn graph is bipartite)
    assert all((sub[nbr[i]] != sub[i]).all() for i in range(len(pos)))


def test_heisenberg_bcc_tc_matches_textbook():
    """MC T_c/K sits at the classical bcc value ≈2.054, not the mean-field 2.667."""
    nbr, _, _ = bcc_lattice(8)
    temps = np.linspace(1.75, 2.35, 7)
    r = heisenberg_mc(nbr, 1.0, temps, n_equil=300, n_sample=500, seed=1)
    tc = curie_temperature(r["temp"], r["chi"])
    assert 1.9 < tc < 2.25, f"bcc T_c/K = {tc} (textbook 2.054, MFA 2.667)"
    # ordered below, disordered above T_c
    assert r["mag"][0] > 0.5 and r["mag"][-1] < 0.3
    # and it is genuinely below mean field
    assert tc < mean_field_tc(1.0, z=8, kb=1.0)


def test_heisenberg_fcc_tc_matches_textbook():
    """The general (non-bipartite) path: fcc nn has triangular loops, so the
    greedy colouring gives >2 colours; MC T_c/K sits near the classical fcc value
    ≈3.18 (mean-field 4.0)."""
    nbr, pos = fcc_lattice(5)
    assert nbr.shape[1] == 12 and len(pos) == 4 * 5 ** 3
    temps = np.linspace(2.9, 3.5, 7)
    r = heisenberg_mc(nbr, 1.0, temps, n_equil=300, n_sample=500, seed=2)
    tc = curie_temperature(r["temp"], r["chi"])
    assert 3.0 < tc < 3.4, f"fcc T_c/K = {tc} (textbook 3.18, MFA 4.0)"
    assert tc < mean_field_tc(1.0, z=12, kb=1.0)
