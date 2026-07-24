"""Bader partitioning: synthetic-density unit checks + a real Si SCF cross-check.

The synthetic tests build a density of well-separated Gaussians with known
integrals, so the exact per-atom electron counts and charge signs are known and
the on-grid partition can be held to a tight tolerance without running an SCF.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gradwave.postscf.bader import bader
from tests.helpers import RY, si_upf

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


def _grid_frac(shape):
    axes = [np.arange(n) / n for n in shape]
    fi, fj, fk = np.meshgrid(*axes, indexing="ij")
    return np.stack([fi, fj, fk], axis=-1)  # (n1,n2,n3,3)


def _gaussian_density(cell, shape, atom_frac, electrons, sigma):
    """Sum of periodic (min-image) Gaussians, each normalised to `electrons[a]`."""
    cell = np.asarray(cell, float)
    volume = abs(np.linalg.det(cell))
    voxel = volume / (shape[0] * shape[1] * shape[2])
    gfrac = _grid_frac(shape)
    rho = np.zeros(shape, dtype=np.float64)
    for af, ne in zip(atom_frac, electrons, strict=True):
        dfrac = gfrac - np.asarray(af)
        dfrac -= np.round(dfrac)  # minimum image
        dcart = dfrac @ cell
        r2 = np.einsum("...i,...i->...", dcart, dcart)
        g = np.exp(-r2 / (2 * sigma**2))
        g *= ne / (g.sum() * voxel)  # normalise discretely to exactly `ne`
        rho += g
    return rho


def _fake_result(cell, shape, atom_frac, valence, rho, rho_spin=None):
    cell = np.asarray(cell, float)
    pos = np.asarray(atom_frac, float) @ cell
    grid = SimpleNamespace(
        cell=cell, shape=tuple(shape), volume=float(abs(np.linalg.det(cell)))
    )
    system = SimpleNamespace(
        grid=grid,
        positions=torch.tensor(pos, dtype=torch.float64),
        charges=torch.tensor(valence, dtype=torch.float64),
        species_of_atom=list(range(len(valence))),
        rho_core=None,
    )
    return SimpleNamespace(
        rho=torch.tensor(rho, dtype=torch.float64),
        rho_spin=rho_spin,
        system=system,
    )


def test_two_gaussians_charge_and_assignment():
    cell = np.diag([6.0, 6.0, 6.0])
    shape = (48, 48, 48)
    frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
    ne = [6.0, 8.0]  # asymmetric occupation
    rho = _gaussian_density(cell, shape, frac, ne, sigma=0.4)
    res = _fake_result(cell, shape, frac, valence=[7.0, 7.0], rho=rho)

    out = bader(res)

    # exactly two attractors, one bound to each nucleus
    assert out.n_attractors == 2
    assert sorted(out.attractor_atom.tolist()) == [0, 1]
    # electrons recovered to well under 1% (only Gaussian-tail leakage)
    assert out.electrons[0] == pytest.approx(6.0, abs=0.02)
    assert out.electrons[1] == pytest.approx(8.0, abs=0.02)
    # charge = Z_val - N: atom 0 is a cation (+1), atom 1 an anion (-1)
    assert out.charges[0] == pytest.approx(+1.0, abs=0.02)
    assert out.charges[1] == pytest.approx(-1.0, abs=0.02)
    # partition conserves charge and fills the cell volume
    assert out.electrons.sum() == pytest.approx(14.0, abs=1e-6)
    assert out.total_electrons == pytest.approx(14.0, abs=1e-6)
    assert out.volumes.sum() == pytest.approx(216.0, abs=1e-9)
    assert len(out.nonnuclear) == 0


def test_non_orthogonal_cell():
    # sheared cell: the hop-length weighting must use true Cartesian distances
    cell = np.array([[6.0, 0.0, 0.0], [2.0, 6.0, 0.0], [1.0, 1.5, 6.0]])
    shape = (54, 54, 54)
    frac = [(0.3, 0.3, 0.3), (0.7, 0.7, 0.7)]
    rho = _gaussian_density(cell, shape, frac, [5.0, 5.0], sigma=0.45)
    res = _fake_result(cell, shape, frac, valence=[5.0, 5.0], rho=rho)

    out = bader(res)
    assert out.n_attractors == 2
    assert out.electrons[0] == pytest.approx(5.0, abs=0.03)
    assert out.electrons[1] == pytest.approx(5.0, abs=0.03)
    assert np.abs(out.charges).max() < 0.03


def test_spin_moment_partition():
    cell = np.diag([6.0, 6.0, 6.0])
    shape = (48, 48, 48)
    frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
    rho_up = _gaussian_density(cell, shape, frac, [5.0, 3.0], sigma=0.4)
    rho_dn = _gaussian_density(cell, shape, frac, [3.0, 3.0], sigma=0.4)
    rho = rho_up + rho_dn
    res = _fake_result(
        cell, shape, frac, valence=[8.0, 6.0], rho=rho,
        rho_spin=[torch.tensor(rho_up), torch.tensor(rho_dn)],
    )

    out = bader(res)
    assert out.moments is not None
    # atom 0 carries the 2 μ_B moment, atom 1 is unpolarised
    assert out.moments[0] == pytest.approx(2.0, abs=0.02)
    assert out.moments[1] == pytest.approx(0.0, abs=0.02)


def test_vacuum_threshold_drops_stray_basin():
    # a bulk-like pair plus a faint diffuse blob that forms its own low-ρ basin
    cell = np.diag([8.0, 8.0, 8.0])
    shape = (48, 48, 48)
    frac = [(0.2, 0.2, 0.2), (0.8, 0.8, 0.8)]
    rho = _gaussian_density(cell, shape, frac, [6.0, 6.0], sigma=0.4)
    blob = _gaussian_density(cell, shape, [(0.5, 0.5, 0.5)], [0.05], sigma=1.2)
    res = _fake_result(cell, shape, frac, valence=[6.0, 6.0], rho=rho + blob)

    peak = (rho + blob).max()
    out = bader(res, vacuum_threshold=0.01 * peak)
    # the diffuse basin is excluded from atom assignment, so no spurious charge
    assert out.electrons[0] == pytest.approx(6.0, abs=0.05)
    assert out.electrons[1] == pytest.approx(6.0, abs=0.05)


@pytest.mark.standard
def test_bader_on_real_si_scf():
    """End-to-end on a true SCFResult, and a regression on the pseudopotential
    caveat: the bare norm-conserving VALENCE density of covalent Si has no cusp
    at the nuclei, so its density maxima sit in the bonds, not on the atoms.

    The partition-invariants (charge conservation, cell neutrality) must hold
    regardless; per-atom charges are NOT meaningful for a cusp-free valence
    density (you would add the core / use PAW for that), and this test pins that
    behaviour so a future change to the maxima search is caught."""
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.scf.loop import scf, setup_system

    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    upf = si_upf()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged

    out = bader(res)
    # invariants that hold for ANY density: charge is conserved and the cell is
    # neutral (Σ electrons == Σ Z_val == n_electrons).
    assert out.total_electrons == pytest.approx(8.0, abs=1e-3)
    assert out.electrons.sum() == pytest.approx(8.0, abs=1e-3)
    assert abs(out.charges.sum()) < 1e-3
    # the documented caveat: bond-centred (non-nuclear) maxima dominate, each
    # ~1 Å off the Si sites — none of the attractors sits on a nucleus.
    assert len(out.nonnuclear) == out.n_attractors
    assert out.attractor_dist.min() > 0.5
