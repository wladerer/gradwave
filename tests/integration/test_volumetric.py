"""Volumetric export (CHGCAR / PARCHG / ELF analogs).

Physics contracts checked against a converged Si SCF:
  * ∫ρ dr = n_electrons                              (density is normalized)
  * ∫|ψ_nk|² dr = 1                                  (PARCHG is a probability)
  * Σ_k w_k Σ_n f_nk |ψ_nk|² = ρ                     (PARCHG sums to CHGCAR)
  * 0 ≤ ELF ≤ 1
Plus a round-trip through ASE's .cube/.xsf writers.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.core.xc.pbe import PBE
from gradwave.postscf import volumetric as V
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_fcc


@pytest.fixture(scope="module")
def si_res():
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    system = setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=(2, 2, 2), nbands=8)
    return scf(system, PBE(), smearing="gaussian", width=0.05,
               etol=1e-8, rhotol=1e-7, verbose=False)


def _dv(res):
    grid = res.system.grid
    return float(grid.volume) / grid.n_points


def test_density_integrates_to_electron_count(si_res):
    rho = V.density(si_res)
    ne = rho.sum() * _dv(si_res)
    assert ne == pytest.approx(si_res.system.n_electrons, rel=1e-6)


def test_band_density_is_normalized(si_res):
    bd = V.band_density(si_res, band=0, kpoint=0)
    assert bd.sum() * _dv(si_res) == pytest.approx(1.0, abs=1e-6)
    assert (bd >= 0.0).all()


def test_weighted_band_sum_reproduces_density(si_res):
    """The defining relation ρ = Σ_k w_k Σ_n f_nk |ψ_nk|² — to machine precision."""
    rho = V.density(si_res)
    kw = si_res.system.kweights.cpu().numpy()
    occ = si_res.occupations.cpu().numpy()
    nk, nb = occ.shape
    acc = np.zeros_like(rho)
    for k in range(nk):
        for n in range(nb):
            w = kw[k] * occ[k, n]
            if w < 1e-8:
                continue
            acc += w * V.band_density(si_res, band=n, kpoint=k)
    assert np.abs(acc - rho).max() < 1e-10


def test_elf_is_bounded(si_res):
    e = V.elf(si_res)
    assert np.isfinite(e).all()
    assert e.min() >= 0.0
    assert e.max() <= 1.0


@pytest.mark.parametrize("ext", [".cube", ".xsf"])
def test_writers_roundtrip(si_res, tmp_path, ext):
    from ase.io import read

    rho = V.density(si_res)
    out = V.write_density(si_res, tmp_path / f"chg{ext}")
    atoms = read(out)  # ASE parses the atoms block back
    assert len(atoms) == len(si_res.system.species_of_atom)
    if ext == ".cube":
        from ase.io.cube import read_cube_data

        data, _ = read_cube_data(out)
        assert data.shape == rho.shape
        assert np.abs(data - rho).max() < 1e-6


def test_unknown_extension_rejected(si_res, tmp_path):
    with pytest.raises(ValueError, match="unknown volumetric extension"):
        V.write_density(si_res, tmp_path / "chg.chgcar")
