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


def test_run_wiring_writes_requested_fields(si_res, tmp_path):
    """The api.run() call site (_write_volumetric) honors the spec and records
    filenames; an unsupported field is skipped, not fatal."""
    from gradwave.api import _write_volumetric
    from gradwave.inputs import VolumetricParams

    spec = VolumetricParams(density=True, elf=True, bands=((0, 0),), format="cube")
    written = _write_volumetric(si_res, spec, tmp_path, verbose=False)
    assert written == {
        "density": "density.cube",
        "elf": "elf.cube",
        "parchg_b0_k0": "parchg_b0_k0.cube",
    }
    for name in written.values():
        assert (tmp_path / name).exists()

    # magnetization is unavailable for a collinear result — skipped, run survives
    skip = _write_volumetric(si_res, VolumetricParams(magnetization=True), tmp_path,
                             verbose=False)
    assert skip == {}


# --- noncollinear / SOC ----------------------------------------------------

@pytest.mark.standard  # full SOC SCF; not a fast-gate test
def test_noncollinear_spinor_export():
    """PARCHG sums the two spinor components (∫=1), density is the total, and
    the magnetization density integrates to res.mag_vec. ELF is guarded."""
    import torch

    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.noncollinear import scf_noncollinear
    from tests.helpers import FIX

    torch.set_num_threads(4)
    cell, pos = si_fcc(5.653)
    ga = parse_upf(FIX / "qe" / "pseudos" / "Ga_ONCV_PBE_FR-1.0.upf")
    as_ = parse_upf(FIX / "qe" / "pseudos" / "As_ONCV_PBE_FR-1.1.upf")
    system = setup_system(cell, pos, [0, 1], [ga, as_], ecut=20 * RY,
                          kmesh=(1, 1, 1), nbands=13, time_reversal=False)
    # the reconstruction identities are grid-exact at any state, so a few SCF
    # steps suffice — this tests the export path, not convergence.
    res = scf_noncollinear(system, NoncollinearXC(SpinPBE()),
                           mag_vec_init=[[0, 0, 1.0], [0, 0, 0]], smearing="gaussian",
                           width=0.1, etol=1e-7, rhotol=1e-6, verbose=False, max_iter=6)
    assert res.formalism == "noncollinear"

    dv = float(system.grid.volume) / system.grid.n_points
    assert V.density(res).sum() * dv == pytest.approx(system.n_electrons, rel=1e-5)
    # |ψ↑|² + |ψ↓|² integrates to 1
    assert V.band_density(res, band=0, kpoint=0).sum() * dv == pytest.approx(1.0, abs=1e-5)
    # magnetization components integrate to res.mag_vec
    for axis, i in (("x", 0), ("y", 1), ("z", 2)):
        assert V.magnetization(res, axis).sum() * dv == pytest.approx(
            float(res.mag_vec[i]), abs=1e-3)
    assert (V.magnetization(res, "abs") >= 0.0).all()

    with pytest.raises(NotImplementedError):
        V.elf(res)
    with pytest.raises(ValueError, match="noncollinear"):
        V.density(res, spin=0)
