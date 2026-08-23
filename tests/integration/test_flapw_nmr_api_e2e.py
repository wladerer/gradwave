"""Rutile-TiO2 electric field gradient end-to-end through api.run (standard tier).

Builds a rutile-TiO2 EFG `Input` and runs it all the way through `api.run`,
reproducing a per-site V_zz / C_Q. The k-mesh/ecut here are a cheap MUFFIN-TIN
smoke config (Γ point, ecut 150, 12 iterations) — deterministic and stable
(TiO2 muffin-tin has no marginal-mode fragility), chosen so the test fits the
standard tier. It is a regression pin on the api → FLAPW → EFG → C_Q wiring, NOT
a physically-converged Elk comparison (the converged campaign numbers need
fullpot + kerker + newton_polish at k≥(3,3,3), tens of minutes — out of scope for
CI and independent of the EFG-accuracy work this bucket is decoupled from).

The pinned V_zz values were measured through api.run itself and match a direct
`flapw.crystal_scf_multi(..., efg=True)` call with identical parameters
bit-for-bit, which is the "reproduces the hand-driven numbers" acceptance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gradwave import api
from gradwave.constants import BOHR_ANG
from gradwave.flapw.nmr import quadrupolar_coupling
from gradwave.inputs import FlapwParams, Input, KPointsParams, NmrParams

pytestmark = pytest.mark.standard


def _rutile_tio2() -> Atoms:
    # a,a,c in Bohr (the campaign's exact rutile cell); ASE cell in Å
    a_bohr = np.array([8.68083, 8.68083, 5.59096])
    cell = np.diag(a_bohr * BOHR_ANG)
    u = 0.3048
    frac = [[0, 0, 0], [0.5, 0.5, 0.5], [u, u, 0], [1 - u, 1 - u, 0],
            [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5]]
    return Atoms("Ti2O4", scaled_positions=frac, cell=cell, pbc=True)


def test_rutile_tio2_efg_end_to_end(tmp_path):
    inp = Input(
        atoms=_rutile_tio2(), pseudo_dir=Path("."), pseudo_map={}, ecut=1.0,
        task="nmr", symmetry=True, kpoints=KPointsParams(mesh=(1, 1, 1)),
        flapw=FlapwParams(radii={"Ti": 0.95, "O": 0.80}, ecut=150.0, lmax=2,
                          iters=12, smearing=0.0, fullpot=False),
        nmr=NmrParams(task="efg", isotopes={"Ti": "49Ti", "O": "17O"}),
        output_dir=tmp_path, verbose=False)

    summary = api.run(inp, verbose=False)
    nmr = summary["nmr"]

    assert summary["task"] == "nmr"
    assert nmr["observable"] == "efg"
    assert nmr["n_sites"] == 6

    ti = next(s for s in nmr["sites"] if s["species"] == "Ti")
    o = next(s for s in nmr["sites"] if s["species"] == "O")

    # regression pin (deterministic muffin-tin smoke config; rtol tolerates
    # cross-machine BLAS drift while catching a real wiring/physics regression)
    assert o["V_zz_eV_ang2"] == pytest.approx(224.648, rel=2e-2)
    assert o["eta"] == pytest.approx(0.5254, rel=2e-2)
    assert ti["V_zz_eV_ang2"] == pytest.approx(-27.675, rel=2e-2)
    assert o["isotope"] == "17O" and ti["isotope"] == "49Ti"

    # C_Q wiring reproduces flapw.nmr.quadrupolar_coupling exactly
    for s in (ti, o):
        ref = quadrupolar_coupling(s["V_zz_eV_ang2"], s["eta"], s["isotope"])
        assert s["C_Q_MHz"] == pytest.approx(ref["C_Q_MHz"], rel=1e-12)
        assert s["abs_C_Q_MHz"] == pytest.approx(abs(ref["C_Q_MHz"]), rel=1e-12)
        assert s["C_Q_MHz"] == pytest.approx(
            2.4180 * s["Q_barn"] * s["V_zz_eV_ang2"], rel=1e-9)

    # the tensor is present and its largest-|eigenvalue| is the reported V_zz
    o_tensor = np.asarray(o["tensor_eV_ang2"])
    assert o_tensor.shape == (3, 3)
    eig = np.linalg.eigvalsh(0.5 * (o_tensor + o_tensor.T))
    assert np.max(np.abs(eig)) == pytest.approx(abs(o["V_zz_eV_ang2"]), rel=1e-6)

    # summary is serialized and the report rendered
    assert (tmp_path / "nmr.json").exists()
    assert "electric field gradient" in (tmp_path / "nmr.out").read_text()
