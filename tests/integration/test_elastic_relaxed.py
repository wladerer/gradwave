"""Relaxed-ion elastic mode (elastic.mode: relaxed): plumbing + Kleinman drop.

Coarse-setting diamond Si (12 Ry, 2×2×2), one clamped and one relaxed-ion run.
Diamond is the minimal internal-strain case: normal strains induce no
symmetry-allowed internal displacement (C11/C12 and the bulk modulus must agree
between the modes to FD noise), while a shear strain shifts the two sublattices
against each other (the Kleinman parameter), so the relaxed-ion C44 must drop
well below the clamped one. The physically-converged relaxed C44 band is
asserted at production settings in test_elastic.py (slow tier); this file gates
the driver plumbing at standard-tier cost.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.api import run_elastic
from gradwave.inputs import ElasticParams, Input, KPointsParams, SmearingParams
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard


def _si_input(mode: str) -> Input:
    a = 5.47
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    return Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=12 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(2, 2, 2)),
        smearing=SmearingParams(type="none"),
        elastic=ElasticParams(strain=0.01, mode=mode, fmax=0.005))


@pytest.fixture(scope="module")
def si_pair():
    torch.set_num_threads(8)
    clamped = run_elastic(_si_input("clamped"), verbose=False)
    relaxed = run_elastic(_si_input("relaxed"), verbose=False)
    return clamped, relaxed


def test_relaxed_ion_block_schema(si_pair):
    clamped, relaxed = si_pair
    assert clamped["mode"] == "clamped"
    assert "relax_steps" not in clamped
    assert relaxed["mode"] == "relaxed"
    assert relaxed["all_converged"] and relaxed["relax_all_converged"]
    assert len(relaxed["relax_steps"]) == 12  # 6 Voigt strains × 2 signs
    # the unstrained input sits at a symmetry-pinned geometry: reference forces
    # vanish, so the reported ref_fmax must be far below the relax gate
    assert relaxed["ref_fmax_eV_ang"] < 1e-3


def test_relaxed_ion_kleinman_c44_drop(si_pair):
    clamped, relaxed = si_pair
    cc = np.array(clamped["c_GPa"])
    cr = np.array(relaxed["c_GPa"])
    c44_c = np.mean([cc[3, 3], cc[4, 4], cc[5, 5]])
    c44_r = np.mean([cr[3, 3], cr[4, 4], cr[5, 5]])
    # the internal sublattice shift softens the shear constants substantially
    # (production-setting PBE Si: 98 → 76); coarse settings shift both values
    # but not the sign or rough size of the drop
    assert c44_r < c44_c - 8.0, (c44_c, c44_r)
    # normal strains allow no internal displacement in diamond, so C11/C12
    # (and hence K) agree between the modes to FD/SCF noise
    assert np.allclose(cc[:3, :3], cr[:3, :3], atol=2.0), (cc[:3, :3], cr[:3, :3])


def test_relaxed_ion_shear_relaxes_normal_does_not(si_pair):
    _, relaxed = si_pair
    steps = relaxed["relax_steps"]  # FD order: j = 0..5, +h then −h
    normal, shear = steps[:6], steps[6:]
    # normal strains: forces stay zero by symmetry — BFGS converges immediately
    assert all(s == 0 for s in normal), steps
    # every shear strain must actually have moved the ions
    assert all(s >= 1 for s in shear), steps
