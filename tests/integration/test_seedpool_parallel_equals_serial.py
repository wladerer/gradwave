"""SeedPool campaign parity: the PARALLEL (n_workers>1) forward campaigns must
reproduce the SERIAL (n_workers=1) result for a small cubic metal (fcc Al, PBE).

Slow tier — each case runs a full hub-and-spoke campaign twice (serial then
parallel over worker processes). The parallel path computes the reference once,
checkpoints it, and fans the independent spoke SCFs out to a ProcessPoolExecutor
that warm-starts every spoke from that checkpoint. The physics is unchanged, so:

  * phonons  — the folded frequencies match to ~1 cm⁻¹,
  * elastic  — the clamped-ion stiffness C matches to a fraction of a GPa,
  * eos      — V0/B0 match (parallel warm-starts every volume from the
               reference rather than the neighbour chain, so agreement is to SCF
               tolerance, not bit-for-bit).

Never run this on the laptop — it executes many SCFs. The parent runs it on asus.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gradwave.api import run_elastic, run_eos, run_phonons
from gradwave.inputs import (
    ElasticParams,
    EOSParams,
    Input,
    KPointsParams,
    PhononParams,
    SmearingParams,
)
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.slow


def _al_input(**overrides) -> Input:
    """fcc Al, coarse but converged enough for a serial-vs-parallel parity check."""
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    atoms = Atoms("Al", positions=[[0.0, 0, 0]], cell=cell, pbc=True)
    base = dict(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Al": "Al_ONCV_PBE-1.2.upf"}, ecut=18 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(6, 6, 6)),
        smearing=SmearingParams(type="gaussian", width=0.1),
    )
    base.update(overrides)
    return Input(**base)


def test_phonons_parallel_matches_serial():
    ph = PhononParams(supercell=(2, 2, 2), displacement=0.02, npoints=30,
                      dos_mesh=(0, 0, 0))
    serial = run_phonons(_al_input(phonons=ph), verbose=False)
    ph_par = PhononParams(supercell=(2, 2, 2), displacement=0.02, npoints=30,
                          dos_mesh=(0, 0, 0), n_workers=2)
    parallel = run_phonons(_al_input(phonons=ph_par), verbose=False)
    fs = np.array(serial["frequencies_cm1"])
    fp = np.array(parallel["frequencies_cm1"])
    assert fs.shape == fp.shape
    assert np.allclose(fs, fp, atol=1.0)   # cm⁻¹


def test_elastic_clamped_parallel_matches_serial():
    el = ElasticParams(strain=0.005, mode="clamped")
    serial = run_elastic(_al_input(elastic=el), verbose=False)
    el_par = ElasticParams(strain=0.005, mode="clamped", n_workers=2)
    parallel = run_elastic(_al_input(elastic=el_par), verbose=False)
    cs = np.array(serial["c_GPa"])
    cp = np.array(parallel["c_GPa"])
    assert np.allclose(cs, cp, atol=0.5)   # GPa
    assert np.isclose(serial["bulk_modulus_GPa"]["hill"],
                      parallel["bulk_modulus_GPa"]["hill"], atol=0.5)


def test_eos_parallel_matches_serial():
    eos = EOSParams(scales=(0.96, 0.98, 1.00, 1.02, 1.04), energy="total")
    serial = run_eos(_al_input(eos=eos), verbose=False)
    eos_par = EOSParams(scales=(0.96, 0.98, 1.00, 1.02, 1.04), energy="total",
                        n_workers=2)
    parallel = run_eos(_al_input(eos=eos_par), verbose=False)
    assert np.isclose(serial["v0_ang3_per_atom"], parallel["v0_ang3_per_atom"],
                      atol=1e-2)
    assert np.isclose(serial["b0_GPa"], parallel["b0_GPa"], atol=1.0)
    # energies agree per volume to SCF tolerance (shared-seed vs neighbour chain)
    assert np.allclose(serial["energies_eV_per_atom"],
                       parallel["energies_eV_per_atom"], atol=1e-4)
