"""Golden-energy gate for the refactor (docs/refactor_plan.md, stage 0).

Three cheap systems spanning PAW insulator, bare ultrasoft, and smeared
PAW metal, converged tightly and compared against recorded values at
1e-9 eV. A refactor stage that moves any of these numbers is changing
physics and stops until the difference is understood. Regenerate the
fixture ONLY for an intentional physics change, never to make a
refactor pass."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures"
RY = 13.605693122994
GOLD = json.loads((FIX / "golden" / "scf_golden.json").read_text())


def _check(name, f):
    ref = GOLD[name]
    assert abs(f - ref) < 1e-9, f"{name}: {f!r} vs golden {ref!r}"


def test_golden_si2_kjpaw():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "qe/pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    r = scf_uspp(setup_uspp(cell, pos, [0, 0], [paw], ecut=15 * RY,
                            kmesh=(2, 2, 2), ecutrho=60 * RY),
                 PBE(), etol=1e-11, rhotol=1e-10, verbose=False, max_iter=80)
    assert r["converged"]
    _check("si2_kjpaw_15ry", float(r["energies"].free_energy))


def test_golden_si2_rrkjus():
    torch.set_num_threads(8)
    us = parse_upf_paw(FIX / "qe/pseudos" / "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    r = scf_uspp(setup_uspp(cell, pos, [0, 0], [us], ecut=15 * RY,
                            kmesh=(2, 2, 2), ecutrho=60 * RY),
                 PBE(), etol=1e-11, rhotol=1e-10, verbose=False, max_iter=80)
    assert r["converged"]
    _check("si2_rrkjus_15ry", float(r["energies"].free_energy))


def test_golden_al_smeared():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "qe/pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4.04
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    r = scf_uspp(setup_uspp(cell, np.zeros((1, 3)), [0], [paw],
                            ecut=20 * RY, kmesh=(2, 2, 2),
                            ecutrho=100 * RY, nbands=8),
                 PBE(), smearing="gaussian", width=0.5, etol=1e-11,
                 rhotol=1e-10, verbose=False, max_iter=120)
    assert r["converged"]
    _check("al_kjpaw_smeared_20ry", float(r["energies"].free_energy))


def test_opts_path_reproduces_golden():
    """scf_uspp(opts=SCFOptions(...)) must be the same computation as the
    flat-kwargs path (stage-1 gate for the options object)."""
    from gradwave.scf.options import SCFOptions

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "qe/pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    opts = SCFOptions.from_kwargs(etol=1e-11, rhotol=1e-10, max_iter=80,
                                  verbose=False)
    r = scf_uspp(setup_uspp(cell, pos, [0, 0], [paw], ecut=15 * RY,
                            kmesh=(2, 2, 2), ecutrho=60 * RY),
                 PBE(), opts=opts)
    assert r["converged"]
    _check("si2_kjpaw_15ry", float(r["energies"].free_energy))
