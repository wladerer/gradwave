"""ε∞ and Born effective charges vs QE 7.5 ph.x (epsil/zeu, E-field DFPT).

Si is the precision test: valence-only pseudo, and both codes agree on the
(4×4×4-mesh) raw physics to ~1e-4 — including the raw pre-ASR Born charge,
which is a mesh artifact both codes must reproduce identically. MgO is the
polar test (Z* ≈ ±2) at cross-code tolerance (~1%): the residual there is
implementation-independent (dk-invariant) methodological difference.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.slow
def test_si_epsilon_and_born_vs_qe():
    torch.set_num_threads(8)
    ref = json.load(open(FIX / "eps_born" / "reference.json"))["si"]
    a = ref["a_angstrom"]
    si = parse_upf(FIX / "pseudos" / ref["pseudo"])
    system = setup_system(a / 2 * FCC, np.array([[0.0, 0, 0], [a / 4] * 3]),
                          [0, 0], [si], ecut=ref["ecutwfc_ry"] * RY,
                          kmesh=tuple(ref["kmesh"]), use_symmetry=False)
    res = scf(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    out = dielectric_born(res, PBE())
    eps = out["eps"]
    # isotropic + diagonal by symmetry (emerges numerically, not imposed)
    assert float((eps - torch.diag(torch.diagonal(eps))).abs().max()) < 1e-3
    assert abs(out["eps_iso"] - ref["eps_inf"]) < 5e-3  # QE to 4 decimal places
    # raw (pre-ASR) Born charge matches QE's raw value; ASR-corrected is 0
    z_raw = float(out["born"][0, 0, 0])
    assert abs(z_raw - ref["born_raw_diag"]) < 2e-3
    z_asr = out["born"] - out["asr"][None] / 2.0
    assert float(z_asr.abs().max()) < 2e-3


@pytest.mark.slow
def test_mgo_epsilon_and_born_vs_qe():
    torch.set_num_threads(8)
    ref = json.load(open(FIX / "eps_born" / "reference.json"))["mgo"]
    a = ref["a_angstrom"]
    mg = parse_upf(FIX / "pseudos" / ref["pseudos"]["Mg"])
    o = parse_upf(FIX / "pseudos" / ref["pseudos"]["O"])
    system = setup_system(a / 2 * FCC, np.array([[0.0, 0, 0], [a / 2] * 3]),
                          [0, 1], [mg, o], ecut=ref["ecutwfc_ry"] * RY,
                          kmesh=tuple(ref["kmesh"]), use_symmetry=False)
    res = scf(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    out = dielectric_born(res, PBE())
    assert abs(out["eps_iso"] - ref["eps_inf"]) / ref["eps_inf"] < 0.01
    assert abs(float(out["born"][0, 0, 0]) - ref["born_raw"]["Mg"]) < 0.01
    assert abs(float(out["born"][1, 0, 0]) - ref["born_raw"]["O"]) < 0.01
    assert abs(float(out["asr"][0, 0]) - ref["asr_sum_raw"]) < 0.02
