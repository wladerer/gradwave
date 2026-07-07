"""SCF total energies vs Quantum ESPRESSO at IDENTICAL parameters
(same UPF, ecut, k-mesh, smearing). Fixtures are committed; QE never runs
in the test suite (regenerate with tests/fixtures/qe/regenerate.py).

CI variants (low ecut, 2×2×2) run always; converged variants are @slow.
Tolerance: 1 meV/atom on total (free) energy — the M1 acceptance bar.
Note QE's printed/XML 'etot' IS the free energy F when smearing is on.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994

SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0, 0], [5.43 / 4] * 3])
AL_CELL = 4.05 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
C_CELL = 3.567 / 2 * FCC
C_POS = np.array([[0.0, 0, 0], [3.567 / 4] * 3])
GAAS_CELL = 5.653 / 2 * FCC
GAAS_POS = np.array([[0.0, 0, 0], [5.653 / 4] * 3])
MGO_CELL = 4.212 / 2 * FCC
MGO_POS = np.array([[0.0, 0.0, 0.0], [2.106, 2.106, 2.106]])  # frac (.5,.5,.5)
CU_CELL = 3.615 / 2 * FCC

CASES = {
    "si_lda_ci": dict(xc=LDA_PW92, ecut=15 * RY, kmesh=(2, 2, 2), cell=SI_CELL, pos=SI_POS,
                      pseudo="Si_ONCV_PBE-1.2.upf", nat=2, smearing="none", slow=False),
    "si_pbe_ci": dict(xc=PBE, ecut=15 * RY, kmesh=(2, 2, 2), cell=SI_CELL, pos=SI_POS,
                      pseudo="Si_ONCV_PBE-1.2.upf", nat=2, smearing="none", slow=False),
    "al_pbe_ci": dict(xc=PBE, ecut=20 * RY, kmesh=(2, 2, 2), cell=AL_CELL,
                      pos=np.zeros((1, 3)), pseudo="Al_ONCV_PBE-1.2.upf", nat=1,
                      smearing="gaussian", nbands=10, slow=False),
    # new chemistries: C (hard, light), GaAs (two species, Ga d-channel l=2
    # projectors + 3d semicore), MgO (ionic, O)
    "c_pbe_ci": dict(xc=PBE, ecut=30 * RY, kmesh=(2, 2, 2), cell=C_CELL, pos=C_POS,
                     pseudo="C_ONCV_PBE-1.2.upf", nat=2, smearing="none", slow=False),
    # PBE+SG15 GaAs at this mesh is a ZERO-gap semimetal (PBE gap collapse):
    # fixed occupations oscillate between degenerate states, so both codes
    # use matched 0.02 eV Gaussian smearing here.
    "gaas_pbe_ci": dict(xc=PBE, ecut=30 * RY, kmesh=(2, 2, 2), cell=GAAS_CELL,
                        pos=GAAS_POS, pseudo=("Ga_ONCV_PBE-1.2.upf", "As_ONCV_PBE-1.2.upf"),
                        nat=2, smearing="gaussian", width=0.02, nbands=13, slow=False),
    "mgo_pbe_ci": dict(xc=PBE, ecut=40 * RY, kmesh=(2, 2, 2), cell=MGO_CELL,
                       pos=MGO_POS, pseudo=("Mg_ONCV_PBE-1.2.upf", "O_ONCV_PBE-1.2.upf"),
                       nat=2, smearing="none", slow=False),
    # Cu: d-band metal (3s3p semicore + 3d10 4s1, l=2 projectors, smeared)
    "cu_pbe_ci": dict(xc=PBE, ecut=40 * RY, kmesh=(2, 2, 2), cell=CU_CELL,
                      pos=np.zeros((1, 3)), pseudo="Cu_ONCV_PBE-1.2.upf",
                      nat=1, smearing="gaussian", nbands=16, slow=False),
    "si_lda_scf": dict(xc=LDA_PW92, ecut=30 * RY, kmesh=(4, 4, 4), cell=SI_CELL, pos=SI_POS,
                       pseudo="Si_ONCV_PBE-1.2.upf", nat=2, smearing="none", slow=True),
    "si_pbe_scf": dict(xc=PBE, ecut=30 * RY, kmesh=(4, 4, 4), cell=SI_CELL, pos=SI_POS,
                       pseudo="Si_ONCV_PBE-1.2.upf", nat=2, smearing="none", slow=True),
    "al_pbe_scf": dict(xc=PBE, ecut=40 * RY, kmesh=(4, 4, 4), cell=AL_CELL,
                       pos=np.zeros((1, 3)), pseudo="Al_ONCV_PBE-1.2.upf", nat=1,
                       smearing="gaussian", nbands=10, slow=True),
}


def run_case(name):
    cfg = CASES[name]
    ref = json.loads((FIX / name / "reference.json").read_text())
    pseudos = cfg["pseudo"] if isinstance(cfg["pseudo"], tuple) else (cfg["pseudo"],)
    upfs = [parse_upf(FIX / "pseudos" / p) for p in pseudos]
    species = list(range(len(upfs))) if len(upfs) == cfg["nat"] else [0] * cfg["nat"]
    system = setup_system(
        cfg["cell"], cfg["pos"], species, upfs,
        ecut=cfg["ecut"], kmesh=cfg["kmesh"], nbands=cfg.get("nbands"),
    )
    res = scf(system, cfg["xc"](), smearing=cfg["smearing"], width=cfg.get("width", 0.1),
              etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged, f"{name}: SCF did not converge"
    assert res.n_iter < 25, f"{name}: too many SCF iterations ({res.n_iter})"
    ours = float(res.energies.free_energy)
    diff_mev_atom = abs(ours - ref["etot_eV"]) / cfg["nat"] * 1000
    assert diff_mev_atom < 1.0, (
        f"{name}: {ours:.8f} vs QE {ref['etot_eV']:.8f} -> {diff_mev_atom:.4f} meV/atom"
    )
    return res, ref


@pytest.mark.parametrize("name", [n for n, c in CASES.items() if not c["slow"]])
def test_scf_vs_qe_ci(name):
    torch.set_num_threads(4)
    run_case(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c["slow"]])
def test_scf_vs_qe_converged(name):
    torch.set_num_threads(8)
    res, ref = run_case(name)
    if "fermi_eV" in ref:
        assert abs(res.fermi - ref["fermi_eV"]) < 0.010  # 10 meV
