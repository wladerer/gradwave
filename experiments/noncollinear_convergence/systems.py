"""System builders for the non-collinear convergence campaign.

Every cell is a 1-atom (Pt/Ni/Fe fcc/bcc) or 2-atom (bcc canted) primitive with
a modest ecut / k-mesh: this is a convergence-BEHAVIOUR study, not a QE-accuracy
benchmark, so the settings are chosen to make each run cheap while keeping the
Fermi surface (the near-Stoner magnetization response) intact. Pseudos are the
committed QE fixtures under tests/fixtures/qe/pseudos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

RY = 13.605693122994  # eV per Ry (ecut passed in eV, following the NC tests)
_REPO = Path(__file__).resolve().parents[2]
PSE = _REPO / "tests" / "fixtures" / "qe" / "pseudos"

_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
_BCC = np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])

PSEUDOS = {
    "Ni_fr": "Ni_ONCV_PBE_FR-1.0.upf",   # has_so=T
    "Ni_sr": "PD_Ni_PBE.upf",            # has_so=F (scalar)
    "Fe_fr": "Fe_ONCV_PBE_FR-1.0.upf",   # has_so=T
    "Fe_sr": "Fe_ONCV_PBE-1.2.upf",      # has_so=F (scalar)
    "Pt_fr": "Pt_ONCV_PBE_FR-1.0.upf",   # has_so=T
}


def _mono(cell, upf_key, ecut_ry, kmesh, nbands):
    upf = parse_upf(PSE / PSEUDOS[upf_key])
    return setup_system(cell, np.array([[0.0, 0.0, 0.0]]), [0], [upf],
                        ecut=ecut_ry * RY, kmesh=kmesh, nbands=nbands,
                        use_symmetry=False, time_reversal=False)


def ni_fcc(soc: bool, ecut_ry=40, kmesh=(4, 4, 4), nbands=16):
    a = 3.52
    return _mono(a / 2 * _FCC, "Ni_fr" if soc else "Ni_sr", ecut_ry, kmesh, nbands)


def fe_bcc(soc: bool, ecut_ry=40, kmesh=(4, 4, 4), nbands=16):
    a = 2.87
    return _mono(a / 2 * _BCC, "Fe_fr" if soc else "Fe_sr", ecut_ry, kmesh, nbands)


def pt_fcc(ecut_ry=40, kmesh=(4, 4, 4), nbands=18):
    a = 3.92
    return _mono(a / 2 * _FCC, "Pt_fr", ecut_ry, kmesh, nbands)


def fe_bcc_2atom(soc: bool, ecut_ry=35, kmesh=(3, 3, 3), nbands=24):
    """Conventional 2-atom bcc Fe cell (corner + body-center): the substrate for
    a genuinely non-collinear seed (two moments started at a relative angle)."""
    a = 2.87
    cell = a * np.eye(3)
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ cell
    upf = parse_upf(PSE / PSEUDOS["Fe_fr" if soc else "Fe_sr"])
    return setup_system(cell, pos, [0, 0], [upf], ecut=ecut_ry * RY, kmesh=kmesh,
                        nbands=nbands, use_symmetry=False, time_reversal=False)
