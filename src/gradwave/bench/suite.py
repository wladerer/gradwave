"""A hardness-graded suite of small SCF benchmark cases.

Each :class:`BenchCase` builds a cheap primitive-cell system and carries the
run-time SCF knobs plus a descriptor (the feature vector a model conditions on).
The point is *convergence dynamics*, not total-energy accuracy — small cells,
modest cutoff — so the recorder telemetry differs meaningfully across the
hardness gradient: wide-gap insulator → small-gap insulator → free-electron
metal → transition-metal d-band (the charge-sloshing regime).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

# committed fixture pseudopotentials; override for an installed (non-repo) run
_DEFAULT_PSEUDO_DIR = Path(__file__).resolve().parents[3] / "tests/fixtures/qe/pseudos"
_PSEUDO_DIR = Path(os.environ.get("GRADWAVE_PSEUDO_DIR", _DEFAULT_PSEUDO_DIR))


def _fcc(a: float) -> np.ndarray:
    return (a / 2.0) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _bcc(a: float) -> np.ndarray:
    return (a / 2.0) * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])


@dataclass
class BenchCase:
    """One benchmark system + its fixed run-time SCF configuration."""

    name: str
    hardness: str                       # insulator | simple-metal | transition-metal
    _build: Callable[[], object]
    scf_kwargs: dict[str, Any]          # smearing/width/max_iter/etol/rhotol/nspin
    descriptor: dict[str, Any] = field(default_factory=dict)

    def build(self):
        return self._build()

    def xc(self):
        return PBE()


def _case(name, hardness, pseudo, cell, pos, *, ecut_ry, kmesh, nband_buffer,
          smearing, width, gap_hint, use_symmetry=True):
    upf = parse_upf(_PSEUDO_DIR / pseudo)
    zval = float(upf.z_valence)
    natoms = len(pos)
    nelec = zval * natoms
    nbands = int(np.ceil(nelec / 2)) + nband_buffer

    def build():
        # Forward-SCF benchmarks reduce k to the IBZ by the crystal point group
        # (setup_system use_symmetry) — exact (energy bit-for-bit) and ~10-15x
        # faster on symmetric metals. Set use_symmetry=False for a case that will
        # feed the differentiable/response path (a perturbation breaks the group).
        return setup_system(np.asarray(cell, float), np.asarray(pos, float),
                            [0] * natoms, [upf], ecut=ecut_ry * RY, kmesh=kmesh,
                            nbands=nbands, use_symmetry=use_symmetry)

    desc = dict(hardness=hardness, n_atoms=natoms, z_valence=zval, n_electrons=nelec,
                ecut_ry=ecut_ry, kmesh="x".join(map(str, kmesh)), nbands=nbands,
                smearing=smearing, width_eV=width, gap_hint_eV=gap_hint,
                use_symmetry=use_symmetry, pseudo=pseudo)
    scf_kw = dict(smearing=smearing, width=width, max_iter=70, etol=1e-6,
                  rhotol=1e-5, verbose=False)
    return BenchCase(name, hardness, build, scf_kw, desc)


def build_suite() -> list[BenchCase]:
    """Four cases spanning the convergence-difficulty gradient."""
    a_c, a_si, a_al, a_cu = 3.567, 5.431, 4.050, 3.610
    return [
        _case("C-diamond", "insulator", "C_ONCV_PBE-1.2.upf", _fcc(a_c),
              [[0, 0, 0], [a_c / 4] * 3], ecut_ry=45, kmesh=(4, 4, 4),
              nband_buffer=4, smearing="none", width=0.0, gap_hint=5.5),
        _case("Si", "insulator", "Si_ONCV_PBE-1.2.upf", _fcc(a_si),
              [[0, 0, 0], [a_si / 4] * 3], ecut_ry=30, kmesh=(4, 4, 4),
              nband_buffer=4, smearing="none", width=0.0, gap_hint=0.6),
        _case("Al", "simple-metal", "Al_ONCV_PBE-1.2.upf", _fcc(a_al),
              [[0, 0, 0]], ecut_ry=30, kmesh=(8, 8, 8),
              nband_buffer=8, smearing="gaussian", width=0.15, gap_hint=0.0),
        _case("Cu", "transition-metal", "Cu_ONCV_PBE-1.2.upf", _fcc(a_cu),
              [[0, 0, 0]], ecut_ry=45, kmesh=(8, 8, 8),
              nband_buffer=10, smearing="gaussian", width=0.15, gap_hint=0.0),
    ]


SUITE: list[BenchCase] = build_suite()
