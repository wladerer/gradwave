"""Shared test constants and small structure factories.

Importable from both tests/unit/ and tests/integration/ as
``from tests.helpers import RY, FIX, si_fcc, fept_l10`` (tests/conftest.py
puts the repo root on sys.path so the import resolves regardless of the
working directory pytest was launched from).

Paths are anchored to this file, so tests that reference fixtures keep
working when pytest runs from outside the repo root.
"""

from pathlib import Path

import numpy as np

# Rydberg -> eV (CODATA), the ecut/energy unit used throughout the suite.
RY = 13.605693122994

# tests/fixtures, resolved absolutely from this file's location.
FIX = Path(__file__).parent / "fixtures"
PSEUDOS = FIX / "qe" / "pseudos"


def pseudo(name: str) -> str:
    """Absolute path (str) to a pseudopotential under tests/fixtures/qe/pseudos."""
    return str(PSEUDOS / name)


# FCC primitive-cell matrix; scale by a/2 for a conventional lattice constant a.
_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def si_fcc(a: float = 5.43):
    """Two-atom diamond-Si primitive cell and Cartesian positions (Angstrom)."""
    cell = a / 2 * _FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return cell, pos


def fept_l10(a: float = 2.723, c: float = 3.712):
    """L1_0 FePt tetragonal cell and Cartesian positions (Fe at origin, Pt at body center)."""
    cell = np.diag([a, a, c])
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    return cell, pos
