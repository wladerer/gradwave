"""Shared Δ-factor EOS case definitions (PBE, ONCV-1.2 pseudos, fcc cells)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from structures import EOS_SCALES as SCALES  # noqa: E402,F401
from structures import RY, fcc_geometry, scaled_a  # noqa: E402,F401

CASES = {
    "si":   dict(a=5.43,  elems=["Si", "Si"], frac=[[0, 0, 0], [0.25] * 3],
                 ecut_ry=50, kmesh=(4, 4, 4), smearing="none", width=0.0, nbands=None),
    "c":    dict(a=3.567, elems=["C", "C"],   frac=[[0, 0, 0], [0.25] * 3],
                 ecut_ry=60, kmesh=(4, 4, 4), smearing="none", width=0.0, nbands=None),
    "gaas": dict(a=5.653, elems=["Ga", "As"], frac=[[0, 0, 0], [0.25] * 3],
                 ecut_ry=60, kmesh=(4, 4, 4), smearing="gaussian", width=0.02, nbands=13),
    "mgo":  dict(a=4.212, elems=["Mg", "O"],  frac=[[0, 0, 0], [0.5] * 3],
                 ecut_ry=65, kmesh=(4, 4, 4), smearing="none", width=0.0, nbands=None),
    "al":   dict(a=4.05,  elems=["Al"],       frac=[[0, 0, 0]],
                 ecut_ry=60, kmesh=(8, 8, 8), smearing="gaussian", width=0.1, nbands=10),
    "cu":   dict(a=3.615, elems=["Cu"],       frac=[[0, 0, 0]],
                 ecut_ry=90, kmesh=(8, 8, 8), smearing="gaussian", width=0.1, nbands=16),
}


def geometry(case, scale):
    """(cell, cart_positions, elems) for a volume factor `scale`."""
    cfg = CASES[case]
    return fcc_geometry(scaled_a(cfg["a"], scale), cfg["frac"], cfg["elems"])
