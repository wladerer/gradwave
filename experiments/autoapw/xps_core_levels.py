"""FLAPW core-level (XPS) validation driver — within-cell shifts and the
cross-cell Si-2p Si-vs-SiO2 chemical shift.

Runs the small all-electron FLAPW cells used in ``xps_validation.md`` and prints
the numbers that go into the gradwave-vs-Elk / gradwave-vs-experiment tables:

* ``rutile``   — real-crystal equivalent-site null (all O/Ti identical -> 0).
* ``si``       — bulk diamond Si: Si 2p core level + valence-band maximum (VBM).
* ``sio2``     — ideal beta-cristobalite SiO2 (6-atom FCC primitive): Si 2p + VBM.
* ``headline`` — runs ``si`` and ``sio2`` and reports the cross-cell initial-state
                 Si-2p binding-energy shift referenced to each cell's VBM.

Absolute FLAPW core levels float on each cell's interstitial zero, so the physics
is in WITHIN-CELL shifts (site vs site) and, across cells, in VBM-referenced
binding energies (``flapw.core_levels.cross_cell_binding_shift``) — the reference
cancels the common wander. Run on asus (dense O(N^3) FLAPW); keep OMP modest.

    uv run python experiments/autoapw/xps_core_levels.py headline 2 220
"""

from __future__ import annotations

import sys

import numpy as np

from gradwave.flapw import crystal_scf_multi
from gradwave.flapw.core_levels import cross_cell_binding_shift

BOHR = 0.529177210903


def _vbm(bands: dict, nocc: int) -> float:
    """Gamma-point valence-band maximum (eV) = highest occupied band. Same
    interstitial-zero frame as the core levels, so BE = VBM - e_core cancels it."""
    ev = np.asarray(bands["ev"], dtype=float)
    return float(ev[nocc - 1])


def run_rutile(km: int, ec: float) -> dict:
    cell = np.array([[8.68083, 0, 0], [0, 8.68083, 0], [0, 0, 5.59096]])  # bohr
    u = 0.3048
    atoms = [((0, 0, 0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
             ((u, u, 0), "O"), ((1 - u, 1 - u, 0), "O"),
             ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]
    _, info = crystal_scf_multi(cell, atoms, {"Ti": 1.098, "O": 0.824}, ecut=ec,
                                iters=45, kmesh=(km, km, km), use_symmetry=True)
    return info


def run_si(km: int, ec: float) -> tuple[dict, dict]:
    a = 5.431 / BOHR
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])  # bohr
    atoms = [((0.0, 0.0, 0.0), "Si"), ((0.25, 0.25, 0.25), "Si")]
    bands, info = crystal_scf_multi(cell, atoms, {"Si": 1.10}, ecut=ec, iters=50,
                                    kmesh=(km, km, km), use_symmetry=True)
    return bands, info


def run_sio2(km: int, ec: float, a_ang: float = 7.40) -> tuple[dict, dict]:
    a = a_ang / BOHR
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])  # bohr
    atoms = [((0.0, 0.0, 0.0), "Si"), ((0.25, 0.25, 0.25), "Si"),
             ((0.125, 0.125, 0.125), "O"), ((0.625, 0.125, 0.125), "O"),
             ((0.125, 0.625, 0.125), "O"), ((0.125, 0.125, 0.625), "O")]
    bands, info = crystal_scf_multi(cell, atoms, {"Si": 0.80, "O": 0.75}, ecut=ec,
                                    iters=60, kmesh=(km, km, km), use_symmetry=True)
    return bands, info


def _print_levels(tag: str, info: dict) -> None:
    print(f"=== {tag} core levels (eV, interstitial-zero frame) ===")
    for k, rec in info["core_levels"].items():
        print(f"  {k} {rec['symbol']:2s} " +
              " ".join(f"{o}={v:.4f}" for o, v in rec["levels"].items()))
    print(f"  within-cell shifts: {info['core_level_shifts']}")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "headline"
    km = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ec = float(sys.argv[3]) if len(sys.argv) > 3 else 220.0

    if what == "rutile":
        _print_levels("rutile TiO2", run_rutile(km, ec))
        return
    if what in ("si", "headline"):
        b_si, i_si = run_si(km, ec)
        _print_levels("bulk Si", i_si)
        vbm_si = _vbm(b_si, nocc=4)          # 2 Si x 4 e / 2
        print(f"  bulk Si VBM(Gamma) = {vbm_si:.4f} eV")
    if what in ("sio2", "headline"):
        b_ox, i_ox = run_sio2(km, ec)
        _print_levels("beta-cristobalite SiO2", i_ox)
        vbm_ox = _vbm(b_ox, nocc=16)         # (2 Si x 4 + 4 O x 6) / 2
        print(f"  SiO2 VBM(Gamma) = {vbm_ox:.4f} eV")
    if what == "headline":
        shift = cross_cell_binding_shift(i_si["core_levels"], vbm_si,
                                         i_ox["core_levels"], vbm_ox, "Si", "2p")
        print("=== Si 2p cross-cell shift (SiO2 - Si), VBM-referenced ===")
        print(f"  BE(Si)   = {shift['BE_a_eV']:.3f} eV")
        print(f"  BE(SiO2) = {shift['BE_b_eV']:.3f} eV")
        print(f"  Delta BE = {shift['delta_BE_eV']:+.3f} eV   (experiment ~ +4 eV)")


if __name__ == "__main__":
    main()
