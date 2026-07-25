#!/usr/bin/env python3
"""Regenerate the finite-difference reference for
tests/integration/test_metal_forces_vs_qe.py::test_smeared_forces_match_free_energy_fd.

That test checks the analytic smeared force against a central finite difference
of the free energy. Computing the FD reference live costs 4 full SCFs per
smearing scheme (±h on two Cartesian components of atom 0). We cache those
displaced free energies here so the test runs a single live SCF — for the
analytic force under test — instead of five.

The cache is a *reference*, not a snapshot of the asserted quantity: the
analytic force is still computed live in the test and compared against this
numerical derivative. Rerun after any change to the free-energy model, the
smearing schemes, or the Al test cell:

    uv run python scripts/gen_al_forces_fd.py

It writes tests/fixtures/qe/al_forces_fd/fd_reference.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # make the `tests` package importable

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402
from tests.helpers import RY  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "qe"
AL_A = 4.05
AL_CELL = AL_A * np.eye(3)
AL_FRAC = np.array(
    [[0.03, 0.02, -0.015], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)
H = 1e-4


def free_energy(upf, smearing, pos) -> float:
    system = setup_system(AL_CELL, pos, [0] * 4, [upf],
                          ecut=20 * RY, kmesh=(2, 2, 2), nbands=32)
    res = scf(system, PBE(), smearing=smearing, width=0.1,
              etol=1e-11, rhotol=1e-10, verbose=False)
    assert res.converged
    return float(res.energies.free_energy)


def main() -> None:
    upf = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")
    base = AL_FRAC @ AL_CELL
    out: dict[str, dict] = {}
    for smearing in ("mp1", "cold"):
        rec: dict[str, object] = {"h": H}
        for comp in (0, 1):
            dp = np.zeros((4, 3))
            dp[0, comp] = H
            fp = free_energy(upf, smearing, base + dp)
            fm = free_energy(upf, smearing, base - dp)
            rec[str(comp)] = {"fp": fp, "fm": fm}
            print(f"{smearing} comp={comp}: fp={fp:.10f} fm={fm:.10f}")
        out[smearing] = rec
    dst = FIX / "al_forces_fd" / "fd_reference.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
