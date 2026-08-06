#!/usr/bin/env python3
"""Regenerate the finite-difference reference for
tests/integration/test_forces_nlcc_metagga.py::test_metagga_nlcc_force_matches_finite_difference.

That test checks the analytic r2SCAN NLCC force against a central finite
difference of the r2SCAN total energy. Computing the FD reference live costs
six full meta-GGA SCFs (±h on three Cartesian components); the analytic force
needs a seventh. We cache the six displaced total energies here so the test
runs a single live SCF — for the analytic force under test — instead of seven.

The cache is a *reference*, not a snapshot of the asserted quantity: the
analytic force is still computed live in the test and compared against this
numerical derivative. Rerun after any change to the r2SCAN model, the NLCC
core-correction force path, or the carbon test cell:

    uv run python scripts/gen_metagga_nlcc_fd.py

It writes tests/fixtures/qe/metagga_nlcc_fd/fd_reference.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # make the `tests` package importable

from gradwave.core.xc.r2scan import R2SCAN  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402
from tests.helpers import PSEUDOS, RY  # noqa: E402

# Same low-symmetry (triclinic, rattled) 2-atom carbon cell as the test.
A = 3.2
CELL = A * np.array([[1.0, 0.0, 0.0], [0.12, 1.0, 0.0], [0.05, 0.08, 1.05]])
FRAC = np.array([[0.02, 0.01, 0.0], [0.27, 0.31, 0.24]])
POS = FRAC @ CELL
H = 1e-4
# The three components probed by the test (atom, Cartesian direction).
COMPONENTS = [(1, 0), (1, 1), (0, 2)]


def total_energy(upf, pos) -> float:
    system = setup_system(CELL, pos, [0, 0], [upf], ecut=30 * RY, kmesh=(2, 2, 2))
    res = scf(system, R2SCAN(), smearing="gaussian", width=0.05,
              etol=1e-11, rhotol=1e-10, verbose=False)
    assert res.converged
    return float(res.energies.total)


def main() -> None:
    upf = parse_upf(PSEUDOS / "C_ONCV_PBE_sr.upf")
    assert upf.core_rho is not None, "fixture must have an NLCC core charge"
    out: dict[str, object] = {"h": H}
    for ia, ic in COMPONENTS:
        vals = {}
        for sign in (+1, -1):
            pos = POS.copy()
            pos[ia, ic] += sign * H
            key = "ep" if sign > 0 else "em"
            vals[key] = total_energy(upf, pos)
        out[f"{ia},{ic}"] = vals
        print(f"comp ({ia},{ic}): ep={vals['ep']:.10f} em={vals['em']:.10f}")
    dst = REPO / "tests" / "fixtures" / "qe" / "metagga_nlcc_fd" / "fd_reference.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
