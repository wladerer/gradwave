#!/usr/bin/env python3
"""Regenerate the finite-difference reference for
tests/integration/test_forces_nlcc.py::test_nlcc_force_matches_finite_difference.

That test checks the analytic NLCC core-correction force (−∫ v_xc ∂ρ_core/∂τ)
against a central finite difference of the total energy. Computing the FD live
costs 6 full SCFs (±h on three Cartesian components); we cache those displaced
total energies here so the test runs a single live SCF — for the analytic force
under test — instead of seven.

The cache is a *reference*, not a snapshot of the asserted quantity: the analytic
force is still computed live in the test and compared against this numerical
derivative. The cell, cutoff, k-mesh, displacement set and step are imported from
the test module, so they cannot drift. Rerun after any change to the total-energy
model, the NLCC term, or those inputs:

    uv run python scripts/gen_nlcc_forces_fd.py

It writes tests/fixtures/qe/nlcc_forces_fd/fd_reference.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # make the `tests` package importable

from gradwave.pseudo.upf import parse_upf  # noqa: E402
from tests.helpers import PSEUDOS  # noqa: E402
from tests.integration.test_forces_nlcc import (  # noqa: E402
    _DISP,
    _H,
    FD_FIX,
    POS,
    _total_energy,
)


def main() -> None:
    upf = parse_upf(PSEUDOS / "C_ONCV_PBE_sr.upf")
    assert upf.core_rho is not None, "fixture pseudo must carry an NLCC core charge"
    out: dict[str, object] = {"h": _H}
    for ia, ic in _DISP:
        pos_p = np.array(POS, copy=True)
        pos_m = np.array(POS, copy=True)
        pos_p[ia, ic] += _H
        pos_m[ia, ic] -= _H
        fp = _total_energy(upf, pos_p)
        fm = _total_energy(upf, pos_m)
        out[f"{ia},{ic}"] = {"fp": fp, "fm": fm}
        print(f"comp ({ia},{ic}): fp={fp:.10f} fm={fm:.10f}")
    FD_FIX.parent.mkdir(parents=True, exist_ok=True)
    FD_FIX.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {FD_FIX}")


if __name__ == "__main__":
    main()
