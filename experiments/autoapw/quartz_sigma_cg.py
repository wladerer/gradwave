"""Trustworthy alpha-quartz Si + O GIPAW shielding on the CG backend (one rung).

Deliverable 2 of the shielding profile: the per-site sigma_iso for BOTH Si AND
O at production-ish settings, with the hard-O ``response_backend='auto'`` cond(S)
gate routing O onto the ecut-stable matrix-free CG resolvent (PR #401). On the
OLD dense-eigh path the O sigma_iso diverged with ecut (the #392/#399 trust
boundary); on CG it is sane and ecut-stable.

One rung per invocation (argv: ecut_Ry ecutrho_Ry na nb nc), so O ecut-stability
is shown by running two rungs and diffing the O sigma_iso. Prints per-site Si
and O sigma_iso and writes ``quartz_sigma_<ecut>_<mesh>.json``.

    python quartz_sigma_cg.py 40 320 2 2 2
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.spacegroup import crystal

from gradwave.constants import RY_EV
from gradwave.inputs import Input, KPointsParams, NmrParams

PSEUDOS = Path("tests/fixtures/qe/pseudos").resolve()


def alpha_quartz():
    a, c = 4.9134, 5.4052
    return crystal(
        symbols=["Si", "O"],
        basis=[(0.4697, 0.0, 1.0 / 3.0), (0.4135, 0.2669, 0.1191)],
        spacegroup=152, cellpar=[a, a, c, 90, 90, 120])


def main() -> int:
    ecut_ry = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    ecutrho_ry = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0 * ecut_ry
    mesh = tuple(int(x) for x in sys.argv[3:6]) if len(sys.argv) >= 6 else (2, 2, 2)

    from gradwave.api import run_nmr

    atoms = alpha_quartz()
    inp = Input(
        atoms=atoms, pseudo_dir=PSEUDOS,
        pseudo_map={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"},
        ecut=ecut_ry * RY_EV, ecutrho=ecutrho_ry * RY_EV, xc="pbe",
        kpoints=KPointsParams(mesh=mesh), task="nmr", symmetry=False,
        nmr=NmrParams(task="shielding", shielding_level="gipaw", chunk_k=1),
        verbose=False)

    print(f"alpha-quartz {len(atoms)} atoms  ecut={ecut_ry:.0f}/{ecutrho_ry:.0f} Ry  "
          f"mesh={mesh}  gipaw + CG(auto)", flush=True)
    t0 = time.time()
    block = run_nmr(inp, verbose=False)
    dt = time.time() - t0

    si = [s for s in block["sites"] if s["species"] == "Si"]
    ox = [s for s in block["sites"] if s["species"] == "O"]
    si_iso = [s["sigma_iso_ppm"] for s in si]
    o_iso = [s["sigma_iso_ppm"] for s in ox]
    print(f"  Si sigma_iso (ppm): {[round(v, 2) for v in si_iso]}  "
          f"mean={np.mean(si_iso):.2f}  spread={max(si_iso) - min(si_iso):.2f}",
          flush=True)
    print(f"  O  sigma_iso (ppm): {[round(v, 2) for v in o_iso]}  "
          f"mean={np.mean(o_iso):.2f}  spread={max(o_iso) - min(o_iso):.2f}",
          flush=True)
    print(f"  ({dt:.0f}s)", flush=True)

    out = {
        "ecut_Ry": ecut_ry, "ecutrho_Ry": ecutrho_ry, "mesh": list(mesh),
        "wall_s": round(dt, 1),
        "si_iso_ppm": si_iso, "si_mean_ppm": float(np.mean(si_iso)),
        "o_iso_ppm": o_iso, "o_mean_ppm": float(np.mean(o_iso)),
        "sites": block["sites"],
    }
    tag = f"{int(ecut_ry)}_{''.join(str(m) for m in mesh)}"
    path = Path(f"quartz_sigma_{tag}.json")
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"  wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
