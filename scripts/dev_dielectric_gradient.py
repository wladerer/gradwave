"""Dev probe: composition derivative of ε∞ by FD of the E-field DFPT (rung: response tier).

SiC -> C isovalent substitution: compute dε∞/dλ, check it is finite and sensible
and h-convergent. Gauges the cost of dielectric_born on the alchemical System.

    uv run python scripts/dev_dielectric_gradient.py
"""

import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.alchemical_response import alchemical_dielectric_gradient
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import setup_alchemical_substitution
from gradwave.scf.loop import scf

torch.set_num_threads(8)
RY = 13.605693122994
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qe" / "pseudos"
si = parse_upf(FIX / "Si_ONCV_PBE_sr.upf")
c = parse_upf(FIX / "C_ONCV_PBE_sr.upf")

a = 4.36
cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
ECUT, KM = 35 * RY, (2, 2, 2)
SCF_KW = dict(smearing="none", etol=1e-10, rhotol=1e-9, max_iter=300, verbose=False)


def converge(lam):
    sysm = setup_alchemical_substitution(cell, pos, [si, c], [0, 1], {0: c}, lam,
                                         ecut=ECUT, kmesh=KM, use_symmetry=False)
    return scf(sysm, PBE(), **SCF_KW)


def main():
    lam = 0.5
    t = time.time()
    d0 = dielectric_born(converge(lam), PBE())
    print(f"ε∞(λ={lam}) iso = {d0['eps_iso']:.4f}   (one dielectric_born: {time.time() - t:.1f}s)")
    for h in (0.04, 0.02):
        t = time.time()
        g = alchemical_dielectric_gradient(converge, PBE(), lam, h=h)
        print(f"  h={h}: dε∞_iso/dλ = {g['d_eps_iso']:+.4f}   "
              f"ε∞(+)={g['plus']['eps_iso']:.4f} ε∞(-)={g['minus']['eps_iso']:.4f}   "
              f"({time.time() - t:.1f}s)")
    print("  (h-convergence of dε∞_iso/dλ tests the FD; sign shows how C widens/narrows"
          " the dielectric response vs Si)")


if __name__ == "__main__":
    main()
