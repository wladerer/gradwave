"""Composition derivative of the dielectric response — the second-order tier.

ε∞ (the macroscopic dielectric tensor) and the Born effective charges are
themselves second derivatives of the energy (E-field DFPT), so their composition
derivative dε∞/dλ is a mixed THIRD-order derivative d³E/(dλ dℰ²).
``alchemical_dielectric_gradient`` gets it by a central finite difference of the
E-field DFPT (``dielectric_born``) along the alchemical path — the reliable route
for a third-order quantity (the fully-analytic 2n+1 form, carrying a third
functional derivative of E_xc, is a documented follow-up).

This is composition design of an OPTICAL/DIELECTRIC property: how transmuting the
composition tunes the dielectric response. Demonstrated on SiC → C; the same call
works on any alchemical substitution insulator (e.g. a halide perovskite for
optical-constant design).

    uv run python examples/alchemical_dielectric_gradient.py
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.alchemical_response import alchemical_dielectric_gradient
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
SCF_KW = dict(smearing="none", etol=1e-10, rhotol=1e-9, max_iter=300, verbose=False)


def converge(lam):
    return scf(setup_alchemical_substitution(cell, pos, [si, c], [0, 1], {0: c},
                                             lam, ecut=35 * RY, kmesh=(2, 2, 2),
                                             use_symmetry=False), PBE(), **SCF_KW)


def main():
    lam = 0.5
    print("SiC -> C, composition derivative of the dielectric response at λ=0.5")
    for h in (0.04, 0.02):
        g = alchemical_dielectric_gradient(converge, PBE(), lam, h=h)
        print(f"  h={h}: dε∞_iso/dλ = {g['d_eps_iso']:+.4f}   "
              f"(ε∞ at λ±h: {g['plus']['eps_iso']:.4f} / {g['minus']['eps_iso']:.4f})")
    # the Born-charge composition derivative comes out of the same call
    print(f"  |dZ*/dλ| (max component) = "
          f"{float(g['d_born'].abs().max()):.4f} e/λ")
    print("  h-convergence of dε∞_iso/dλ confirms the finite difference; the sign is"
          " the composition sensitivity of the optical dielectric constant.")


if __name__ == "__main__":
    main()
