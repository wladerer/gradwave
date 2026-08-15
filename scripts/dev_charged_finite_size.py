"""Probe: is gradwave's charged-cell electrostatics consistent (rung 3)?

A single Na atom charged to Na+ (tot_charge=+1, 8 electrons) in cubic boxes of
increasing side L, Γ-only. If the periodic charged-cell electrostatics are
consistent (the implicit −q jellium background is handled), the total energy
follows the Makov-Payne finite-size law

    E(L) = E_inf + q² α_M e² / (2 L)        α_M(SC) = 2.837297

so E vs 1/L is a straight line whose slope is q²·α_M·e²/2 ≈ 20.4 eV·Å (for q=1).
A wrong slope (or non-linear / erratic) means the net-charge G=0 monopole term is
missing and must be added. Neutral Na (q=0) is the control: no 1/L term.

    uv run python scripts/dev_charged_finite_size.py
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

torch.set_num_threads(8)
RY = 13.605693122994
ALPHA_M = 2.8372975  # simple-cubic Madelung constant
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qe" / "pseudos"
na = parse_upf(FIX / "Na_ONCV_PBE_sr.upf")

ECUT = 35 * RY
KW = dict(smearing="fermi-dirac", width=0.05, etol=1e-9, rhotol=1e-8,
          max_iter=400, verbose=False)


def energy(L, q):
    cell = L * np.eye(3)
    pos = np.array([[0.5, 0.5, 0.5]]) * L  # atom at box centre
    sysm = setup_system(cell, pos, [0], [na], ECUT, (1, 1, 1),
                        use_symmetry=False, tot_charge=q)
    res = scf(sysm, PBE(), **KW)
    assert res.converged, f"L={L} q={q} not converged"
    return float(res.energies.free_energy)


def scan(q, label):
    Ls = [8.0, 10.0, 12.0, 14.0]
    Es = [energy(L, q) for L in Ls]
    inv = np.array([1.0 / L for L in Ls])
    # linear fit E = E_inf + slope/L
    slope, e_inf = np.polyfit(inv, np.array(Es), 1)
    print(f"--- {label} (q={q}) ---")
    for L, E in zip(Ls, Es):
        print(f"  L={L:5.1f} Å   E={E:12.5f} eV   1/L={1 / L:.4f}")
    print(f"  fit: E_inf={e_inf:.5f} eV   slope(E vs 1/L)={slope:+.4f} eV·Å")
    if q != 0:
        expect = q * q * ALPHA_M * E2 / 2.0
        print(f"  Makov-Payne expected slope = q²·α_M·e²/2 = {expect:+.4f} eV·Å")
        print(f"  ratio measured/expected = {slope / expect:+.4f}\n")
    else:
        print()


def main():
    scan(0, "neutral Na  (control)")
    scan(1, "Na+ charged cell")


if __name__ == "__main__":
    main()
