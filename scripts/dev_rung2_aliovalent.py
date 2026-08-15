"""Dev validation: aliovalent alchemical energy gradient with Janak (rung 2).

A single-site aliovalent transmutation in a 2-atom diamond Si cell: substitute
one Si -> As (ΔZ = +1, n-type). The electron count follows the ionic charge,
N(λ) = ΣZ_i(λ), so N is fractional mid-path and the cell is metallic (smearing
on). The energy gradient dF/dλ then carries the Janak chemical-potential term
μ·dN/dλ on top of the bare Hellmann-Feynman ionic derivative — dropping it would
miss the whole N-change contribution. Checked against a central finite
difference of re-converged free energies. An isovalent control (Si -> C, ΔZ = 0)
confirms the Janak term vanishes there and the gradient still matches FD.

    uv run python scripts/dev_rung2_aliovalent.py
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import alchemical_energy_gradient, setup_alchemical_substitution
from gradwave.scf.loop import scf

torch.set_num_threads(8)
RY = 13.605693122994
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qe" / "pseudos"

si = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
as_ = parse_upf(FIX / "As_ONCV_PBE-1.2.upf")
c = parse_upf(FIX / "C_ONCV_PBE-1.2.upf")

a = 5.43
cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
ECUT, KM = 30 * RY, (3, 3, 3)
SCF_KW = dict(smearing="fermi-dirac", width=0.15, etol=1e-10, rhotol=1e-9,
              max_iter=400, verbose=False)


def experiment(name, target, dZ):
    print(f"=== {name}: one Si -> {name.split('->')[1].strip()}  (ΔZ={dZ:+d}) ===")

    def run(lam):
        sysm = setup_alchemical_substitution(
            cell, pos, [si], [0, 0], {1: target}, lam, ecut=ECUT, kmesh=KM,
            use_symmetry=False)
        return scf(sysm, PBE(), **SCF_KW)

    lam = 0.5
    res = run(lam)
    assert res.converged, "SCF did not converge"
    dN_dlam = float((res.system.alchemical["z_target"]
                     - res.system.alchemical["z_base"]).sum())
    janak = float(res.fermi) * dN_dlam
    print(f"  N(λ)={res.system.n_electrons:.3f}  μ={res.fermi:.4f} eV  "
          f"dN/dλ={dN_dlam:+.1f}  Janak μ·dN/dλ={janak:+.4f} eV")

    dF = float(alchemical_energy_gradient(res, lam, xc=PBE()))
    h = 0.01
    Fp = float(run(lam + h).energies.free_energy)
    Fm = float(run(lam - h).energies.free_energy)
    fd = (Fp - Fm) / (2 * h)
    print(f"  dF/dλ analytic (with Janak) = {dF:+.4f} eV")
    print(f"  dF/dλ finite difference     = {fd:+.4f} eV   "
          f"(agree {abs(dF - fd) * 1e3:.2f} meV)")
    print(f"  Janak share of the gradient = {janak / dF * 100:+.1f}%\n")


def main():
    experiment("Si -> As", as_, +1)   # aliovalent, n-type
    experiment("Si -> C", c, 0)        # isovalent control, Janak vanishes


if __name__ == "__main__":
    main()
