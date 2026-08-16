"""Optoelectronic composition design of a halide perovskite.

The two properties that matter for a perovskite absorber are the band gap (which
sets the absorption onset) and the electronic dielectric constant ε∞ (which sets
the screening / exciton binding / optical response). Both are differentiable in
composition here, so one converged reference gives their full sensitivity to
transmuting the halide sublattice CsPbI3 -> CsPbCl3:

  * dGap/dλ   — the relaxed band-gap gradient, analytic (composition DFPT).
  * dε∞/dλ    — the composition derivative of the dielectric response, by a
                central finite difference of the E-field DFPT (the second-order
                tier; the fully-analytic 2n+1 form is a documented follow-up).

Physical sanity: iodine -> chlorine widens the gap (dGap/dλ > 0) and lowers ε∞
(Cl is less polarizable and the gap is wider, so ε∞ ~ 1/gap² drops) — so the two
design gradients should have opposite sign. This is the composition-to-property
Jacobian a perovskite absorber would be tuned along.

    uv run python examples/perovskite_optoelectronic_design.py

Scalar-relativistic PBE, nspin=1 insulator, use_symmetry=False.
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.api._common import _gap
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.alchemical_response import alchemical_dielectric_gradient
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import alchemical_gap_gradient, setup_alchemical_substitution
from gradwave.scf.loop import scf

torch.set_num_threads(8)
RY = 13.605693122994
ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
DG = ROOT / "benchmarks" / "delta_gauge" / "pseudos"

cs = parse_upf(FIX / "Cs_ONCV_PBE_sr.upf")
pb = parse_upf(DG / "Pb.upf")
iod = parse_upf(FIX / "I_ONCV_PBE_sr.upf")
cl = parse_upf(FIX / "Cl_ONCV_PBE_sr.upf")

A = 6.29
cell = A * np.eye(3)
frac = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
pos = frac @ cell
species = [0, 1, 2, 2, 2]
X = {2: cl, 3: cl, 4: cl}
ECUT, KM = 30 * RY, (2, 2, 2)
SCF_KW = dict(smearing="none", etol=1e-9, rhotol=1e-9, max_iter=300, verbose=False)


def converge(lam):
    return scf(setup_alchemical_substitution(cell, pos, [cs, pb, iod], species, X,
                                             lam, ecut=ECUT, kmesh=KM,
                                             use_symmetry=False), PBE(), **SCF_KW)


def main():
    lam = 0.5
    res = converge(lam)
    gap = _gap(res.eigenvalues, res.occupations, res.nspin)
    d_gap = alchemical_gap_gradient(res, PBE()).dgap
    diel = dielectric_born(res, PBE())
    d_eps = alchemical_dielectric_gradient(converge, PBE(), lam, h=0.05)

    print(f"CsPb(I,Cl)3 at λ=0.5  (0=CsPbI3, 1=CsPbCl3)")
    print(f"  band gap     E_g   = {gap:.4f} eV     dE_g/dλ   = {d_gap:+.4f} eV")
    print(f"  dielectric   ε∞_iso = {diel['eps_iso']:.4f}        dε∞/dλ    = "
          f"{d_eps['d_eps_iso']:+.4f}")
    print()
    print("  composition-to-property gradient (I -> Cl):")
    print(f"    gap {'widens' if d_gap > 0 else 'narrows'} "
          f"({d_gap:+.3f} eV), dielectric {'rises' if d_eps['d_eps_iso'] > 0 else 'falls'} "
          f"({d_eps['d_eps_iso']:+.3f}) — the expected opposite-sign optoelectronic trade-off"
          if d_gap * d_eps['d_eps_iso'] < 0 else
          f"    dGap/dλ={d_gap:+.3f}, dε∞/dλ={d_eps['d_eps_iso']:+.3f}")


if __name__ == "__main__":
    main()
