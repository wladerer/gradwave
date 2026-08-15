"""Alchemical band-gap design in a halide perovskite (CsPbI3 -> CsPbCl3).

The X-site halides of cubic CsPbI3 are transmuted toward Cl with an alchemical
weight lambda, holding the Cs (A-site) and Pb (B-site) sublattices fixed. Because
composition is now a differentiable coordinate, the sensitivity of the fundamental
gap to substitution, d(E_gap)/d(lambda), is a composition-to-property design
gradient computed analytically by the composition DFPT (alchemical_gap_gradient):
the relaxed derivative, with the self-consistent density response, from a single
converged SCF. It is cross-checked here against a central finite difference of
re-converged SCFs, and the frozen (sudden) estimate is shown for contrast -- the
density response, not the bare matrix element, carries the gradient.

Replacing the heavier halide (I) by the lighter one (Cl) widens the gap of a
lead-halide perovskite, so d(E_gap)/d(lambda_Cl) > 0 is the expected, checkable
sign. Scalar-relativistic here (this shows the mechanics); the quantitatively
accurate gap of a Pb perovskite needs spin-orbit coupling.

This also validates the heterogeneous alchemical engine: at lambda=0 the
transmuted system must reproduce a plain CsPbI3 SCF, and at lambda=1 a plain
CsPbCl3 SCF, to numerical noise.

    uv run python examples/perovskite_alchemical_bandgap.py
"""

from pathlib import Path

import numpy as np

from gradwave.api._common import _gap
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import (
    alchemical_gap_gradient,
    setup_alchemical_substitution,
)
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
DG = ROOT / "benchmarks" / "delta_gauge" / "pseudos"

cs = parse_upf(FIX / "Cs_ONCV_PBE_sr.upf")
pb = parse_upf(DG / "Pb.upf")
iodine = parse_upf(FIX / "I_ONCV_PBE_sr.upf")
cl = parse_upf(FIX / "Cl_ONCV_PBE_sr.upf")

A = 6.29  # cubic CsPbI3 lattice constant [Ang]
cell = A * np.eye(3)
frac = np.array([
    [0.0, 0.0, 0.0],   # Cs  (A-site, fixed)
    [0.5, 0.5, 0.5],   # Pb  (B-site, fixed)
    [0.5, 0.5, 0.0],   # I   (X-site)
    [0.5, 0.0, 0.5],   # I   (X-site)
    [0.0, 0.5, 0.5],   # I   (X-site)
])
pos = frac @ cell
species = [0, 1, 2, 2, 2]
pseudos = [cs, pb, iodine]
X_SITES = {2: cl, 3: cl, 4: cl}   # transmute the three I sites toward Cl

ECUT = 40 * RY
KMESH = (2, 2, 2)   # Gamma-centered, so it contains R = (1/2,1/2,1/2), the ABX3 gap
# use_symmetry=False: the composition DFPT (alchemical_gap_gradient) needs the
# full TR-reduced mesh, since the perturbation breaks the crystal symmetry.
SCF_KW = dict(smearing="none", etol=1e-9, rhotol=1e-9, max_iter=300, verbose=False)


def _run(system):
    res = scf(system, PBE(), **SCF_KW)
    return (res, float(res.energies.free_energy),
            _gap(res.eigenvalues, res.occupations, res.nspin))


def alch(lam):
    return setup_alchemical_substitution(
        cell, pos, pseudos, species, X_SITES, lam, ecut=ECUT, kmesh=KMESH,
        use_symmetry=False)


def _base(pseudo_list):
    return setup_system(cell, pos, species, pseudo_list, ECUT, KMESH,
                        use_symmetry=False)


def main():
    # endpoint reproduction: alchemical lam=0 == pure CsPbI3, lam=1 == pure CsPbCl3
    _, e_i, gap_i = _run(_base(pseudos))
    _, e_a0, gap_a0 = _run(alch(0.0))
    _, e_cl, gap_cl = _run(_base([cs, pb, cl]))
    _, e_a1, gap_a1 = _run(alch(1.0))
    print(f"CsPbI3    : E = {e_i:.6f} eV   gap = {gap_i}")
    print(f"  alch(0) : E = {e_a0:.6f} eV   gap = {gap_a0}   "
          f"|dE| = {abs(e_a0 - e_i) * 1e3:.2e} meV")
    print(f"CsPbCl3   : E = {e_cl:.6f} eV   gap = {gap_cl}")
    print(f"  alch(1) : E = {e_a1:.6f} eV   gap = {gap_a1}   "
          f"|dE| = {abs(e_a1 - e_cl) * 1e3:.2e} meV")

    # band-gap design gradient d(E_gap)/d(lambda_Cl), analytically by the
    # composition DFPT (the relaxed derivative, with the self-consistent density
    # response), checked against a central finite difference of re-converged SCF
    # gaps. The frozen (sudden) estimate is printed too, to show how much of the
    # gradient is the density response.
    lam = 0.5
    res, _, gap_m = _run(alch(lam))
    g = alchemical_gap_gradient(res, PBE())
    h = 0.05
    _, _, gap_p = _run(alch(lam + h))
    _, _, gap_n = _run(alch(lam - h))
    fd = (gap_p - gap_n) / (2 * h)
    print(f"\nat lambda={lam}  gap = {gap_m:.4f} eV")
    print(f"d(E_gap)/d(lambda_Cl)  DFPT (relaxed) = {g.dgap:+.4f} eV")
    print(f"                       finite diff     = {fd:+.4f} eV   "
          f"(agree to {abs(g.dgap - fd) * 1e3:.2f} meV)")
    print(f"                       frozen (sudden) = {g.dgap_frozen:+.4f} eV   "
          f"(density response carries the rest)")
    print(f"endpoint gap change    {gap_cl - gap_i:+.3f} eV   (CsPbCl3 - CsPbI3, "
          "I->Cl widens the gap as expected)")


if __name__ == "__main__":
    main()
