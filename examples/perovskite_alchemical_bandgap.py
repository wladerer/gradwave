"""Alchemical band-gap design in a halide perovskite (CsPbI3 -> CsPbCl3).

The X-site halides of cubic CsPbI3 are transmuted toward Cl with an alchemical
weight lambda, holding the Cs (A-site) and Pb (B-site) sublattices fixed. Because
composition is now a differentiable coordinate, the sensitivity of the fundamental
gap to substitution, d(E_gap)/d(lambda), is read from a small finite difference of
full SCFs -- a composition-to-property design gradient at (a few) SCF cost.

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
from gradwave.scf.alchemical import setup_alchemical_substitution
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
SCF_KW = dict(smearing="none", etol=1e-8, rhotol=1e-8, max_iter=200, verbose=False)


def gap_of(system):
    res = scf(system, PBE(), **SCF_KW)
    return (float(res.energies.free_energy),
            _gap(res.eigenvalues, res.occupations, res.nspin),
            bool(res.converged))


def alch(lam):
    return setup_alchemical_substitution(
        cell, pos, pseudos, species, X_SITES, lam, ecut=ECUT, kmesh=KMESH)


def main():
    # endpoint reproduction: alchemical lam=0 == pure CsPbI3, lam=1 == pure CsPbCl3
    e_i, gap_i, _ = gap_of(setup_system(cell, pos, species, pseudos, ECUT, KMESH))
    e_a0, gap_a0, _ = gap_of(alch(0.0))
    e_cl, gap_cl, _ = gap_of(setup_system(cell, pos, species, [cs, pb, cl], ECUT, KMESH))
    e_a1, gap_a1, _ = gap_of(alch(1.0))
    print(f"CsPbI3    : E = {e_i:.6f} eV   gap = {gap_i}")
    print(f"  alch(0) : E = {e_a0:.6f} eV   gap = {gap_a0}   "
          f"|dE| = {abs(e_a0 - e_i) * 1e3:.2e} meV")
    print(f"CsPbCl3   : E = {e_cl:.6f} eV   gap = {gap_cl}")
    print(f"  alch(1) : E = {e_a1:.6f} eV   gap = {gap_a1}   "
          f"|dE| = {abs(e_a1 - e_cl) * 1e3:.2e} meV")

    # band-gap design gradient d(gap)/d(lambda_Cl) by a forward finite difference
    h = 0.1
    _, gap_h, _ = gap_of(alch(h))
    dgap = (gap_h - gap_a0) / h
    print(f"\nd(E_gap)/d(lambda_Cl) ~ {dgap:+.3f} eV   (I->Cl should widen the gap, i.e. > 0)")
    print(f"endpoint gap change    {gap_cl - gap_i:+.3f} eV   (CsPbCl3 - CsPbI3)")


if __name__ == "__main__":
    main()
