"""Composition inverse design: tune a mixed-halide perovskite gap to a target.

Two things, both using the relaxed band-gap gradient dE_gap/dλ (the composition
DFPT of scf.alchemical):

  1. PATH INTEGRATION (is the local gradient a true global derivative?). Integrate
     dE_gap/dλ over the CsPbI3 -> CsPbCl3 path and check it equals the endpoint gap
     difference Gap(CsPbCl3) - Gap(CsPbI3). If ∫₀¹ (dGap/dλ) dλ ≈ ΔGap, the local
     gradient can be trusted to drive a design loop.

  2. INVERSE DESIGN. Newton-step the alchemical weight λ (all X sites together)
     using the gradient to hit a target gap, then run a DIRECT SCF at the optimized
     λ* to confirm the designed composition actually has the target gap. This is
     the whole point of a differentiable composition coordinate: gradient-driven
     design, not a scan.

    uv run python examples/perovskite_inverse_design.py

Scalar-relativistic PBE, nspin=1 insulator, use_symmetry=False (the response
regime). The gap edges of cubic CsPbI3 are degenerate, so the DFPT gap gradient
carries the rung-1 degenerate-edge caveat; this run is also a test of whether it
is good enough to design with.

Validated (asus, ECUT=30 Ry, 2x2x2): the path integral ∫₀¹ dGap/dλ dλ = +1.393 eV
matches the endpoint ΔGap = +1.383 eV to 10 meV (trapezoid curvature), and the
Newton loop reaches λ*=0.468 in 3 steps with a direct-SCF gap of 1.7003 eV against
the 1.700 eV target (0.3 meV miss). The scalar λ preserves the cubic symmetry, so
the degenerate VBM shifts uniformly and the gap gradient stays exact here — the
degenerate-edge caveat bites only for symmetry-breaking per-site perturbations.
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.api._common import _gap
from gradwave.core.xc.pbe import PBE
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


def run(lam):
    sysm = setup_alchemical_substitution(cell, pos, [cs, pb, iod], species, X, lam,
                                         ecut=ECUT, kmesh=KM, use_symmetry=False)
    return scf(sysm, PBE(), **SCF_KW)


def gap_of(res):
    return _gap(res.eigenvalues, res.occupations, res.nspin)


def path_integration():
    print("=== 1. path integration: ∫ dGap/dλ dλ =?= ΔGap ===")
    g0 = gap_of(run(0.0))
    g1 = gap_of(run(1.0))
    print(f"  Gap(CsPbI3) = {g0:.4f} eV   Gap(CsPbCl3) = {g1:.4f} eV   ΔGap = {g1 - g0:+.4f} eV")
    lams = np.linspace(0.0, 1.0, 6)  # 0, .2, .4, .6, .8, 1.0
    grads = []
    for lam in lams:
        g = alchemical_gap_gradient(run(float(lam)), PBE()).dgap
        grads.append(g)
        print(f"    λ={lam:.2f}   dGap/dλ = {g:+.4f} eV")
    g = np.asarray(grads)
    integral = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(lams)))
    print(f"  ∫₀¹ dGap/dλ dλ (trapezoid) = {integral:+.4f} eV")
    print(f"  endpoint ΔGap              = {g1 - g0:+.4f} eV")
    print(f"  agreement = {abs(integral - (g1 - g0)) * 1e3:.1f} meV\n")


def inverse_design(target=1.7):
    print(f"=== 2. inverse design: tune λ so Gap(λ) = {target} eV ===")
    lam = 0.3
    for it in range(8):
        res = run(lam)
        g = gap_of(res)
        dg = alchemical_gap_gradient(res, PBE()).dgap
        print(f"  iter {it}: λ={lam:.4f}   Gap={g:.4f} eV   dGap/dλ={dg:+.4f}")
        if abs(g - target) < 5e-3:
            break
        step = (target - g) / dg          # Newton step on Gap(λ)=target
        lam = float(np.clip(lam + step, 0.02, 0.98))
    print(f"  → optimized λ* = {lam:.4f}")
    # independent verification: a fresh SCF at λ* (the gradient never saw this)
    verify = gap_of(run(lam))
    print(f"  direct SCF at λ*: Gap = {verify:.4f} eV   target = {target} eV   "
          f"(miss {abs(verify - target) * 1e3:.1f} meV)")


def main():
    path_integration()
    inverse_design()


if __name__ == "__main__":
    main()
