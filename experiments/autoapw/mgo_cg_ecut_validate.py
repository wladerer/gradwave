"""Validate the ShieldingDq matrix-free S-metric CG reroute on MgO ¹⁷O.

The dense-eigh analytic-USPP bare shielding DIVERGED with ecut on MgO ¹⁷O
(−1912 → −3758 ppm, 40/160 → 60/360 Ry) because it expanded the ∂/∂q response
in the ill-conditioned S-orthonormal conduction eigenbasis (cond(S) 426 →
3245). :class:`ShieldingDq` now solves the SAME resolvent matrix-free by the
S-metric CG Sternheimer, which never forms that eigenbasis. This script
reproduces the two ecut rungs and reports:

  1. bare σ_iso(Mg, O) at each rung — must now AGREE across ecut (was ×2),
  2. cond(S) over the mesh (now a diagnostic, no longer a divergence predictor),
  3. the full GIPAW total σ_iso(O) — should move toward the ≈ +215 ppm anchor.

Run on asus, OMP_NUM_THREADS=5, one at a time. The Mg semicore PAW pseudo is
argv[1] (e.g. Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.postscf import kgeometry_nmr as kg
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

MG = sys.argv[1] if len(sys.argv) > 1 else "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF"
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"

A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL  # Mg at 0, O at (1/2)

RUNGS = [(40.0, 160.0), (60.0, 400.0)]


def build(ecut_ry: float, ecutrho_ry: float):
    torch.set_num_threads(5)
    paw_mg = parse_upf_paw(MG)
    paw_o = parse_upf_paw(O)
    system = setup_uspp(
        CELL, POS, [0, 1], [paw_mg, paw_o],
        ecut=ecut_ry * RY, kmesh=(2, 2, 2), ecutrho=ecutrho_ry * RY, nbands=12,
    )
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=80)
    assert res["converged"], f"SCF not converged at {ecut_ry}/{ecutrho_ry}"
    return [paw_mg, paw_o], res


def run(ecut_ry: float, ecutrho_ry: float) -> None:
    paws, res = build(ecut_ry, ecutrho_ry)
    ctx = kg.build_uspp_response_ctx(res, PBE())
    system = ctx.system
    cond = kg.uspp_overlap_conditioning(ctx)

    sig_bare = kg.sigma_shielding_dq(res, uspp=ctx, use_symmetry=False,
                                     cg_tol=1e-10, max_iter=600)
    bare = [float(torch.diagonal(sig_bare[a]).mean()) for a in range(2)]

    out = kg.sigma_shielding_gipaw(res, ctx, paws, use_symmetry=False,
                                   cg_tol=1e-10, max_iter=600)
    total = [float(torch.diagonal(out["total"][a]).mean()) for a in range(2)]
    terms = {k: [float(torch.diagonal(out[k][a]).mean()) for a in range(2)]
             for k in ("bare", "core", "dia_aug", "para_aug")}

    print(f"\n=== ecut {ecut_ry:.0f}/{ecutrho_ry:.0f} Ry  "
          f"shape={tuple(system.grid.shape)}  "
          f"npw~{system.spheres[0].miller.shape[0]}  "
          f"cond(S) max={cond['max']:.0f} ===", flush=True)
    print(f"  bare  σ_iso:  Mg = {bare[0]:+.2f}   O = {bare[1]:+.2f} ppm", flush=True)
    for k, v in terms.items():
        print(f"  {k:8s}      Mg = {v[0]:+.2f}   O = {v[1]:+.2f} ppm", flush=True)
    print(f"  TOTAL σ_iso:  Mg = {total[0]:+.2f}   O = {total[1]:+.2f} ppm "
          f"(¹⁷O anchor ≈ +215)", flush=True)


if __name__ == "__main__":
    for e, er in RUNGS:
        run(e, er)
    print("MGO_CG_VALIDATE_DONE", flush=True)
