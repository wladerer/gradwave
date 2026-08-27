"""Continuity-closure ecut scan for the MgO analytic-USPP smooth current.

For each ecut rung, run one axis' +q velocity Sternheimer solve (USPP ctx) and
evaluate `uspp_smooth_continuity`: the identity

    q.mean[j_kin + j_nl] = mean s + (augmentation current + truncation).

The remainder R = (q_j_kin + q_j_nl) - source is the on-site augmentation-charge
current plus basis truncation. If |R| GROWS with ecut, the missing augmentation
current is ecut-divergent (a physical missing term); if |R| is small/stable the
smooth continuity closes and the shielding divergence lives elsewhere.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

MG = sys.argv[1] if len(sys.argv) > 1 else "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF"
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL


def run(ecut_ry: float, ecutrho_ry: float):
    from gradwave.postscf import kgeometry_nmr as kg

    torch.set_num_threads(5)
    paw_mg = parse_upf_paw(MG)
    paw_o = parse_upf_paw(O)
    system = setup_uspp(CELL, POS, [0, 1], [paw_mg, paw_o], ecut=ecut_ry * RY,
                        kmesh=(2, 2, 2), ecutrho=ecutrho_ry * RY, nbands=12)
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=80)
    assert res["converged"]
    ctx = kg.build_uspp_response_ctx(res, PBE())

    b = np.linalg.inv(CELL).T * 2 * np.pi
    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axis = next(i for i in range(3) if mesh_n[i] > 1)
    q_frac = np.zeros(3)
    q_frac[axis] = 1.0 / mesh_n[axis]
    q_cart = q_frac @ b
    qh = q_cart / np.linalg.norm(q_cart)
    # transverse polarization
    a0 = np.zeros(3)
    a0[int(np.argmin(np.abs(qh)))] = 1.0
    t1 = a0 - (a0 @ qh) * qh
    t1 /= np.linalg.norm(t1)

    sol = kg.velocity_perturbation_q(res, q_frac, cg_tol=1e-9, max_iter=800, uspp=ctx)
    d = kg.uspp_smooth_continuity(res, ctx, sol, t1)
    src, jk, jnl = d["source"], d["q_j_kin"], d["q_j_nl"]
    remainder = (jk + jnl) - src
    print(f"\n=== ecut {ecut_ry}/{ecutrho_ry} Ry  shape={tuple(system.grid.shape)} "
          f"npw~{system.spheres[0].miller.shape[0]} ===")
    print(f"  source        = {src:.6e}")
    print(f"  q.j_kin       = {jk:.6e}")
    print(f"  q.j_nl        = {jnl:.6e}")
    print(f"  q.(j_kin+j_nl)= {jk + jnl:.6e}")
    print(f"  REMAINDER (aug+trunc) = {remainder:.6e}  |R|={abs(remainder):.4e}")
    return abs(remainder)


if __name__ == "__main__":
    rs = []
    for e, er in [(40.0, 160.0), (60.0, 360.0)]:
        rs.append(run(e, er))
    print(f"\n|remainder| ecut-scaling: {rs[0]:.4e} -> {rs[1]:.4e}  "
          f"(ratio {rs[1] / rs[0]:.2f})")
