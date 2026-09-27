"""MgO ¹⁷O bare-shielding ecut convergence: dense-eigh vs matrix-free CG backend.

The analytic-USPP bare ¹⁷O magnetic shielding (`sigma_shielding_dq` through the
S-metric context) DIVERGED with plane-wave ecut on the hard-augmentation O
dataset: its dense-eigh S-orthonormal conduction resolvent (`_resolvent_apply_s`)
expands the ∂/∂q response in the ill-conditioned S-orthonormal eigenbasis of the
augmentation overlap S(k), which goes overcomplete (cond(S) → 10³–10⁴) as ecut
grows, breaking the S-completeness cancellation. The shipped matrix-free S-metric
CG Sternheimer backend (`_SMetricResolventCG`, selected by
`response_backend="auto"` above cond(S) > 50) never forms that eigenbasis.

This script runs the SAME MgO ground state across an ecut ladder TWICE — the old
DENSE resolvent (`response_backend="dense"`, reproduce the divergence) and the CG
resolvent (`response_backend="auto"`) — and records σ_iso(¹⁷O) bare + total for
each, to test whether CG PLATEAUS where dense DIVERGES.

PAW route (the divergence + fix live on the S-metric path; the NC/ONCV route has
S = I and does not diverge). Pseudos: Mg PAW (argv[1]) + committed
`O.pbe-n-kjpaw_psl.1.0.0.UPF` (the hard first-row anion that drives cond(S)).
Run on asus via gwq, OMP/torch threads = 8.
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.postscf import kgeometry_nmr as kg
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

MG = sys.argv[1] if len(sys.argv) > 1 else "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF"
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"
OUT = sys.argv[2] if len(sys.argv) > 2 else "mgo_17o_cg_convergence.json"

# MgO rocksalt, a = 4.21 Angstrom, 2-atom primitive (fcc). Mg at 0, O at (1/2).
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL

# ecut(wfc) ladder at fixed ecutrho = 4x ratio and fixed 2x2x2 k-mesh, so only
# the plane-wave cutoff varies. 40/160 matches the documented divergence rung.
RUNGS = [(40.0, 160.0), (50.0, 200.0), (60.0, 240.0), (70.0, 280.0)]
KMESH = (2, 2, 2)
NBANDS = 12
BACKENDS = ["dense", "auto"]  # (a) old dense-eigh, (b) CG (auto routes >cond 50)


def build(ecut_ry: float, ecutrho_ry: float):
    torch.set_num_threads(8)
    paw_mg = parse_upf_paw(MG)
    paw_o = parse_upf_paw(O)
    system = setup_uspp(
        CELL, POS, [0, 1], [paw_mg, paw_o],
        ecut=ecut_ry * RY, kmesh=KMESH, ecutrho=ecutrho_ry * RY, nbands=NBANDS,
    )
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=120)
    assert res["converged"], f"SCF not converged at {ecut_ry}/{ecutrho_ry}"
    return [paw_mg, paw_o], res


def iso(t3: torch.Tensor) -> float:
    return float(torch.diagonal(t3).mean())


def main() -> None:
    records = []
    t0 = time.time()
    for ecut, ecutrho in RUNGS:
        paws, res = build(ecut, ecutrho)
        ctx = kg.build_uspp_response_ctx(res, PBE())
        system = ctx.system
        cond = kg.uspp_overlap_conditioning(ctx)
        npw = int(system.spheres[0].miller.shape[0])
        shape = tuple(int(x) for x in system.grid.shape)
        cond_max = float(cond["max"])
        routes_cg = cond_max > float(kg._USPP_COND_CG_THRESHOLD)
        print(f"\n=== ecut {ecut:.0f}/{ecutrho:.0f} Ry  grid={shape}  npw~{npw}  "
              f"cond(S)max={cond_max:.0f}  (auto->{'cg' if routes_cg else 'dense'}, "
              f"thr={kg._USPP_COND_CG_THRESHOLD}) ===", flush=True)
        for backend in BACKENDS:
            tb = time.time()
            # One full GIPAW assembly per (rung, backend): out["bare"] IS the
            # sigma_shielding_dq bare term (the diverging quantity), out["total"]
            # the anchor. No separate bare call — it would double the response
            # solves (the gipaw path already runs sigma_shielding_dq internally).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = kg.sigma_shielding_gipaw(
                    res, ctx, paws, use_symmetry=False, response_backend=backend,
                    cg_tol=1e-10, max_iter=600)
            bare = [iso(out["bare"][a]) for a in range(2)]
            total = [iso(out["total"][a]) for a in range(2)]
            terms = {k: [iso(out[k][a]) for a in range(2)]
                     for k in ("bare", "core", "dia_aug", "para_aug")}
            dt = time.time() - tb
            rec = {
                "ecut": ecut, "ecutrho": ecutrho, "grid": shape, "npw": npw,
                "cond_S_max": cond_max, "backend": backend,
                "auto_routes_cg": routes_cg,
                "bare_Mg": bare[0], "bare_O": bare[1],
                "total_Mg": total[0], "total_O": total[1],
                "terms": terms, "seconds": dt,
            }
            records.append(rec)
            print(f"  [{backend:5s}] bare O = {bare[1]:+.2f}  total O = {total[1]:+.2f} "
                  f"| bare Mg = {bare[0]:+.2f}  total Mg = {total[0]:+.2f}  "
                  f"({dt:.0f}s)", flush=True)
            with open(OUT, "w") as f:
                json.dump({"pseudo_mg": MG, "pseudo_o": O, "cell_a": A,
                           "kmesh": list(KMESH), "nbands": NBANDS,
                           "records": records}, f, indent=2)
    print(f"\nWROTE {OUT}  total {time.time() - t0:.0f}s", flush=True)
    print("MGO_17O_CG_CONVERGENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
