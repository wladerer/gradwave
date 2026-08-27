"""Measure the CG warm-start iteration saving on the MgO ¹⁷O bare shielding.

Builds one MgO PAW ground state (40/160 Ry) and runs the analytic ∂/∂q bare
shielding assembly (:class:`ShieldingDq`, CG backend) twice — cold (every CG
solve from zero) and warm (each solve seeded from the adjacent ∂/∂q stage:
previous q̂ pol → δu⁰, δu⁰ → δu¹) — over the sampled mesh axes, summing the CG
iteration counts (``eng._cg_iters``). Reports total, mean, and the reduction.

Run on asus, OMP_NUM_THREADS=5. Mg semicore PAW is argv[1].
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.grids import reciprocal_cell
from gradwave.postscf.kgeometry_nmr import (
    ShieldingDq,
    _transverse_frame,
    build_uspp_response_ctx,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

MG = sys.argv[1]
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL


def axis_specs(system):
    b = reciprocal_cell(system.grid.cell)
    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    out = []
    for i in axes:
        qh = np.asarray(b[i], dtype=float) / np.linalg.norm(b[i])
        out.append((qh, list(_transverse_frame(qh)), None, None, None))
    return out


def total_iters(res, ctx, specs, *, warm_start: bool) -> list[int]:
    eng = ShieldingDq(res, uspp=ctx, response_backend="cg", warm_start=warm_start,
                      cg_tol=1e-10, max_iter=600)
    eng.branch_fields_all_axes(specs)
    return eng._cg_iters


def main() -> None:
    torch.set_num_threads(5)
    system = setup_uspp(CELL, POS, [0, 1], [parse_upf_paw(MG), parse_upf_paw(O)],
                        ecut=40 * RY, kmesh=(2, 2, 2), ecutrho=160 * RY, nbands=12)
    res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
                   verbose=False, max_iter=80)
    assert res["converged"]
    ctx = build_uspp_response_ctx(res, PBE())
    specs = axis_specs(ctx.system)

    cold = total_iters(res, ctx, specs, warm_start=False)
    warm = total_iters(res, ctx, specs, warm_start=True)
    n = len(cold)
    tc, tw = sum(cold), sum(warm)
    print(f"CG solves: {n}", flush=True)
    print(f"cold: total={tc}  mean={tc / n:.1f}  max={max(cold)}", flush=True)
    print(f"warm: total={tw}  mean={tw / n:.1f}  max={max(warm)}", flush=True)
    # signed change of warm relative to cold: positive => warm is SLOWER (more
    # iterations), negative => warm saves. Measured positive on MgO (a null).
    delta = tw - tc
    print(f"warm - cold: {delta:+d} iters  ({100.0 * delta / tc:+.1f}%; "
          f"{'warm SLOWER' if delta > 0 else 'warm saves'})", flush=True)
    print("WARMSTART_DONE", flush=True)


if __name__ == "__main__":
    main()
