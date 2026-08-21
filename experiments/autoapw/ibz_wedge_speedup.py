"""Measured wall-clock speedup of the little-group-of-q IBZ wedge reduction of
the ANALYTIC NMR shielding route (``sigma_shielding_dq``), and its route
equivalence to the full-mesh path.

For each (system, mesh) we run one SCF then time ``sigma_shielding_dq`` twice:
``use_symmetry=False`` (the full TR-folded mesh — the oracle) and
``use_symmetry="auto"`` (the opt-in wedge reduction). The reported speedup is
``wall_full / wall_reduced``; the residual ``‖σ_red − σ_full‖_max`` is the
route-equivalence proof. The k-count reduction is ecut-independent (per-k cost
is constant at fixed ecut), so a modest ecut gives the same factor as the
production ecut at a fraction of the wall.

Systems:
  - Si (diamond, Oh) — the GO case; expect ~2.7×/3.4× at 4³/6³.
  - tetragonal Si (I4₁/amd, D4h; [001]-strained diamond) — the rutile-class
    anisotropic-axis stand-in (real rutile TiO₂ needs a Ti pseudo absent from
    the NC fixtures); expect the smaller ~1.5×/2.0× band.
  - low-symmetry (a displaced atom, trivial point group) — the auto-fallback:
    the planner declines, both calls run the identical full mesh, so the ratio
    is ~1.0 (never slower).

Run on asus:  OMP_NUM_THREADS=2 uv run python experiments/autoapw/ibz_wedge_speedup.py
"""
from __future__ import annotations

import time

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.kgeometry_nmr import _plan_wedge_reduction, sigma_shielding_dq
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf


def _res(cell, pos, kmesh, ecut=6, fft=(15, 15, 15)):
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=ecut * RY,
                          kmesh=kmesh, nbands=8, use_symmetry=False, fft_shape=fft)
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


def _time(res):
    k_frac = np.stack([s.k_frac for s in res.system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    plan = _plan_wedge_reduction(res, axes, "auto")
    if plan is None:
        nk_wedge = "-"
    else:
        union, per_axis = plan
        nk_wedge = f"union={len(union)} axes={[len(p[2]) for p in per_axis]}"

    t0 = time.perf_counter()
    sig_full = sigma_shielding_dq(res, use_symmetry=False)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    sig_red = sigma_shielding_dq(res, use_symmetry="auto")
    t_red = time.perf_counter() - t0

    scale = float(sig_full.abs().max())
    resid = float((sig_red - sig_full).abs().max())
    return len(res.system.spheres), nk_wedge, t_full, t_red, resid, scale


def main() -> None:
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    strain = np.diag([1.0, 1.0, 1.06])
    cell_t, pos_t = cell @ strain, pos @ strain
    pos_low = pos.copy()
    pos_low[1] += np.array([0.31, 0.17, 0.09])

    jobs = [
        ("Si (Oh)", cell, pos, (4, 4, 4)),
        ("Si (Oh)", cell, pos, (6, 6, 6)),
        ("tet-Si (D4h)", cell_t, pos_t, (4, 4, 4)),
        ("tet-Si (D4h)", cell_t, pos_t, (6, 6, 6)),
        ("low-sym (C1)", cell, pos_low, (4, 4, 4)),
    ]
    print(f"{'system':14s} {'mesh':8s} {'nk_full':7s} {'wedge':24s} "
          f"{'full[s]':>9s} {'red[s]':>9s} {'speedup':>7s} {'resid_ppm':>10s} {'rel':>9s}")
    for name, c, p, km in jobs:
        res = _res(c, p, km)
        nkf, nkw, tf, tr, resid, scale = _time(res)
        print(f"{name:14s} {str(km):8s} {nkf:<7d} {str(nkw):24s} "
              f"{tf:9.2f} {tr:9.2f} {tf / tr:7.2f} {resid:10.2e} {resid / scale:9.2e}")


if __name__ == "__main__":
    main()
