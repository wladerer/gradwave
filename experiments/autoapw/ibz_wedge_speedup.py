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

import os
import sys
import time

import numpy as np
import torch

# make ``tests.helpers`` (Si fixture) importable when run as a plain script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.postscf.kgeometry_nmr import (  # noqa: E402
    _plan_wedge_reduction,
    sigma_shielding_dq,
)
from gradwave.scf.loop import scf, setup_system  # noqa: E402
from tests.helpers import RY, si_fcc, si_upf  # noqa: E402


def _res(cell, pos, kmesh, ecut=6, fft=(15, 15, 15)):
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=ecut * RY,
                          kmesh=kmesh, nbands=8, use_symmetry=False, fft_shape=fft)
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


# measured setup/solve wall split of one sigma_shielding_dq pass (scoping §2,
# Si 4³ profile): the k-reduction speedup is 1/(fs/r_setup + fv/r_solve).
_FS, _FV = 0.0811, 0.9189


def _plan(res):
    k_frac = np.stack([s.k_frac for s in res.system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    return axes, _plan_wedge_reduction(res, axes, "auto")


def _factor(res):
    """Exact k-count-derived net speedup (per-k cost is constant at fixed ecut,
    so wall ∝ nk; scoping measured this linearity at 0.99)."""
    axes, plan = _plan(res)
    nk_full = len(res.system.spheres)  # the TR-folded mesh the route pays
    if plan is None:
        return nk_full, "-", 1.0
    union, per_axis = plan
    wedges = [len(p[2]) for p in per_axis]
    r_setup = nk_full / len(union)
    r_solve = (nk_full * len(axes)) / sum(wedges)
    net = 1.0 / (_FS / r_setup + _FV / r_solve)
    return nk_full, f"union={len(union)} axes={wedges}", net


def _wall(res):
    t0 = time.perf_counter()
    sig_full = sigma_shielding_dq(res, use_symmetry=False)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    sig_red = sigma_shielding_dq(res, use_symmetry="auto")
    t_red = time.perf_counter() - t0
    scale = float(sig_full.abs().max())
    resid = float((sig_red - sig_full).abs().max())
    return t_full, t_red, resid, scale


def main() -> None:
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    strain = np.diag([1.0, 1.0, 1.06])
    cell_t, pos_t = cell @ strain, pos @ strain
    pos_low = pos.copy()
    pos_low[1] += np.array([0.31, 0.17, 0.09])

    # measure=True runs the two sigma passes (slow); measure=False reports only
    # the exact k-count-derived factor (instant).
    jobs = [
        ("Si (Oh)", cell, pos, (4, 4, 4), True),
        ("Si (Oh)", cell, pos, (6, 6, 6), True),
        ("tet-Si (D4h)", cell_t, pos_t, (4, 4, 4), True),
        ("tet-Si (D4h)", cell_t, pos_t, (6, 6, 6), False),
        ("low-sym (C1)", cell, pos_low, (4, 4, 4), True),
    ]
    print(f"{'system':14s} {'mesh':8s} {'nk_full':7s} {'wedge':22s} "
          f"{'k-factor':>8s} {'full[s]':>9s} {'red[s]':>9s} {'measured':>8s} "
          f"{'rel_resid':>10s}")
    for name, c, p, km, meas in jobs:
        res = _res(c, p, km)
        nkf, nkw, net = _factor(res)
        if meas:
            tf, tr, resid, scale = _wall(res)
            row = f"{tf:9.2f} {tr:9.2f} {tf / tr:8.2f} {resid / scale:10.2e}"
        else:
            row = f"{'-':>9s} {'-':>9s} {'-':>8s} {'-':>10s}"
        print(f"{name:14s} {str(km):8s} {nkf:<7d} {str(nkw):22s} {net:8.2f} {row}")


if __name__ == "__main__":
    main()
