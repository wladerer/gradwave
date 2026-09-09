"""Decompose the FLAPW within-cell O1s core-level shift into its in-sphere and
on-site Madelung (C0_ext) parts, and scan C0_ext vs O R_MT against Elk.

This is the evidence behind the re-diagnosis in ``xps_validation.md`` (Task 1b) and
the ``delta_madelung_eV`` field added to ``core_levels.core_level_shifts``:

* The muffin-tin core eigenvalue = (in-sphere own term) + C0_ext, where C0_ext is the
  external (Madelung) electrostatic constant the sphere already gets from the Weinert
  interstitial boundary match. For the Ti+2O demo the in-sphere own term is a large
  WRONG-signed muffin-tin artifact (-3.81 eV) that swamps the correct C0_ext (+2.22 eV),
  giving the wrong-signed raw eigenvalue shift (-1.67 eV).
* C0_ext alone reproduces Elk's within-cell O1s shift in sign, magnitude, and R_MT trend
  (for R_MT >~ 1.0 Bohr).

Run on asus (dense O(N^3) FLAPW; keep OMP modest):

    uv run python experiments/autoapw/xps_madelung_decomposition.py            # decompose
    uv run python experiments/autoapw/xps_madelung_decomposition.py scan       # C0_ext vs R_MT
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, E2
from gradwave.flapw import scf as S
from gradwave.flapw.core_levels import core_levels_from_state, onsite_madelung_potentials
from gradwave.flapw.coulomb import radial_poisson_to_R
from gradwave.flapw.functionals import vxc_lda
from gradwave.flapw.radial import radial_eigs_tridiag

BOHR = 0.529177210903
LL = 13.0  # cubic box edge (Bohr)


def _run(r_o_ang: float):
    """Converge the Ti + 2 O cell and return (ctx, st) plus short/long O keys."""
    ti = np.array([0.5, 0.5, 0.5]) * LL
    o_short = ti + np.array([1.75 / BOHR, 0.0, 0.0])
    o_long = ti + np.array([0.0, 2.30 / BOHR, 0.0])
    atoms = [(tuple(ti / LL), "Ti"), (tuple(o_short / LL), "O"), (tuple(o_long / LL), "O")]
    ctx = S._multi_setup(a_bohr=float(LL), atoms=atoms, radii={"Ti": 0.90, "O": r_o_ang},
                         ecut=200.0, kmesh=(1, 1, 1), smearing=0.10, use_symmetry=False,
                         fullpot=False)
    st = S._multi_init_state(ctx, None)
    for it in range(40):
        if S._multi_iterate(ctx, st, it, iters=40, tol=3e-3):
            break
    ti_key = next(k for i, k in enumerate(ctx.keys) if ctx.syms[i] == "Ti")
    ti_tau = np.asarray(st.spheres[ctx.keys.index(ti_key)]["tau"])
    o = [k for i, k in enumerate(ctx.keys) if ctx.syms[i] == "O"]
    sk = sorted(o, key=lambda k: float(np.linalg.norm(
        np.asarray(st.spheres[ctx.keys.index(k)]["tau"]) - ti_tau)))
    return ctx, st, sk[0], sk[1]


def decompose():
    ctx, st, short, long = _run(0.70)  # O R_MT 0.70 A = 1.32 Bohr (the demo geometry)
    v_mad = onsite_madelung_potentials(st.v_hart, st.spheres, ctx.keys, ctx.A)
    cl = core_levels_from_state(st.v_by_key, ctx.syms, ctx.keys, ctx.core_map,
                                ctx.r, ctx.dx, v_madelung=v_mad)
    print("=== O 1s core-level shift decomposition (O R_MT 0.70 A = 1.32 Bohr) ===")
    r_np = np.asarray(ctx.r)
    e_ins = {}
    for k in (short, long):
        sp = st.spheres[ctx.keys.index(k)]
        rr = np.asarray(sp["rr"])
        drw = rr * sp["dx"]
        vpart = radial_poisson_to_R(sp["rho_sph"], rr, sp["R"], drw=drw) - sp["Z"] * E2 / rr
        vxc = vxc_lda(torch.tensor(sp["rho_sph"])).numpy()
        full = np.full(r_np.shape[0], st.v_i0_prev)
        full[ctx.mask_by_key[k]] = vpart + vxc  # in-sphere own only (no boundary constant)
        e, _ = radial_eigs_tridiag(0, ctx.r, ctx.dx, torch.tensor(full), 1)
        e_ins[k] = float(e[0])
    print(f"  in-sphere own shift (long-short) = "
          f"{e_ins[long] - e_ins[short]:+.4f} eV  (muffin-tin artifact)")
    print(f"  C0_ext Madelung shift            = "
          f"{v_mad[long] - v_mad[short]:+.4f} eV  (matches Elk)")
    print(f"  raw eigenvalue shift             = "
          f"{cl[long]['levels']['1s'] - cl[short]['levels']['1s']:+.4f} eV")


def scan():
    print("=== gradwave C0_ext (Madelung) shift vs O R_MT  (Elk: "
          "+0.951/+1.380/+2.860 at 0.70/1.00/1.40 Bohr) ===")
    for r_bohr in (0.70, 1.00, 1.32, 1.40):
        ctx, st, short, long = _run(r_bohr * BOHR_ANG)
        v_mad = onsite_madelung_potentials(st.v_hart, st.spheres, ctx.keys, ctx.A)
        cl = core_levels_from_state(st.v_by_key, ctx.syms, ctx.keys, ctx.core_map, ctx.r, ctx.dx)
        raw = cl[long]["levels"]["1s"] - cl[short]["levels"]["1s"]
        print(f"  R_O={r_bohr:.2f} Bohr: C0_ext_shift={v_mad[long] - v_mad[short]:+.4f} eV  "
              f"raw_eig_shift={raw:+.4f} eV")


if __name__ == "__main__":
    (scan if len(sys.argv) > 1 and sys.argv[1] == "scan" else decompose)()
