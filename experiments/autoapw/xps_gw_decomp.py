# ruff: noqa: E402, E501  # scratch probe
"""Stage 0.5: gradwave O1s decomposition with the SAME first-order estimator as
``xps_elk_decomp.py`` so the gw and Elk columns are apples-to-apples.

For each O sphere of the converged Ti+2O FLAPW state:
  own_es(r)  = radial_poisson_to_R(rho_sph) - Z e^2/r     (own electrostatic)
  v_full(r)  = st.v_by_key[k]  restricted to the sphere    (full MT potential gw solves the core in)
  V_ext(r)   = v_full - (own_es + vxc)                     (gw's external field inside the sphere)
  Delta_own  = <w| d(own_es+vxc) >,  Delta_ext = <w| dV_ext >,  total = Delta_own + Delta_ext
with w = |u_1s|^2 from the own-potential solve. Also prints C0_ext (onsite_madelung_potentials)
and the raw re-solved eigenvalue shift for continuity with the prior stage.

Usage (asus):  uv run python experiments/autoapw/xps_gw_decomp.py [ecut=200]
"""
import math
import sys

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, E2
from gradwave.flapw import scf as S
from gradwave.flapw.core_levels import (
    core_levels_from_state,
    onsite_madelung_potentials,
)
from gradwave.flapw.coulomb import radial_poisson_to_R
from gradwave.flapw.functionals import vxc_lda
from gradwave.flapw.radial import radial_eigs_tridiag

LL = 13.0
BOHR = 0.529177210903


def run(r_o_bohr, ecut):
    ti = np.array([0.5, 0.5, 0.5]) * LL
    o_short = ti + np.array([1.75 / BOHR, 0.0, 0.0])
    o_long = ti + np.array([0.0, 2.30 / BOHR, 0.0])
    atoms = [(tuple(ti / LL), "Ti"), (tuple(o_short / LL), "O"), (tuple(o_long / LL), "O")]
    ctx = S._multi_setup(a_bohr=float(LL), atoms=atoms, radii={"Ti": 0.90, "O": r_o_bohr * BOHR},
                         ecut=ecut, kmesh=(1, 1, 1), smearing=0.10, use_symmetry=False, fullpot=False)
    st = S._multi_init_state(ctx, None)
    for it in range(60):
        if S._multi_iterate(ctx, st, it, iters=60, tol=3e-3):
            break
    ti_tau = np.asarray(st.spheres[ctx.keys.index(
        next(k for i, k in enumerate(ctx.keys) if ctx.syms[i] == "Ti"))]["tau"])
    o = [k for i, k in enumerate(ctx.keys) if ctx.syms[i] == "O"]
    sk = sorted(o, key=lambda k: float(np.linalg.norm(
        np.asarray(st.spheres[ctx.keys.index(k)]["tau"]) - ti_tau)))
    return ctx, st, sk[0], sk[1]


def fields(ctx, st, k, Z=8.0):
    sp = st.spheres[ctx.keys.index(k)]
    rr = np.asarray(sp["rr"], dtype=float)          # Angstrom
    dx = float(sp["dx"])
    R = float(sp["R"])
    rho = np.asarray(sp["rho_sph"], dtype=float)     # e/Angstrom^3
    drw = rr * dx
    own_es = radial_poisson_to_R(rho, rr, R, drw) - Z * E2 / rr
    vxc = vxc_lda(torch.tensor(rho)).numpy()
    mask = ctx.mask_by_key[k]
    v_full = np.asarray(st.v_by_key[k])[mask]        # aligned to sp rr order
    v_ext = v_full - (own_es + vxc)
    q_sph = float(np.sum(4 * math.pi * rho * rr**2 * drw))
    return dict(rr=rr, dx=dx, R=R, own_es=own_es, vxc=vxc, v_full=v_full, v_ext=v_ext, q_sph=q_sph)


def weight(f):
    _, u = radial_eigs_tridiag(0, torch.tensor(f["rr"]), f["dx"], torch.tensor(f["own_es"] + f["vxc"]), 1)
    w = np.asarray(u[:, 0])**2 * (f["rr"] * f["dx"])
    return w / w.sum()


def main():
    ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
    print(f"=== gradwave O1s decomposition (ecut={ecut}) ===")
    for r_bohr in (0.70, 1.00, 1.40):
        ctx, st, short, long = run(r_bohr, ecut)
        v_mad = onsite_madelung_potentials(st.v_hart, st.spheres, ctx.keys, ctx.A)
        cl = core_levels_from_state(st.v_by_key, ctx.syms, ctx.keys, ctx.core_map, ctx.r, ctx.dx)
        fs, fl = fields(ctx, st, short), fields(ctx, st, long)
        w = weight(fs)
        d_own = float(np.sum(w * ((fl["own_es"] + fl["vxc"]) - (fs["own_es"] + fs["vxc"]))))
        d_ext = float(np.sum(w * (fl["v_ext"] - fs["v_ext"])))
        rb = fs["R"] / BOHR_ANG
        raw = cl[long]["levels"]["1s"] - cl[short]["levels"]["1s"]
        vext_prof_s = [float(np.interp(x, fs["rr"], fs["v_ext"])) for x in (0.05, 0.2, 0.5*fs["R"], fs["R"])]
        vext_prof_l = [float(np.interp(x, fl["rr"], fl["v_ext"])) for x in (0.05, 0.2, 0.5*fl["R"], fl["R"])]
        print(f"\n-- O R_MT = {rb:.2f} Bohr  q_l0 s/l = {fs['q_sph']:.4f}/{fl['q_sph']:.4f} --")
        print(f"   V_ext short (r=.05,.2,R/2,R Ang-frac): {['%+.3f'%v for v in vext_prof_s]}")
        print(f"   V_ext long : {['%+.3f'%v for v in vext_prof_l]}   d(l-s):{['%+.3f'%(a-b) for a,b in zip(vext_prof_l,vext_prof_s)]}")
        print(f"   own      = {d_own:+.4f}")
        print(f"   external = {d_ext:+.4f}   (C0_ext(onsite_madelung) d = {v_mad[long]-v_mad[short]:+.4f})")
        print(f"   total    = {d_own + d_ext:+.4f}   (raw re-solved eig shift = {raw:+.4f})")


if __name__ == "__main__":
    main()
