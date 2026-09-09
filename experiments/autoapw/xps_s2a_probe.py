# ruff: noqa: E402, E501
"""Stage 2a probe: localize the gw C0_ext deficit into vbc0 (interstitial-potential
surface value) vs own_mono (E2(q_sph-Z)/R), per site, and save v_hart for the
gw-vs-Elk interstitial comparison. Single radius (arg), unbuffered, per-iter marker.

Usage (asus): PYTHONUNBUFFERED=1 uv run python xps_s2a_probe.py <r_o_bohr> [ecut=200] [iters=60]
"""
import math
import sys
import time

import numpy as np

from gradwave.constants import BOHR_ANG, E2
from gradwave.flapw import scf as S
from gradwave.flapw.coulomb import (
    cell_matrix,
    gvec_ylm_tables,
    sphere_interstitial_moments,
)
from gradwave.flapw.core_levels import core_levels_from_state, onsite_madelung_potentials
from gradwave.flapw.efg import interstitial_boundary_multi

LL = 13.0
BOHR = 0.529177210903
Y00 = 1.0 / math.sqrt(4.0 * math.pi)


def run(r_o_bohr, ecut, iters):
    ti = np.array([0.5, 0.5, 0.5]) * LL
    o_short = ti + np.array([1.75 / BOHR, 0.0, 0.0])
    o_long = ti + np.array([0.0, 2.30 / BOHR, 0.0])
    atoms = [(tuple(ti / LL), "Ti"), (tuple(o_short / LL), "O"), (tuple(o_long / LL), "O")]
    ctx = S._multi_setup(a_bohr=float(LL), atoms=atoms, radii={"Ti": 0.90, "O": r_o_bohr * BOHR},
                         ecut=ecut, kmesh=(1, 1, 1), smearing=0.10, use_symmetry=False, fullpot=False)
    st = S._multi_init_state(ctx, None)
    t0 = time.time()
    for it in range(iters):
        done = S._multi_iterate(ctx, st, it, iters=iters, tol=3e-3)
        print(f"  [it {it:2d}] t={time.time()-t0:6.1f}s done={done}", flush=True)
        if done:
            break
    return ctx, st


def main():
    r_bohr = float(sys.argv[1])
    ecut = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    print(f"=== gw stage-2a probe  r_O={r_bohr} Bohr  ecut={ecut} nfft-driven ===", flush=True)
    ctx, st = run(r_bohr, ecut, iters)
    A = ctx.A
    nfft = ctx.nfft
    print(f"  nfft={nfft}  cell(A)=\n{np.asarray(A)}", flush=True)

    # identify Ti and the two O (sorted by Ti distance)
    ti_key = next(k for i, k in enumerate(ctx.keys) if ctx.syms[i] == "Ti")
    ti_tau = np.asarray(st.spheres[ctx.keys.index(ti_key)]["tau"])
    o = [k for i, k in enumerate(ctx.keys) if ctx.syms[i] == "O"]
    sk = sorted(o, key=lambda k: float(np.linalg.norm(
        np.asarray(st.spheres[ctx.keys.index(k)]["tau"]) - ti_tau)))
    short, long = sk[0], sk[1]

    v_mad = onsite_madelung_potentials(st.v_hart, st.spheres, ctx.keys, A)
    cl = core_levels_from_state(st.v_by_key, ctx.syms, ctx.keys, ctx.core_map, ctx.r, ctx.dx)

    # rho_I in G-space for interstitial moments
    rho_g = (np.fft.fftn(st.rho_I) / nfft**3).reshape(-1)
    gvec, gnorm, ylm = gvec_ylm_tables(A, nfft, 0)
    vol = float(abs(np.linalg.det(cell_matrix(A))))
    rhoI_tot = float(st.rho_I.mean() * vol)

    rows = {}
    print("\n  --- PER-SITE (eV, Bohr) ---", flush=True)
    print("  site     Z   q_sph    dist_Ti   own_mono    vbc0       C0_ext     q_rhoI_in", flush=True)
    for k in [ti_key, short, long]:
        sp = st.spheres[ctx.keys.index(k)]
        R = float(sp["R"])
        rr = np.asarray(sp["rr"], float)
        drw = rr * float(sp["dx"])
        Z = float(sp["Z"])
        q_sph = float(np.sum(4 * math.pi * np.asarray(sp["rho_sph"], float) * rr**2 * drw))
        own = E2 * (q_sph - Z) / R
        vbc0 = float(interstitial_boundary_multi(st.v_hart, sp["tau"], R, A, [(0, 0)])[(0, 0)].real / math.sqrt(4 * math.pi))
        c0 = vbc0 - own
        qi = sphere_interstitial_moments(rho_g, R, sp["tau"], gvec, gnorm, ylm, [0])[(0, 0)]
        q_rhoI_in = float(qi.real) * math.sqrt(4 * math.pi)
        dist = float(np.linalg.norm(np.asarray(sp["tau"]) - ti_tau)) / BOHR_ANG
        rows[k] = dict(R=R, q_sph=q_sph, own=own, vbc0=vbc0, c0=c0, q_rhoI_in=q_rhoI_in)
        print(f"  {('Ti' if k==ti_key else ('O_sh' if k==short else 'O_lo')):5s} {Z:5.1f} {q_sph:7.4f} {dist:8.4f}  {own:+9.3f}  {vbc0:+9.3f}  {c0:+9.3f}  {q_rhoI_in:+8.4f}", flush=True)

    dq = rows[long]["q_sph"] - rows[short]["q_sph"]
    d_own = rows[long]["own"] - rows[short]["own"]
    d_vbc0 = rows[long]["vbc0"] - rows[short]["vbc0"]
    d_c0 = rows[long]["c0"] - rows[short]["c0"]
    raw = cl[long]["levels"]["1s"] - cl[short]["levels"]["1s"]
    print(f"\n  Δ(long-short):  Δq_sph={dq:+.4f}  Δown_mono={d_own:+.4f}  Δvbc0={d_vbc0:+.4f}  ΔC0_ext={d_c0:+.4f}", flush=True)
    print(f"  onsite_madelung ΔC0_ext={v_mad[long]-v_mad[short]:+.4f}   raw eig Δ1s={raw:+.4f}", flush=True)
    print(f"  rho_I total charge in cell = {rhoI_tot:.4f}", flush=True)

    # save v_hart + geometry for gw-vs-Elk interstitial comparison
    tag = f"r{int(round(r_bohr*100)):03d}"
    np.savez(f"/tmp/gw_vhart_{tag}.npz",
             v_hart=st.v_hart, cell=np.asarray(A), nfft=nfft,
             ti=ti_tau, o_short=np.asarray(st.spheres[ctx.keys.index(short)]["tau"]),
             o_long=np.asarray(st.spheres[ctx.keys.index(long)]["tau"]),
             R_ti=rows[ti_key]["R"], R_o=rows[short]["R"])
    print(f"  saved /tmp/gw_vhart_{tag}.npz", flush=True)
    # pickle the converged Weinert inputs so fix-variants can be tried without re-running SCF
    import pickle
    with open(f"/tmp/gw_state_{tag}.pkl", "wb") as fh:
        pickle.dump(dict(rho_I=st.rho_I, spheres=st.spheres, A=np.asarray(A), nfft=nfft,
                         keys=list(ctx.keys), syms=list(ctx.syms),
                         short=short, long=long, ti_key=ti_key), fh)
    print(f"  saved /tmp/gw_state_{tag}.pkl", flush=True)


if __name__ == "__main__":
    main()
