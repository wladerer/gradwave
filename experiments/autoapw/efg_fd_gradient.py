"""Finite-difference reference for dV_zz/du and dV_zz/d(c/a) of rutile TiO2 (EFG).

The ground-truth gradient every FLAPW-EFG autograd claim is checked against. Reconverges
the fullpot FLAPW EFG at the base geometry and at the internal parameter u -> u +/- delta
(and delta/2) and the axial ratio c/a -> (c/a) +/- delta (and delta/2), then central-
differences the Ti and O principal component V_zz (and the full Cartesian tensor). The
delta/2 pass is the linear-regime convergence check.

Perturbations preserve the P4_2/mnm symmetry (u is the internal free coordinate; c/a is the
axial ratio at fixed a), so use_symmetry stays valid and the Anderson map stays contractive.

Run on asus. Warm-starts every perturbed point from the base converged full state so each
lands near the fixed point, then converges TIGHTLY (small tol) for a reproducible tensor.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi

# Rutile TiO2 (P4_2/mnm), experimental geometry.
U0 = 0.3048
A = 8.68083          # a = b (Bohr)
C0 = 5.59096         # c (Bohr)
RADII = {"Ti": 1.098, "O": 0.824}

# config knobs
ECUT = float(sys.argv[1]) if len(sys.argv) > 1 else 250.0
LMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 3
FP_LMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 4
KK = int(sys.argv[4]) if len(sys.argv) > 4 else 2
SMEAR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0  # 0 essential: TiO2 is an insulator
DELTA = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3
ITERS = int(sys.argv[7]) if len(sys.argv) > 7 else 80
TOL = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-6
KW = int(sys.argv[9]) if len(sys.argv) > 9 else 4
MODE = sys.argv[10] if len(sys.argv) > 10 else "fp"  # "fp" (fullpot) or "mt" (muffin-tin)
KERKER = float(sys.argv[11]) if len(sys.argv) > 11 else 0.7  # interstitial screen (Å⁻¹); the
# campaign's validated convergence recipe — plain Anderson on TiO2 k222 is chaotically fragile.

FULLPOT = MODE == "fp"
KMESH = (KK, KK, KK)


def atoms_for(u):
    return [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
            ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
            ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]


def run(u, c, tag="", v_sph=None):
    # Warm-start each perturbed point from the base's SPHERICAL potentials (per-key dict, NOT
    # __full_state__): a per-key v_start keeps mt_phase=True (re-stages the muffin-tin loop) while
    # seeding the spherical potential near converged — this biases every point to the base's basin
    # (the fullpot k222 fixed point is chaotically multistable; cold runs land in different basins,
    # a __full_state__ warm start skips staging and diverges). v_sph=None => cold (the base).
    a_bohr = [A, A, c]
    atoms = atoms_for(u)
    t0 = time.time()
    bands, info = crystal_scf_multi(
        a_bohr, atoms, RADII, ecut=ECUT, lmax=LMAX, iters=ITERS, tol=TOL,
        kmesh=KMESH, smearing=SMEAR, efg=True, fullpot=FULLPOT, fullpot_lmax=FP_LMAX,
        kworkers=KW, v_start=v_sph, kerker=(KERKER if KERKER > 0 else None), verbose=False)
    dt = time.time() - t0
    rec = info.get("recorder")
    niter = len(rec.iters) if rec is not None else -1
    last = rec.iters[-1] if rec is not None else {}
    r_v = last.get("r_v")
    r_nsph = last.get("r_nsph")
    conv = (r_v is not None and r_v < 1e-3 and (r_nsph is None or r_nsph < 1e-2))
    ti = info["efg"]["a0"]
    o = info["efg"]["a2"]
    print(f"  [{tag}] u={u:.5f} c/a={c / A:.6f} : {dt:.0f}s niter={niter} "
          f"r_v={r_v:.1e} r_nsph={r_nsph} CONV={conv} "
          f"Ti.Vzz={ti['V_zz']:+.5f} O.Vzz={o['V_zz']:+.5f} "
          f"symdev={info.get('symmetry_dev'):.1e}", flush=True)
    return info


def vzz(info, site):
    return float(info["efg"][site]["V_zz"])


def tensor(info, site):
    return np.asarray(info["efg"][site]["tensor"], dtype=float)


def central(fp, fm, h):
    return (fp - fm) / (2 * h)


def main():
    print(f"CONFIG mode={MODE} ecut={ECUT} lmax={LMAX} fp_lmax={FP_LMAX} kmesh={KMESH} "
          f"smear={SMEAR} kerker={KERKER} delta={DELTA} iters={ITERS} tol={TOL} "
          f"kworkers={KW}", flush=True)
    print("=== base (cold) ===", flush=True)
    base = run(U0, C0, tag="base")
    vsph = {k: np.asarray(v).copy() for k, v in base["v_by_key"].items()}  # base spherical seed

    d1 = DELTA
    d2 = DELTA / 2

    print("=== u sweep (warm from base spherical) ===", flush=True)
    up1 = run(U0 + d1, C0, tag="u+d", v_sph=vsph)
    um1 = run(U0 - d1, C0, tag="u-d", v_sph=vsph)
    up2 = run(U0 + d2, C0, tag="u+d/2", v_sph=vsph)
    um2 = run(U0 - d2, C0, tag="u-d/2", v_sph=vsph)

    print("=== c/a sweep (warm; a fixed, vary c) ===", flush=True)
    # c/a -> (c/a) +/- delta  <=>  c -> c +/- delta*A  (a fixed)
    cp1 = run(U0, C0 + d1 * A, tag="ca+d", v_sph=vsph)
    cm1 = run(U0, C0 - d1 * A, tag="ca-d", v_sph=vsph)
    cp2 = run(U0, C0 + d2 * A, tag="ca+d/2", v_sph=vsph)
    cm2 = run(U0, C0 - d2 * A, tag="ca-d/2", v_sph=vsph)

    print("\n===== FINITE-DIFFERENCE GRADIENTS =====", flush=True)
    for site, lbl in (("a0", "Ti"), ("a2", "O")):
        # dV_zz/du
        g1 = central(vzz(up1, site), vzz(um1, site), d1)
        g2 = central(vzz(up2, site), vzz(um2, site), d2)
        # Richardson (4th order) extrapolation from the two central diffs
        gR = (4 * g2 - g1) / 3
        rel = abs(g1 - g2) / max(abs(g2), 1e-12)
        print(f"[{lbl}] dV_zz/du     : delta={d1:.1e} -> {g1:+.4f}   "
              f"delta/2 -> {g2:+.4f}   Richardson -> {gR:+.4f}   "
              f"rel|d-d/2|={rel:.2e}  (eV/Ang^2 per unit u)", flush=True)
        # dV_zz/d(c/a)
        h1 = central(vzz(cp1, site), vzz(cm1, site), d1)
        h2 = central(vzz(cp2, site), vzz(cm2, site), d2)
        hR = (4 * h2 - h1) / 3
        relc = abs(h1 - h2) / max(abs(h2), 1e-12)
        print(f"[{lbl}] dV_zz/d(c/a) : delta={d1:.1e} -> {h1:+.4f}   "
              f"delta/2 -> {h2:+.4f}   Richardson -> {hR:+.4f}   "
              f"rel|d-d/2|={relc:.2e}  (eV/Ang^2 per unit c/a)", flush=True)
        # full-tensor central difference (delta) — dT/du and dT/d(c/a)
        dT_du = central(tensor(up1, site), tensor(um1, site), d1)
        dT_dca = central(tensor(cp1, site), tensor(cm1, site), d1)
        np.set_printoptions(precision=3, suppress=True)
        print(f"[{lbl}] dTensor/du (eV/Ang^2 per u):\n{dT_du}", flush=True)
        print(f"[{lbl}] dTensor/d(c/a):\n{dT_dca}", flush=True)

    # also dump absolute V_zz at every point for the record
    print("\n===== ABSOLUTE V_zz TABLE =====", flush=True)
    pts = [("base", base), ("u+d", up1), ("u-d", um1), ("u+d/2", up2), ("u-d/2", um2),
           ("ca+d", cp1), ("ca-d", cm1), ("ca+d/2", cp2), ("ca-d/2", cm2)]
    for tag, info in pts:
        print(f"  {tag:8s} Ti.Vzz={vzz(info, 'a0'):+.6f}  O.Vzz={vzz(info, 'a2'):+.6f}  "
              f"Ti.eta={info['efg']['a0']['eta']:.4f}  O.eta={info['efg']['a2']['eta']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
