"""gradwave FLAPW EFG for ANATASE TiO2 (I4_1/amd, primitive 6-atom cell) vs Elk + experiment.

Mirrors the rutile olo_accept.py recipe: aug4/fp4/k222, ecut300, kerker=0.7, shift-invert,
smearing=0 (insulator), conditioned O 2s LO (los={"O":[(0,"2s")]}, el_override O l=0 -> 2p).
Warm-continues to the aspherical gate (r_nsph<1e-3, r_v<rv_gate), persists state, then the EFG
from the exact finalize pass. Reports per site: full Cartesian tensor, |V|-sorted principal
values + eigenvectors, eta, V_zz, C_Q; and the c-axis (Cartesian z) projection.

Env: ECUT LMAX FPLMAX KWORKERS MAXIT RV ITERS KERKER OLO OUT STATE.
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg

from gradwave.flapw.nmr import quadrupolar_coupling

# --- Anatase TiO2 primitive cell (ASE sg141 setting1, a=3.7842 c=9.5146 A, O z=0.2081) ---
# lattice rows in Bohr; Ti-O min bond 1.934 A (radii chosen non-overlapping).
CELL_BOHR = np.array([[-3.575551, 3.575551, 8.989993],
                      [3.575551, -3.575551, 8.989993],
                      [3.575551, 3.575551, -8.989993]])
# Ti first (a0,a1), then O (a2..a5). Fractional coords in the primitive basis.
ATOMS = [((0.000000, 0.000000, 0.000000), "Ti"),
         ((0.750000, 0.250000, 0.500000), "Ti"),
         ((0.208100, 0.208100, 0.000000), "O"),
         ((0.958100, 0.458100, 0.500000), "O"),
         ((0.541900, 0.041900, 0.500000), "O"),
         ((0.791900, 0.791900, 0.000000), "O")]
RADII = {"Ti": float(os.environ.get("RTI", "1.06")), "O": float(os.environ.get("RO", "0.824"))}

AXES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}   # Cartesian; z = crystallographic c


def cfg_from_env():
    olo = int(os.environ.get("OLO", "1"))
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(
        ecut=float(os.environ.get("ECUT", "300")),
        lmax=int(os.environ.get("LMAX", "4")),
        fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")),
        smearing=0.0, use_symmetry=True, subspace_reuse=False,
        kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
        shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")),
    )
    if olo:
        c["los"] = {"O": [(0, "2s")]}
        c["el_override"] = {"O": {0: "2p"}}
    return c, olo


def report_site(site, name, isotope, elk=None):
    t = np.array(site["tensor"])
    w, vecs = np.linalg.eigh(t)
    order = np.argsort(-np.abs(w))
    w = w[order]
    vecs = vecs[:, order]
    proj = {ax: float(n @ t @ n) for ax, n in AXES.items()}
    cq = quadrupolar_coupling(site["V_zz"], site["eta"], isotope)
    print(f"{name} FULL: eig=[{w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}] eta={site['eta']:.3f} "
          f"V_zz={site['V_zz']:+.3f} C_Q({isotope})={cq['abs_C_Q_MHz']:.3f} MHz", flush=True)
    print(f"     proj a/b/c = {proj['a']:+.3f}/{proj['b']:+.3f}/{proj['c']:+.3f}  "
          f"V_zz eigvec=[{vecs[0,0]:+.2f},{vecs[1,0]:+.2f},{vecs[2,0]:+.2f}]", flush=True)
    print(f"     tensor rows: [{t[0,0]:+.3f},{t[0,1]:+.3f},{t[0,2]:+.3f}] "
          f"[{t[1,0]:+.3f},{t[1,1]:+.3f},{t[1,2]:+.3f}] "
          f"[{t[2,0]:+.3f},{t[2,1]:+.3f},{t[2,2]:+.3f}]", flush=True)
    print(f"     ONSITE(valence): V_zz={site['V_zz_valence']:+.3f} eta={site['eta_valence']:.3f}",
          flush=True)
    if elk is not None:
        print(f"     Elk {name}: {elk}", flush=True)


def main():
    cfg, olo = cfg_from_env()
    out = os.environ.get("OUT", os.path.expanduser(f"~/efg_multimat/anatase_OLO{olo}.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# ANATASE TiO2 OLO={olo} ecut={cfg['ecut']} lmax={cfg['lmax']} fp{cfg['fullpot_lmax']} "
          f"k222 kerker={cfg['kerker']} R_Ti={RADII['Ti']} R_O={RADII['O']} "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=rv_gate)
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT ANATASE {status} ({time.time()-t0:.0f}s) state: {out} ===", flush=True)
        report_site(ie["efg"]["a0"], "Ti", "49Ti",
                    "Elk: V_zz=-9.48 eta=0.00 (axial c) C_Q(49Ti)=5.66 MHz")
        report_site(ie["efg"]["a2"], "O", "17O",
                    "Elk: V_zz=-12.42 eta=0.096 C_Q(17O)=0.77 MHz")
    except Exception as e:      # noqa: BLE001
        print(f"=== FAILED ANATASE after {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===",
              flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
