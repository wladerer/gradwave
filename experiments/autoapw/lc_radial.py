"""Dump gw's B-sphere l=2 density rho_2M(r) via a warm efg pass (monkeypatch), then compare the
cumulative on-site V_zz buildup vs radius against Elk's rhomt l=2 (localizes interior vs near-R
tail). src-free: monkeypatches the experiment-side call only.

MAT=hbn (B, key a0) or li3n (Li1, key a0). Warm-starts from ~/efg_lc/<MAT>.pkl.
Writes ~/efg_lc/<MAT>_gw_rho2.npz with rr, R, and rho2m (dict m->array).
"""
import math
import os
import pickle
import sys

import numpy as np
import gradwave.flapw.scf as gscf
from gradwave.constants import E2
from _efgrun import converge_efg  # noqa: F401
from _mgroup import hex_cell_bohr  # noqa: F401 (inject species)

MAT = os.environ.get("MAT", "hbn")
if MAT == "hbn":
    CELL = hex_cell_bohr(2.504, 6.661)
    ATOMS = [((1 / 3, 2 / 3, 1 / 4), "B"), ((2 / 3, 1 / 3, 3 / 4), "B"),
             ((2 / 3, 1 / 3, 1 / 4), "N"), ((1 / 3, 2 / 3, 3 / 4), "N")]
    RADII = {"B": 0.70, "N": 0.70}
    KEY, ECUT_D, ANION = "a0", 400.0, "N"
else:
    CELL = hex_cell_bohr(3.648, 3.875)
    ATOMS = [((0.0, 0.0, 0.5), "Li"), ((1 / 3, 2 / 3, 0.0), "Li"),
             ((2 / 3, 1 / 3, 0.0), "Li"), ((0.0, 0.0, 0.0), "N")]
    RADII = {"Li": 0.90, "N": 0.90}
    KEY, ECUT_D, ANION = "a0", 250.0, "N"

_orig = gscf._efg_from_multipoles


def _patched(rho_by_key, v_hart, acart, keys, R_by_key, rr_by_key, dx, A, qmt_by_sphere=None):
    k = keys[int(KEY[1:])]
    rho = rho_by_key[k]
    rr = np.asarray(rr_by_key[k])
    out = {"rr": rr, "R": float(R_by_key[k]), "dx": float(dx)}
    for m in range(-2, 3):
        out[f"rho2_{m+2}"] = np.asarray(rho[(2, m)])
    np.savez(os.path.expanduser(f"~/efg_lc/{MAT}_gw_rho2.npz"), **out)
    print(f"# dumped gw rho2 for key {k} R={out['R']:.4f} nr={len(rr)}", flush=True)
    return _orig(rho_by_key, v_hart, acart, keys, R_by_key, rr_by_key, dx, A, qmt_by_sphere)


gscf._efg_from_multipoles = _patched


def cfg():
    los = {ANION: [(0, "2s")]}
    return dict(ecut=ECUT_D, lmax=4, fullpot=True, fullpot_lmax=4, smearing=0.0,
                use_symmetry=True, subspace_reuse=False, kerker=0.7, shift_invert=True,
                kworkers=int(os.environ.get("KWORKERS", "4")), los=los,
                el_override={ANION: {0: "2p"}})


def main():
    state = pickle.load(open(os.path.expanduser(f"~/efg_lc/{MAT}.pkl"), "rb"))
    _, ie = gscf.crystal_scf_multi(CELL, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                                   kmesh=(2, 2, 2), v_start={"__full_state__": state}, **cfg())
    s = ie["efg"][KEY]
    # cumulative on-site V_zz buildup vs r (M=0 dominates for the axial c-site)
    d = np.load(os.path.expanduser(f"~/efg_lc/{MAT}_gw_rho2.npz"))
    rr, R, dx = d["rr"], float(d["R"]), float(d["dx"])
    drw = rr * dx
    rho20 = d["rho2_2"]  # M=0
    integ = (rho20.real / rr) * drw
    cum = np.cumsum(integ)
    v0_full = (4 * math.pi * E2 / 5.0) * cum
    vzz = math.sqrt(5.0 / math.pi) * v0_full  # cumulative V_zz(r) contribution (M=0 part)
    print(f"gw {MAT} {KEY}: on-site V_zz(full)={s['V_zz_valence']:+.3f} "
          f"M0-only cum(R)={vzz[-1]:+.3f}", flush=True)
    for frac in (0.25, 0.5, 0.7, 0.85, 0.95, 1.0):
        idx = np.searchsorted(rr, frac * R)
        idx = min(idx, len(rr) - 1)
        print(f"   r/R={frac:.2f} (r={rr[idx]:.4f}b): cumV_zz={vzz[idx]:+.3f} "
              f"({100*vzz[idx]/vzz[-1]:.1f}% of M0 total)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
