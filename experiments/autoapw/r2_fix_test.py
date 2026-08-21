"""Decisive test of the missing-/r^2 hypothesis for the EFG magnitude deficit.

sphere_density_multipoles_bands builds rho_2M from |u|^2 (u=rR) without the /r^2
that _sphere_valence_density applies, so rho_2M^gw = r^2 * rho_true. Downstream
_valence_v computes (4pi E2/5) \int rho_2M/r dr, i.e. \int r*rho_true dr instead of
\int rho_true/r dr. Fix = divide the captured rho_2M by r^2. If the on-site O EFG
jumps ~-3.1 -> ~Elk -10.3, the bug is confirmed.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/home/wladerer/gw-efg/src")
sys.path.insert(0, "/home/wladerer/gw-efg/experiments/autoapw")
from _common import A_BOHR, ATOMS, RADII  # noqa: E402

import gradwave.flapw.scf as scfmod  # noqa: E402
from gradwave.flapw import crystal_scf_multi  # noqa: E402
from gradwave.flapw.efg import efg_tensor, efg_tensor_full, interstitial_l2_boundary  # noqa: E402

TAG = os.environ.get("PR_TAG", "aug4")
LMAX = {"base": 3, "aug4": 4, "aug5": 5}[TAG]
cfg = dict(ecut=300.0, smearing=0.0, fullpot=True, fullpot_lmax=4, use_symmetry=True,
           subspace_reuse=False, kworkers=1, kerker=0.7, lmax=LMAX, los=None)

cap = {}
_orig = scfmod._efg_from_multipoles


def _wrap(rho_by_key, v_grid, acart, keys, R_by_key, rr_by_key, dx, A, vbc_own=None):
    cap.update(rho=rho_by_key, rr=rr_by_key, dx=dx, keys=keys, R=R_by_key, vg=v_grid,
               acart=acart, A=A, vbc_own=vbc_own)
    return _orig(rho_by_key, v_grid, acart, keys, R_by_key, rr_by_key, dx, A, vbc_own=vbc_own)


scfmod._efg_from_multipoles = _wrap
state = pickle.load(open(f"/home/wladerer/tio2_states/{TAG}_k222.pkl", "rb"))
crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                  v_start={"__full_state__": state}, kmesh=(2, 2, 2), **cfg)


def eigs(t):
    w = np.linalg.eigvalsh(t)
    return w[np.argsort(-np.abs(w))]


np.set_printoptions(precision=3, suppress=True)
print(f"=== {TAG}: missing-/r^2 EFG test (Elk on-site O=-10.27, full O=-19.1) ===")
for key, name in (("a0", "Ti1"), ("a2", "O1")):
    rr = cap["rr"][key]
    drw = rr * cap["dx"]
    R = cap["R"][key]
    rho = cap["rho"][key]
    rho_fix = {lm: v / rr ** 2 for lm, v in rho.items()}
    # on-site
    _, vzz0, e0 = efg_tensor(rho, rr, drw)
    _, vzz1, e1 = efg_tensor(rho_fix, rr, drw)
    # full (recompute boundary; lattice term unchanged by the fix)
    vbc = interstitial_l2_boundary(cap["vg"], dict(zip(cap["keys"], [a[0] for a in cap["acart"]]))[key]
                                   if False else cap["acart"][cap["keys"].index(key)][0], R, cap["A"])
    own = cap["vbc_own"][cap["keys"].index(key)] if cap["vbc_own"] else {}
    vbc = {m: vbc[m] - own.get((2, m), 0.0) for m in range(-2, 3)}
    _, vf0, ef0 = efg_tensor_full(rho, rr, drw, vbc, R)
    _, vf1, ef1 = efg_tensor_full(rho_fix, rr, drw, vbc, R)
    print(f"\n{name}:")
    print(f"  on-site  current  Vzz={vzz0:+.3f} eta={e0:.3f}")
    print(f"  on-site  /r^2 fix Vzz={vzz1:+.3f} eta={e1:.3f}")
    print(f"  full     current  Vzz={vf0:+.3f} eta={ef0:.3f}")
    print(f"  full     /r^2 fix Vzz={vf1:+.3f} eta={ef1:.3f}   (NOTE: lattice term itself"
          " also needs the fix in the Weinert moments)")
    # small-r scaling check of |rho_2M|
    P = np.sqrt(sum(np.abs(rho[(2, m)]) ** 2 for m in range(-2, 3)))
    Pf = np.sqrt(sum(np.abs(rho_fix[(2, m)]) ** 2 for m in range(-2, 3)))
    i1, i2 = np.searchsorted(rr, 0.05), np.searchsorted(rr, 0.15)
    n_cur = np.log(P[i2] / P[i1]) / np.log(rr[i2] / rr[i1])
    n_fix = np.log(Pf[i2] / Pf[i1]) / np.log(rr[i2] / rr[i1])
    print(f"  small-r scaling exponent: current r^{n_cur:.2f}, /r^2 fix r^{n_fix:.2f} (Elk r^2)")
