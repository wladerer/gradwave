"""Extract gradwave's on-site l=2 density rho_2M(r) + EFG from a gated state, and
form the ratio profile vs Elk's rhomt (elk_rho2m_<label>.npz).

For each state: warm-start a 1-iter efg=True resume, capture rho_by_key from
_efg_from_multipoles, report on-site (efg_tensor) and full tensors for O/Ti, and
ratio the rotation-invariant l=2 density power P(r)=sqrt(sum_M|rho_2M|^2) against
Elk on Elk's mesh (unit+mesh aligned). Uniform ratio -> weight/channel deficit;
r-localized -> basis.
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
from gradwave.flapw.efg import efg_tensor  # noqa: E402

BOHR = 0.52917721067
VARIANTS = {
    "base": dict(lmax=3, los=None),
    "aug4": dict(lmax=4, los=None),
    "aug5": dict(lmax=5, los=None),
    "weinert_aug4_fp6": dict(lmax=4, los=None),
    "weinert_fp6": dict(lmax=3, los=None),
}
TAG = os.environ.get("PR_TAG", "aug4")
STATE = f"/home/wladerer/tio2_states/{TAG}_k222.pkl"
FPLMAX = int(os.environ.get("PR_FPLMAX", "4"))

cfg = dict(ecut=float(os.environ.get("PR_ECUT","300")), smearing=0.0, fullpot=True, fullpot_lmax=FPLMAX,
           use_symmetry=True, subspace_reuse=False, kworkers=1, kerker=0.7,
           **VARIANTS[TAG])

state = pickle.load(open(STATE, "rb"))
cap = {}
_orig = scfmod._efg_from_multipoles


def _wrap(rho_by_key, v_grid, acart, keys, R_by_key, rr_by_key, dx, A, vbc_own=None):
    cap.update(rho=rho_by_key, rr=rr_by_key, dx=dx, keys=keys, R=R_by_key)
    return _orig(rho_by_key, v_grid, acart, keys, R_by_key, rr_by_key, dx, A, vbc_own=vbc_own)


scfmod._efg_from_multipoles = _wrap
_, info = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                            v_start={"__full_state__": state}, kmesh=(2, 2, 2), **cfg)


def eigs(t):
    w = np.linalg.eigvalsh(t)
    return w[np.argsort(-np.abs(w))]


np.set_printoptions(precision=4, suppress=True)
print(f"=== gradwave state {TAG} (k222 resume) ===")
for key, name in (("a0", "Ti1"), ("a2", "O1")):
    rr = cap["rr"][key]
    drw = rr * cap["dx"]
    rho = cap["rho"][key]
    t_on, vzz_on, eta_on = efg_tensor(rho, rr, drw)
    t_full = info["efg"][key]["tensor"]
    print(f"\n{name}:")
    print(f"  on-site  eigs(eV/A^2)={eigs(t_on)}  Vzz={vzz_on:.3f} eta={eta_on:.3f}")
    print(f"  full     eigs(eV/A^2)={eigs(t_full)}  eta={info['efg'][key]['eta']:.3f}")
    # rotation-invariant l=2 density power P(r) = sqrt(sum_M |rho_2M|^2), gradwave units e/A^3
    P_gw = np.sqrt(sum(np.abs(rho[(2, m)]) ** 2 for m in range(-2, 3)))
    # Elk comparison
    elk_label = {"a0": "Ti1", "a2": "O1"}[key]
    npz = f"/home/wladerer/elk_rho2m_{elk_label}.npz"
    if os.path.exists(npz):
        e = np.load(npz)
        rr_e_bohr = e["rr_bohr"]
        rc = e["rho_real_m"]  # (5, nr) real-harmonic coeffs, e/bohr^3
        P_elk = np.sqrt((rc ** 2).sum(0))  # invariant power, e/bohr^3
        rr_gw_bohr = rr / BOHR
        # gradwave power e/A^3 -> e/bohr^3 : *BOHR^3 (density per volume)
        P_gw_b = P_gw * BOHR ** 3
        P_gw_on_e = np.interp(rr_e_bohr, rr_gw_bohr, P_gw_b)
        print("  ratio P_elk(r)/P_gw(r) [invariant l=2 density power]:")
        print("   r(bohr)  r/R    P_elk        P_gw         ratio")
        Rb = cap["R"][key] / BOHR
        for frac in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.98):
            idx = int(frac * (len(rr_e_bohr) - 1))
            re = rr_e_bohr[idx]
            pe, pg = P_elk[idx], P_gw_on_e[idx]
            ratio = pe / pg if pg > 1e-30 else float("nan")
            print(f"   {re:.4f}  {re/Rb:.2f}  {pe:.4e}  {pg:.4e}  {ratio:.3f}")
