"""Definitive /r^2 check: compare the l=0 multipole density rho_00 (from
sphere_density_multipoles_bands) against gradwave's OWN spherical SCF density
rho_sph (known-correct: it makes the potential the SCF converges to). If
rho_00/sqrt(4pi) == rho_sph only after dividing by r^2, the multipole build carries
a spurious r^2. If it matches as-is, there is no r^2 bug and the deficit is basis.
"""
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/home/wladerer/gw-efg/src")
sys.path.insert(0, "/home/wladerer/gw-efg/experiments/autoapw")
from _common import A_BOHR, ATOMS, RADII  # noqa: E402

import gradwave.flapw.scf as scfmod  # noqa: E402
from gradwave.flapw import crystal_scf_multi  # noqa: E402

TAG = os.environ.get("PR_TAG", "aug4")
LMAX = {"base": 3, "aug4": 4, "aug5": 5}[TAG]
cfg = dict(ecut=300.0, smearing=0.0, fullpot=True, fullpot_lmax=4, use_symmetry=True,
           subspace_reuse=False, kworkers=1, kerker=0.7, lmax=LMAX, los=None)

cap = {}
_oe = scfmod._efg_from_multipoles
_ow = scfmod._weinert_multi


def _we(rho_by_key, v_grid, acart, keys, R_by_key, rr_by_key, dx, A, vbc_own=None):
    cap.update(rho=rho_by_key, rr=rr_by_key)
    return _oe(rho_by_key, v_grid, acart, keys, R_by_key, rr_by_key, dx, A, vbc_own=vbc_own)


def _ww(rho_I, spheres, L, nfft):
    cap["spheres"] = spheres
    return _ow(rho_I, spheres, L, nfft)


scfmod._efg_from_multipoles = _we
scfmod._weinert_multi = _ww
state = pickle.load(open(f"/home/wladerer/tio2_states/{TAG}_k222.pkl", "rb"))
crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                  v_start={"__full_state__": state}, kmesh=(2, 2, 2), **cfg)

keys = list(cap["rho"].keys())
sph_by_key = {f"a{i}": s for i, s in enumerate(cap["spheres"])}
for key, name in (("a0", "Ti1"), ("a2", "O1")):
    rr = cap["rr"][key]
    rho00 = cap["rho"][key][(0, 0)].real
    rho_sph_true = sph_by_key[key]["rho_sph"]  # includes core; valence-only differs but l=0 valence
    # rho_00/sqrt(4pi) should equal the VALENCE spherical density. Core is added separately in
    # rho_sph, so compare shapes on the valence-dominated outer region via the ratio.
    val00 = rho00 / math.sqrt(4 * math.pi)
    print(f"\n=== {name}: is rho_00 == rho_sph_valence, or r^2*that? ===")
    print("  r(bohr)   rho00/sqrt4pi   rho_sph(val+core)   (rho00/s4pi)/r^2   ratio_asis   ratio_/r2")
    for frac in (0.3, 0.5, 0.7, 0.9):
        i = int(frac * (len(rr) - 1))
        rb = rr[i] / 0.52917721067
        a = val00[i]
        c = a / rr[i] ** 2
        print(f"   {rb:.4f}   {a:+.5e}     {rho_sph_true[i]:+.5e}      {c:+.5e}   "
              f"{a/rho_sph_true[i]:+.4f}     {c/rho_sph_true[i]:+.4f}")
    # integrated valence charge two ways
    drw = rr * (rr[1] / rr[0] and np.log(rr[1] / rr[0]))  # dx
    dx = math.log(rr[1] / rr[0])
    drw = rr * dx
    q_asis = math.sqrt(4 * math.pi) * float(np.sum(rho00 * rr ** 2 * drw))
    q_fix = math.sqrt(4 * math.pi) * float(np.sum((rho00 / rr ** 2) * rr ** 2 * drw))
    print(f"  integral sqrt4pi*∫rho00 r^2 dr : as-is={q_asis:.3f} e ,  /r^2={q_fix:.3f} e")
