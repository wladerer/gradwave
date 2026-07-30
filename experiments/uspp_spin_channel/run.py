"""USPP/PAW spin-channel mixing driver.

Tests whether the Stoner magnetization-channel preconditioner already wired
into scf_uspp (``spin_precond=True``, scf/spin_precond.py) rescues the fcc Ni
PAW stagnation the ao-density-seed study localized to the mixer's
magnetization channel. Settings mirror that study exactly so the numbers are
directly comparable: gaussian width 0.1 eV, etol 1e-6, rhotol 1e-5,
mixing_alpha 0.3, max_iter 120, start_mag matrix 0.02/0.05/0.10/0.30.

Systems:
  ni_paw  fcc Ni, Ni.pbe-spn-kjpaw PAW, ecut 50 Ry, 4x4x4, ecutrho 400 Ry  (the target)
  fe_paw  bcc Fe, Fe.pbe-spn-kjpaw PAW, ecut 50 Ry, 6x6x6, ecutrho 400 Ry  (must-not-regress)
  ni_nc   fcc Ni, PD_Ni_PBE NC, ecut 45 Ry, 6x6x6                          (sanity anchor)

Each row runs the same system twice, spin_precond off then on, and prints one
JSON line per (start_mag, spin_precond) with n_iter, converged, F, moment.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from gradwave.pseudo.upf import parse_upf
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY, pseudo


def sys_ni_nc(ecut=45, kmesh=(6, 6, 6), a=3.52):
    upf = parse_upf(pseudo("PD_Ni_PBE.upf"))
    c = a / 2.0
    cell = np.array([[0.0, c, c], [c, 0.0, c], [c, c, 0.0]])
    system = setup_system(cell, np.zeros((1, 3)), [0], [upf], ecut=ecut * RY,
                          kmesh=kmesh, nbands=18, time_reversal=False)
    return ("nc", system)


def sys_ni_paw(ecut=50, kmesh=(4, 4, 4), a=3.52):
    paw = parse_upf_paw(pseudo("Ni.pbe-spn-kjpaw_psl.1.0.0.UPF"))
    c = a / 2.0
    cell = np.array([[0.0, c, c], [c, 0.0, c], [c, c, 0.0]])
    system = setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=ecut * RY,
                        kmesh=kmesh, ecutrho=400 * RY, nbands=18)
    return ("paw", system)


def sys_fe_paw(ecut=50, kmesh=(6, 6, 6), a=2.87):
    paw = parse_upf_paw(pseudo("Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"))
    c = a / 2.0
    cell = np.array([[-c, c, c], [c, -c, c], [c, c, -c]])
    system = setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=ecut * RY,
                        kmesh=kmesh, ecutrho=400 * RY, nbands=18)
    return ("paw", system)


BUILDERS = {"ni_nc": sys_ni_nc, "ni_paw": sys_ni_paw, "fe_paw": sys_fe_paw}


def _xc():
    from gradwave.core.xc.spin import SpinPBE
    return SpinPBE()


def run_one(formalism, system, *, start_mag, spin_precond, smearing, width, xc,
            etol, rhotol, mixing_alpha, max_iter):
    kw = dict(nspin=2, start_mag=[start_mag], smearing=smearing, width=width,
              etol=etol, rhotol=rhotol, mixing_alpha=mixing_alpha,
              max_iter=max_iter, verbose=False)
    t0 = time.perf_counter()
    if formalism == "nc":
        # the NC loop has no spin_precond in main; run it plain (anchor)
        res = scf(system, xc, **kw)
    else:
        res = scf_uspp(system, xc, spin_precond=spin_precond, **kw)
    wall = time.perf_counter() - t0

    def get(name):
        try:
            return getattr(res, name)
        except AttributeError:
            return res[name]

    hist = get("history")
    diag = []
    try:
        rec = getattr(res, "recorder", None)
        if rec is not None:
            diag = [tag for (tag, _reason) in rec.diagnose()]
    except (AttributeError, TypeError):
        diag = []
    return {
        "n_iter": int(get("n_iter")),
        "converged": bool(get("converged")),
        "F_eV": float(get("energies").free_energy),
        "res_final": float(hist[-1]["res"]) if hist else float("nan"),
        "mag_total": float(get("mag_total")),
        "mag_abs": float(get("mag_abs")),
        "wall_s": round(wall, 1),
        "diag": diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=list(BUILDERS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--start-mags", default="0.02,0.05,0.1,0.3")
    ap.add_argument("--precond", default="off,on",
                    help="comma list from {off,on}; nc ignores 'on'")
    ap.add_argument("--smearing", default="gaussian")
    ap.add_argument("--width", type=float, default=0.1)
    ap.add_argument("--etol", type=float, default=1e-6)
    ap.add_argument("--rhotol", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--max-iter", type=int, default=120)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    xc = _xc()
    sm_list = [float(x) for x in args.start_mags.split(",")]
    pc_list = [p.strip() for p in args.precond.split(",")]
    for sm in sm_list:
        for pc in pc_list:
            sp = pc == "on"
            formalism, system = BUILDERS[args.system]()  # fresh per run
            if formalism == "nc" and sp:
                continue  # no spin_precond on the NC path; off-only anchor
            out = run_one(formalism, system, start_mag=sm, spin_precond=sp,
                          smearing=args.smearing, width=args.width, xc=xc,
                          etol=args.etol, rhotol=args.rhotol,
                          mixing_alpha=args.alpha, max_iter=args.max_iter)
            print(json.dumps({"system": args.system, "start_mag": sm,
                              "spin_precond": pc, **out}), flush=True)


if __name__ == "__main__":
    main()
