"""Density-seed comparison driver.

Two experiments:

  magnetic  branch selection from marginal start_mag, default uniform-split SAD
            seed vs the d-localized magnetization seed (experiments.seed).
  flat      nonmagnetic anchor/audit, current SAD seed vs a deliberately cruder
            flat (uniform) seed, to quantify how much the already-present SAD
            buys and to check Z-consistency of the seed.

Each row runs at identical settings, differing only in rho0. Metrics printed as
JSON lines: n_iter, iter-1 residual, seed build cost, converged moment, energy.
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

from experiments.ao_density_seed import patch


# ---------------------------------------------------------------- systems
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
    cell = np.array([[-c, c, c], [c, -c, c], [c, c, -c]])  # bcc primitive
    system = setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=ecut * RY,
                        kmesh=kmesh, ecutrho=400 * RY, nbands=18)
    return ("paw", system)


def sys_si_nc(ecut=30, kmesh=(4, 4, 4), a=5.43):
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    c = a / 2.0
    cell = np.array([[0.0, c, c], [c, 0.0, c], [c, c, 0.0]])
    pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
    system = setup_system(cell, pos, [0, 0], [upf, upf], ecut=ecut * RY,
                          kmesh=kmesh, nbands=8)
    return ("nc", system)


def sys_mgo_nc(ecut=45, kmesh=(4, 4, 4), a=4.21):
    mg = parse_upf(pseudo("Mg_ONCV_PBE-1.2.upf"))
    o = parse_upf(pseudo("O_ONCV_PBE-1.2.upf"))
    c = a / 2.0
    cell = np.array([[0.0, c, c], [c, 0.0, c], [c, c, 0.0]])
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    system = setup_system(cell, pos, [0, 1], [mg, o], ecut=ecut * RY,
                          kmesh=kmesh, nbands=16)
    return ("nc", system)


# ---------------------------------------------------------------- flat seed
def _install_flat():
    """Replace nspin=1 SAD with a flat uniform density of the same N_e."""
    import gradwave.scf.loop as _loop
    import gradwave.scf.uspp_loop as _uspp
    _flat = {}

    def flat_nc(system, nspin, start_from, start_mag, grid, vol):
        if start_from is not None or nspin != 1:
            return _flat["nc"](system, nspin, start_from, start_mag, grid, vol)
        n = system.n_electrons / vol
        return [torch.full(grid.shape, n, dtype=torch.float64, device=grid.g2.device)]

    def flat_uspp(system, grid, vol, dev, nspin, start_from, start_mag):
        if start_from is not None or nspin != 1:
            return _flat["uspp"](system, grid, vol, dev, nspin, start_from, start_mag)
        n = system.n_electrons / vol
        return [torch.full(grid.shape, n, dtype=torch.float64, device=dev)], [None]

    _flat["nc"] = _loop._seed_density
    _flat["uspp"] = _uspp._seed_scf_density
    _loop._seed_density = flat_nc
    _uspp._seed_scf_density = flat_uspp
    return _flat


def _restore_flat(saved):
    import gradwave.scf.loop as _loop
    import gradwave.scf.uspp_loop as _uspp
    _loop._seed_density = saved["nc"]
    _uspp._seed_scf_density = saved["uspp"]


# ---------------------------------------------------------------- run one
def run_one(formalism, system, *, nspin, start_mag, smearing, width, xc,
            etol, rhotol, mixing_alpha, max_iter):
    kw = dict(nspin=nspin, smearing=smearing, width=width, etol=etol,
              rhotol=rhotol, mixing_alpha=mixing_alpha, max_iter=max_iter,
              verbose=False)
    if nspin == 2:
        kw["start_mag"] = start_mag
    t0 = time.perf_counter()
    if formalism == "nc":
        res = scf(system, xc, **kw)
    else:
        res = scf_uspp(system, xc, **kw)
    wall = time.perf_counter() - t0

    def get(name):
        try:
            return getattr(res, name)
        except AttributeError:
            return res[name]

    hist = get("history")
    r1 = float(hist[0]["res"]) if hist else float("nan")
    return {
        "n_iter": int(get("n_iter")),
        "converged": bool(get("converged")),
        "F_eV": float(get("energies").free_energy),
        "res1": r1,
        "mag_total": float(get("mag_total")) if nspin == 2 else 0.0,
        "mag_abs": float(get("mag_abs")) if nspin == 2 else 0.0,
        "wall_s": wall,
    }


def _xc(formalism, nspin):
    if nspin == 2:
        from gradwave.core.xc.spin import SpinPBE
        return SpinPBE()
    from gradwave.core.xc.pbe import PBE
    return PBE()


BUILDERS = {
    "ni_nc": sys_ni_nc, "ni_paw": sys_ni_paw, "fe_paw": sys_fe_paw,
    "si_nc": sys_si_nc, "mgo_nc": sys_mgo_nc,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["magnetic", "flat"])
    ap.add_argument("system", choices=list(BUILDERS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--start-mags", default="0.02,0.05,0.1,0.3")
    ap.add_argument("--smearing", default="gaussian")
    ap.add_argument("--width", type=float, default=0.1)
    ap.add_argument("--etol", type=float, default=1e-6)
    ap.add_argument("--rhotol", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--max-iter", type=int, default=120)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    formalism, system = BUILDERS[args.system]()

    if args.mode == "flat":
        nspin = 1
        xc = _xc(formalism, nspin)
        common = dict(nspin=nspin, start_mag=None, smearing="gaussian",
                      width=args.width, xc=xc, etol=args.etol, rhotol=args.rhotol,
                      mixing_alpha=0.7, max_iter=args.max_iter)
        base = run_one(formalism, system, **common)
        print(json.dumps({"system": args.system, "seed": "sad", **base}), flush=True)
        saved = _install_flat()
        try:
            _, system2 = BUILDERS[args.system]()
            flat = run_one(formalism, system2, **common)
        finally:
            _restore_flat(saved)
        print(json.dumps({"system": args.system, "seed": "flat", **flat}), flush=True)
        return

    # magnetic
    nspin = 2
    xc = _xc(formalism, nspin)
    sm_list = [float(x) for x in args.start_mags.split(",")]
    for sm in sm_list:
        for seed in ("default", "dloc"):
            _, sysx = BUILDERS[args.system]()
            if seed == "dloc":
                patch.install()
            try:
                out = run_one(formalism, sysx, nspin=nspin, start_mag=[sm],
                              smearing=args.smearing, width=args.width, xc=xc,
                              etol=args.etol, rhotol=args.rhotol,
                              mixing_alpha=args.alpha, max_iter=args.max_iter)
            finally:
                patch.restore()
            print(json.dumps({"system": args.system, "start_mag": sm,
                              "seed": seed, **out}), flush=True)


if __name__ == "__main__":
    main()
