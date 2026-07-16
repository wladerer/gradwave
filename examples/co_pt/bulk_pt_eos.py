"""Equation of state for bulk fcc Pt (PBE, PAW), for the CO/Pt learned-U study.

Scans the lattice constant of the one-atom fcc primitive cell, fits a
third-order Birch-Murnaghan E(V), and reports the equilibrium a0 and bulk
modulus B0. The a0 seeds the Pt(111) slab.

psl kjpaw suggests 39/401 Ry; the whole campaign runs at 40/400 Ry so the
bulk, slab, and adsorbate energies are on one cutoff. fcc Pt is a metal, so
a dense k-mesh with gaussian smearing is required.

Run on the asus GPU:
    LD_LIBRARY_PATH=/run/opengl-driver/lib \
      ~/.venvs/base/bin/python examples/co_pt/bulk_pt_eos.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

RY = 13.605693122994
FIX = "tests/fixtures/qe/pseudos"

ECUT = 40 * RY
ECUTRHO = 400 * RY
KMESH = (12, 12, 12)
WIDTH = 0.20            # eV, gaussian smearing
NBANDS = 14            # Pt: 10 valence e -> ~6 occupied, ample buffer
A_GRID = np.linspace(3.82, 4.04, 7)   # Angstrom, brackets the PBE a0 ~ 3.97


def fcc_primitive(a):
    return 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def birch_murnaghan(V, E0, V0, B0, Bp):
    """Third-order Birch-Murnaghan E(V). B0 in the same volume/energy units."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * Bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def fit_bm(volumes, energies):
    """Fit E(V) to third-order BM. Seed from a parabola; refine with scipy
    if available, else keep the parabolic minimum."""
    V = np.asarray(volumes)
    E = np.asarray(energies)
    c2, c1, c0 = np.polyfit(V, E, 2)
    V0 = -c1 / (2 * c2)
    E0 = c0 + c1 * V0 + c2 * V0**2
    B0 = 2 * c2 * V0                     # eV/Å^3
    seed = [E0, V0, B0, 4.0]
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(birch_murnaghan, V, E, p0=seed, maxfev=20000)
        return dict(E0=float(popt[0]), V0=float(popt[1]),
                    B0=float(popt[2]), Bp=float(popt[3]), method="BM3")
    except Exception:
        return dict(E0=float(E0), V0=float(V0), B0=float(B0), Bp=4.0,
                    method="parabola")


def main():
    # CPU is ~8x faster than this laptop GPU for a 1-atom fp64 cell (the RTX
    # 3050 does fp64 at 1/64 of fp32, and the small kernels never fill it).
    # GW_DEVICE overrides; default cpu. GW_THREADS sets the CPU thread count.
    device = os.environ.get("GW_DEVICE", "cpu")
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("GW_THREADS", "16")))
    print(f"device: {device}  threads: {torch.get_num_threads()}", flush=True)
    paw = parse_upf_paw(f"{FIX}/Pt.pbe-n-kjpaw_psl.1.0.0.UPF")

    # Fix the FFT grid to the largest-volume cell so start_from can chain the
    # volumes (a warm start on a fixed grid; the density rescales by V ratio).
    fft_shape = tuple(setup_uspp(fcc_primitive(A_GRID.max()), [[0.0, 0, 0]], [0],
                                 [paw], ecut=ECUT, ecutrho=ECUTRHO, kmesh=(1, 1, 1),
                                 nbands=NBANDS).grid.shape)
    print(f"fixed FFT grid: {fft_shape}", flush=True)

    rows = []
    prev = None
    t_all = time.time()
    for a in sorted(A_GRID):                     # ascending volume, warm-started
        cell = fcc_primitive(a)
        vol = float(abs(np.linalg.det(cell)))    # Å^3, one atom
        system = setup_uspp(cell, [[0.0, 0, 0]], [0], [paw], ecut=ECUT,
                            ecutrho=ECUTRHO, kmesh=KMESH, nbands=NBANDS,
                            use_symmetry=True, fft_shape=fft_shape)
        system = system.to(device)
        t0 = time.time()
        res = scf_uspp(system, PBE(), smearing="gaussian", width=WIDTH,
                       etol=1e-8, rhotol=1e-7, max_iter=80, verbose=False,
                       start_from=prev)
        prev = res
        e = float(res["energies"].free_energy)
        dt = time.time() - t0
        rows.append(dict(a=float(a), volume=vol, energy=e,
                         converged=bool(res["converged"]), n_iter=res["n_iter"],
                         fermi=float(res["fermi"]), seconds=dt))
        print(f"a={a:.4f} Å  V={vol:7.3f} Å³  E={e:14.6f} eV  "
              f"{'ok' if res['converged'] else 'NC'}  "
              f"{res['n_iter']} it  {dt:.1f}s", flush=True)

    fit = fit_bm([r["volume"] for r in rows], [r["energy"] for r in rows])
    V0 = fit["V0"]
    a0 = (4.0 * V0) ** (1.0 / 3.0)          # fcc: V_primitive = a^3/4
    B0_GPa = fit["B0"] * 160.21766208        # eV/Å^3 -> GPa
    summary = dict(
        cutoffs_Ry=dict(wfc=ECUT / RY, rho=ECUTRHO / RY),
        kmesh=KMESH, width_eV=WIDTH, a0_ang=a0, V0_ang3=V0,
        B0_GPa=B0_GPa, fit=fit, rows=rows,
        total_seconds=time.time() - t_all,
    )
    with open("examples/co_pt/bulk_pt_eos.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfcc Pt PBE:  a0 = {a0:.4f} Å   B0 = {B0_GPa:.1f} GPa   "
          f"({fit['method']})", flush=True)
    print(f"experiment ~3.92 Å / 278 GPa; PBE typically ~3.97 Å / ~250 GPa",
          flush=True)
    print("wrote examples/co_pt/bulk_pt_eos.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
