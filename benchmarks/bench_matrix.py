"""Benchmark matrix: materials diversity + Si supercell size scaling.

Usage: uv run python benchmarks/bench_matrix.py [cpu|cuda] [threads] [case ...]
Cases default to all. Symmetry on everywhere (production configuration).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
ROOT = Path(__file__).parents[1]
PSE = ROOT / "tests/fixtures/qe/pseudos"


def diamond(a, elem):
    return a / 2 * FCC, np.array([[0.0, 0, 0], [a / 4] * 3]), [elem, elem]


def si_supercell(nrep):
    """nrep³ repetitions of the 8-atom conventional diamond cell."""
    a = 5.43
    base_frac = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]
    )
    frac = np.vstack([
        (base_frac + np.array([i, j, k])) / nrep
        for i in range(nrep) for j in range(nrep) for k in range(nrep)
    ])
    cell = a * nrep * np.eye(3)
    return cell, frac @ cell, ["Si"] * (8 * nrep**3)


GAAS_A, MGO_A = 5.653, 4.212
CASES = {
    "si2": dict(geom=diamond(5.43, "Si"), ecut=30 * RY, kmesh=(4, 4, 4),
                xc=LDA_PW92, smearing="none", width=0.1, nbands=None),
    "c2": dict(geom=diamond(3.567, "C"), ecut=40 * RY, kmesh=(4, 4, 4),
               xc=PBE, smearing="none", width=0.1, nbands=None),
    "gaas": dict(geom=(GAAS_A / 2 * FCC, np.array([[0.0, 0, 0], [GAAS_A / 4] * 3]),
                       ["Ga", "As"]),
                 ecut=40 * RY, kmesh=(4, 4, 4), xc=PBE,
                 smearing="gaussian", width=0.02, nbands=13),
    "al": dict(geom=(4.05 / 2 * FCC, np.zeros((1, 3)), ["Al"]),
               ecut=40 * RY, kmesh=(8, 8, 8), xc=PBE,
               smearing="gaussian", width=0.1, nbands=10),
    "cu": dict(geom=(3.615 / 2 * FCC, np.zeros((1, 3)), ["Cu"]),
               ecut=45 * RY, kmesh=(8, 8, 8), xc=PBE,
               smearing="gaussian", width=0.1, nbands=16),
    "mgo": dict(geom=(MGO_A / 2 * FCC,
                      np.array([[0.0, 0.0, 0.0], [MGO_A / 2] * 3]), ["Mg", "O"]),
                ecut=50 * RY, kmesh=(4, 4, 4), xc=PBE,
                smearing="none", width=0.1, nbands=None),
    "si8": dict(geom=si_supercell(1), ecut=30 * RY, kmesh=(2, 2, 2),
                xc=LDA_PW92, smearing="none", width=0.1, nbands=None),
    "si64": dict(geom=si_supercell(2), ecut=30 * RY, kmesh=(1, 1, 1),
                 xc=LDA_PW92, smearing="none", width=0.1, nbands=None),
}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    wanted = sys.argv[3:] or list(CASES)
    torch.set_num_threads(threads)

    print(f"device={device} threads={threads}")
    print(f"{'case':6s} {'nat':>4s} {'ne':>5s} {'nk':>4s} {'npw':>6s} {'grid':>12s} "
          f"{'setup':>7s} {'scf':>8s} {'it':>3s} {'E/atom (eV)':>14s}")
    for name in wanted:
        cfg = CASES[name]
        cell, pos, symbols = cfg["geom"]
        species = sorted(set(symbols))
        upfs = [parse_upf(PSE / f"{s}_ONCV_PBE-1.2.upf") for s in species]
        soa = [species.index(s) for s in symbols]

        t0 = time.time()
        system = setup_system(cell, pos, soa, upfs, ecut=cfg["ecut"],
                              kmesh=cfg["kmesh"], nbands=cfg["nbands"],
                              use_symmetry=True)
        if device != "cpu":
            system = system.to(device)
        t_setup = time.time() - t0

        t0 = time.time()
        res = scf(system, cfg["xc"](), smearing=cfg["smearing"], width=cfg["width"],
                  etol=1e-8, rhotol=1e-7, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        t_scf = time.time() - t0

        nat = len(symbols)
        e_per_atom = float(res.energies.free_energy) / nat
        flag = "" if res.converged else "  ***NOT CONVERGED***"
        print(f"{name:6s} {nat:4d} {system.n_electrons:5.0f} {len(system.spheres):4d} "
              f"{system.spheres[0].npw:6d} {str(system.grid.shape):>12s} "
              f"{t_setup:6.1f}s {t_scf:7.1f}s {res.n_iter:3d} {e_per_atom:14.6f}{flag}",
              flush=True)


if __name__ == "__main__":
    main()
