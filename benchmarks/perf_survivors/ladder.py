"""Perf-survivors campaign ladder: graded SCF configs + timing / profiling.

The graded set (small-cell perf story close-out): Si2 insulator, Al-4 metal,
Fe-1 magnetic, Si-64 15 Ry IBZ nk=4, Al-32 IBZ. Modes:

    uv run python benchmarks/perf_survivors/ladder.py time  <case> [solver] [reps]
    uv run python benchmarks/perf_survivors/ladder.py prof  <case> [solver]

`solver` is the registered eigensolver name ("davidson" eager default,
"davidson-native" for the native headline path). Threads capped at 8 (asus
hybrid-CPU rule). Prints one RESULT line per rep:

    RESULT case=<> solver=<> rep=<> wall=<s> iters=<n> eig=<s> E=<eV>

`prof` runs one SCF under cProfile and prints the gradwave-filtered cumtime
table — the Amdahl-gate evidence for the survivor ideas.
"""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.recorder import SCFRecorder

ROOT = Path(__file__).parents[2]
PSE = ROOT / "tests/fixtures/qe/pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _si_supercell(nrep):
    a = 5.43
    base = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75],
         [0.25, 0.75, 0.75]])
    frac = np.vstack([(base + np.array([i, j, k])) / nrep
                      for i in range(nrep) for j in range(nrep) for k in range(nrep)])
    cell = a * nrep * np.eye(3)
    return cell, frac @ cell, ["Si"] * (8 * nrep ** 3)


def _al_conv_supercell(nrep):
    """nrep^3 of the 4-atom conventional fcc Al cell."""
    a = 4.05
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    frac = np.vstack([(base + np.array([i, j, k])) / nrep
                      for i in range(nrep) for j in range(nrep) for k in range(nrep)])
    cell = a * nrep * np.eye(3)
    return cell, frac @ cell, ["Al"] * (4 * nrep ** 3)


CASES = {
    # name: geom, ecut, kmesh, xc, scf kwargs
    "si2": dict(
        geom=(5.43 / 2 * FCC, np.array([[0.0, 0, 0], [5.43 / 4] * 3]), ["Si"] * 2),
        ecut=30 * RY, kmesh=(4, 4, 4), xc=LDA_PW92, nbands=None,
        scf=dict(smearing="none", etol=1e-9, rhotol=1e-8)),
    "al4": dict(
        geom=_al_conv_supercell(1), ecut=40 * RY, kmesh=(4, 4, 4), xc=PBE,
        nbands=None,
        scf=dict(smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8)),
    "fe1": dict(
        geom=((2.87 / 2) * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]]),
              np.zeros((1, 3)), ["Fe"]),
        ecut=40 * RY, kmesh=(4, 4, 4), xc=PBE, nbands=None,
        scf=dict(smearing="gaussian", width=0.1, etol=1e-8, rhotol=1e-7,
                 nspin=2, start_mag=[0.7], max_iter=80)),
    "si64": dict(
        geom=_si_supercell(2), ecut=15 * RY, kmesh=(2, 2, 2), xc=LDA_PW92,
        nbands=128,
        scf=dict(smearing="none", etol=1e-9, rhotol=1e-8)),
    "al32": dict(
        geom=_al_conv_supercell(2), ecut=25 * RY, kmesh=(2, 2, 2), xc=PBE,
        nbands=None,
        scf=dict(smearing="gaussian", width=0.1, etol=1e-8, rhotol=1e-7)),
}


def build(case):
    cfg = CASES[case]
    cell, pos, symbols = cfg["geom"]
    species = sorted(set(symbols))
    upfs = [parse_upf(PSE / f"{s}_ONCV_PBE-1.2.upf") for s in species]
    soa = [species.index(s) for s in symbols]
    system = setup_system(cell, pos, soa, upfs, ecut=cfg["ecut"],
                          kmesh=cfg["kmesh"], nbands=cfg["nbands"],
                          use_symmetry=True)
    return system, cfg


def run_once(case, solver):
    system, cfg = build(case)
    rec = SCFRecorder(system.grid.g2, nspin=cfg["scf"].get("nspin", 1))
    t0 = time.perf_counter()
    res = scf(system, cfg["xc"](), eigensolver=solver, verbose=False,
              recorder=rec, **cfg["scf"])
    wall = time.perf_counter() - t0
    t_eig = sum(float(d.get("t_eig") or 0.0) for d in rec.iters)
    fft = sum(int(d.get("n_fft") or 0) for d in rec.iters)
    hpsi = sum(int(d.get("n_hpsi") or 0) for d in rec.iters)
    return dict(wall=wall, iters=res.n_iter, conv=res.converged,
                E=float(res.energies.total), t_eig=t_eig, fft=fft, hpsi=hpsi)


def main():
    mode = sys.argv[1]
    case = sys.argv[2]
    solver = sys.argv[3] if len(sys.argv) > 3 else "davidson"
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    torch.set_num_threads(8)
    if mode == "time":
        for rep in range(reps):
            r = run_once(case, solver)
            print(f"RESULT case={case} solver={solver} rep={rep} "
                  f"wall={r['wall']:.3f} iters={r['iters']} conv={r['conv']} "
                  f"eig={r['t_eig']:.3f} fft={r['fft']} hpsi={r['hpsi']} "
                  f"E={r['E']:.10f}", flush=True)
    elif mode == "prof":
        # warm the FFT plans / imports with one throwaway run of a cheap case
        pr = cProfile.Profile()
        pr.enable()
        r = run_once(case, solver)
        pr.disable()
        print(f"RESULT case={case} solver={solver} wall={r['wall']:.3f} "
              f"iters={r['iters']} eig={r['t_eig']:.3f} E={r['E']:.10f}")
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(r"gradwave", 60)
        ps.sort_stats("tottime").print_stats(40)
        print(s.getvalue())
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
