"""Ground-truth ecut sweep vs one-shot predicted error curve, on 2-atom Si.

Runs a real ecut sweep (the truth) and one loose probe SCF, then asks whether the
binned/cumulative complement correction from the single probe reproduces the
sweep's energy-vs-cutoff curve, and what ecut each method recommends for
10/3/1 meV/atom targets.

    uv run python -m pytest -q                 # (not this file)
    cd experiments/ecut_recommender && uv run python run_si.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.discretization_error import estimate_density_error
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve import energy_error_curve, recommend_ecut  # noqa: E402

torch.set_num_threads(2)

RY = 13.605693122994
FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "qe" / "pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def si_cell(a=5.43):
    return a / 2 * FCC, np.array([[0.0, 0, 0], [a / 4] * 3])


def run(ecut_ry, kmesh, upf, cell, pos):
    system = setup_system(cell, pos, [0, 0], [upf], ecut=ecut_ry * RY,
                          kmesh=kmesh, use_symmetry=True)
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
              verbose=False)
    assert res.converged
    return res


def main():
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = si_cell()
    natoms = 2
    kmesh = (4, 4, 4)

    sweep_ry = np.array([12.0, 15.0, 18.0, 22.0, 26.0, 32.0, 40.0])
    probe_ry = 15.0
    ecut_large_ry = 40.0                    # annulus top == sweep reference

    print("# Si (2 atoms), PBE, ONCV, 4x4x4, use_symmetry, smearing=none")
    print(f"# probe ecut = {probe_ry} Ry, ecut_large = {ecut_large_ry} Ry\n")

    # ---- ground-truth sweep --------------------------------------------------
    energies = {}
    for ec in sweep_ry:
        res = run(ec, kmesh, upf, cell, pos)
        energies[ec] = float(res.energies.free_energy)
        print(f"  sweep ecut={ec:5.1f} Ry  E = {energies[ec]:+.6f} eV", flush=True)
    ref = energies[ecut_large_ry]

    # ---- one probe SCF -------------------------------------------------------
    res_probe = run(probe_ry, kmesh, upf, cell, pos)
    est = estimate_density_error(res_probe, ecut_large=ecut_large_ry * RY)

    # sanity: full-annulus curve value == shipped denergy
    full = energy_error_curve(res_probe, np.array([probe_ry * RY]),
                              ecut_large=ecut_large_ry * RY)[0]
    print(f"\n  curve(ecut) = {full:.6e} eV   shipped denergy = {est.denergy:.6e} eV"
          f"   (rel diff {abs(full - est.denergy) / abs(est.denergy):.2e})")

    # ---- predicted vs actual at each virtual cutoff --------------------------
    vgrid_ry = sweep_ry[(sweep_ry >= probe_ry) & (sweep_ry < ecut_large_ry)]
    pred = -energy_error_curve(res_probe, vgrid_ry * RY,
                               ecut_large=ecut_large_ry * RY)  # positive remaining err
    print("\n## Predicted vs actual remaining error (relative to 40 Ry), meV/atom\n")
    print(f"{'ecut(Ry)':>9} {'actual':>10} {'predicted':>10} {'ratio':>8}")
    for ec, p in zip(vgrid_ry, pred, strict=True):
        actual = (energies[ec] - ref) / natoms * 1e3
        pm = p / natoms * 1e3
        ratio = pm / actual if abs(actual) > 1e-9 else float("nan")
        print(f"{ec:9.1f} {actual:10.3f} {pm:10.3f} {ratio:8.2f}")

    # ---- recommendation ------------------------------------------------------
    dense_ry = np.arange(probe_ry, ecut_large_ry + 1e-9, 0.5)
    dense_pred = -energy_error_curve(res_probe, dense_ry * RY,
                                     ecut_large=ecut_large_ry * RY)
    # true curve by linear interp of the sweep (relative to ref)
    true_err = np.interp(dense_ry, sweep_ry, [energies[e] - ref for e in sweep_ry])

    print("\n## Recommended ecut [Ry] for energy targets\n")
    print(f"{'target(meV/at)':>14} {'predicted':>10} {'true':>8}")
    for tgt in (10.0, 3.0, 1.0):
        tev = tgt * natoms / 1e3
        rp = recommend_ecut(dense_ry, dense_pred, tev)
        rt = recommend_ecut(dense_ry, true_err, tev)
        print(f"{tgt:14.1f} {rp:10.1f} {rt:8.1f}")


if __name__ == "__main__":
    main()
