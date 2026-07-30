"""Same probe on a metal: 1-atom fcc Al, gaussian smearing.

Does partial occupation / smearing degrade the one-shot error curve relative to
the insulator (Si) case?

    cd experiments/ecut_recommender && uv run python run_al.py
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


def al_cell(a=4.05):
    return a / 2 * FCC, np.array([[0.0, 0.0, 0.0]])


def run(ecut_ry, kmesh, upf, cell, pos, width):
    system = setup_system(cell, pos, [0], [upf], ecut=ecut_ry * RY,
                          kmesh=kmesh, nbands=12, use_symmetry=True)
    res = scf(system, PBE(), smearing="gaussian", width=width,
              etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged
    return res


def main():
    upf = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    cell, pos = al_cell()
    natoms = 1
    kmesh = (6, 6, 6)
    width = 0.14                            # eV, gaussian

    # Al_ONCV_PBE-1.2 is an 11-electron (2s2p semicore) pseudo, converged only by
    # ~80-100 Ry; a 15 Ry probe is meaningless (deeply pre-asymptotic). Probe in
    # the asymptotic window instead.
    sweep_ry = np.array([36.0, 40.0, 45.0, 50.0, 60.0, 70.0, 85.0, 100.0])
    probe_ry = 40.0
    ecut_large_ry = 100.0                   # <= 4*probe, annulus top == reference

    print("# Al (1 atom fcc, 11-e semicore), PBE, ONCV, 6x6x6, sym, gaussian 0.14 eV")
    print(f"# probe ecut = {probe_ry} Ry, ecut_large = {ecut_large_ry} Ry\n")

    energies = {}
    for ec in sweep_ry:
        res = run(ec, kmesh, upf, cell, pos, width)
        energies[ec] = float(res.energies.free_energy)
        print(f"  sweep ecut={ec:5.1f} Ry  F = {energies[ec]:+.6f} eV", flush=True)
    ref = energies[ecut_large_ry]

    res_probe = run(probe_ry, kmesh, upf, cell, pos, width)
    est = estimate_density_error(res_probe, ecut_large=ecut_large_ry * RY)
    full = energy_error_curve(res_probe, np.array([probe_ry * RY]),
                              ecut_large=ecut_large_ry * RY)[0]
    print(f"\n  curve(ecut) = {full:.6e} eV   shipped denergy = {est.denergy:.6e} eV"
          f"   (rel diff {abs(full - est.denergy) / abs(est.denergy):.2e})")

    vgrid_ry = sweep_ry[(sweep_ry >= probe_ry) & (sweep_ry < ecut_large_ry)]
    pred = -energy_error_curve(res_probe, vgrid_ry * RY,
                               ecut_large=ecut_large_ry * RY)
    print("\n## Predicted vs actual remaining error (relative to 40 Ry), meV/atom\n")
    print(f"{'ecut(Ry)':>9} {'actual':>10} {'predicted':>10} {'ratio':>8}")
    for ec, p in zip(vgrid_ry, pred, strict=True):
        actual = (energies[ec] - ref) / natoms * 1e3
        pm = p / natoms * 1e3
        ratio = pm / actual if abs(actual) > 1e-9 else float("nan")
        print(f"{ec:9.1f} {actual:10.3f} {pm:10.3f} {ratio:8.2f}")

    dense_ry = np.arange(probe_ry, ecut_large_ry + 1e-9, 0.5)
    dense_pred = -energy_error_curve(res_probe, dense_ry * RY,
                                     ecut_large=ecut_large_ry * RY)
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
