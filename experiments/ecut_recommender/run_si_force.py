"""Force-error curve from one probe, on DISPLACED 2-atom Si.

If the energy curve works, the same cumulative trick should give the force error
remaining at every virtual cutoff: raise the annulus floor to ecut' (masking
dpsi/drho to T_G > ecut'), then run the shipped fixed-dP force-error propagation.
Compared against a real force sweep.

    cd experiments/ecut_recommender && uv run python run_si_force.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.discretization_error import estimate_force_error
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve import density_error_at_virtual_cutoff  # noqa: E402

torch.set_num_threads(2)

RY = 13.605693122994
FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "qe" / "pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def si_cell_displaced(a=5.43, d=0.10):
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0.0, 0.0], [a / 4, a / 4, a / 4]])
    pos[1, 0] += d                          # break symmetry: displace atom 1 along x
    return cell, pos


def run(ecut_ry, kmesh, upf, cell, pos):
    system = setup_system(cell, pos, [0, 0], [upf], ecut=ecut_ry * RY,
                          kmesh=kmesh, use_symmetry=False)
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
              verbose=False)
    assert res.converged
    return res


def _corr(a, b):
    a, b = a.flatten(), b.flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main():
    upf = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")
    cell, pos = si_cell_displaced()
    kmesh = (4, 4, 4)
    sweep_ry = np.array([15.0, 18.0, 22.0, 26.0, 32.0, 40.0])
    probe_ry = 15.0
    ecut_large_ry = 40.0

    print("# Displaced Si (atom1 +0.10 A along x), PBE, ONCV, 4x4x4, no sym")
    print(f"# probe ecut = {probe_ry} Ry, ecut_large = {ecut_large_ry} Ry\n")

    fmap = {}
    for ec in sweep_ry:
        res = run(ec, kmesh, upf, cell, pos)
        fmap[ec] = forces(res)              # (2,3) eV/A
        print(f"  sweep ecut={ec:5.1f} Ry  Fx(atom1) = {float(fmap[ec][1,0]):+.5f} eV/A",
              flush=True)
    ref = fmap[ecut_large_ry]

    res_probe = run(probe_ry, kmesh, upf, cell, pos)

    vgrid = sweep_ry[(sweep_ry >= probe_ry) & (sweep_ry < ecut_large_ry)]
    print("\n## Force error remaining at virtual cutoff (rel to 40 Ry)")
    print("## metric: max|dF| component over atoms [eV/A], and Fx(atom1)\n")
    print(f"{'ecut(Ry)':>9} {'act_max':>9} {'pred_max':>9} {'ratio':>7} "
          f"{'act_Fx1':>9} {'pred_Fx1':>9} {'corr':>6}")
    for ec in vgrid:
        true_dF = ref - fmap[ec]           # F_exact(~40) - F(ec): error to remove
        err_v = density_error_at_virtual_cutoff(
            res_probe, ec * RY, ecut_large=ecut_large_ry * RY)
        pred_dF = estimate_force_error(res_probe, err_v)  # (2,3)
        am = float(true_dF.abs().max())
        pm = float(pred_dF.abs().max())
        ratio = pm / am if am > 1e-9 else float("nan")
        print(f"{ec:9.1f} {am:9.4f} {pm:9.4f} {ratio:7.2f} "
              f"{float(true_dF[1,0]):+9.4f} {float(pred_dF[1,0]):+9.4f} "
              f"{_corr(true_dF, pred_dF):6.2f}")


if __name__ == "__main__":
    main()
