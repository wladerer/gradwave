"""Reproduce test_warmstart_survives_grid_shape_change and print iter counts.
Usage: GRADWAVE_GAMMA_REAL=<0|1> python scratchpad_warm.py
"""
import os, sys
import numpy as np
import torch
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradwave.calculator import GradWave  # noqa: E402
from tests.helpers import RY, SI_ONCV, pseudo  # noqa: E402
import gradwave.scf.loop as loop  # noqa: E402

A0 = 5.43
RATTLE = np.array([[0.0, 0.0, 0.0], [0.02, -0.01, 0.015]])
BIG = 1.1


def _atoms(scale=1.0):
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]) * scale
    pos = (A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]]) + RATTLE) * scale
    return Atoms("Si2", positions=pos, cell=cell, pbc=True)


def _calc(**extra):
    tight = dict(etol=1e-10, rhotol=1e-9, diago_tol=1e-11, max_iter=200)
    tight.update(extra)
    return GradWave(ecut=10 * RY, pseudopotentials={"Si": pseudo(SI_ONCV)},
                    xc="lda", kpts=(1, 1, 1), **tight)


def main():
    torch.set_num_threads(8)
    print("GAMMA_REAL_ENV =", loop._GAMMA_REAL_ENV)
    cold = _atoms(scale=BIG); cold.calc = _calc()
    e_cold = cold.get_potential_energy(); n_cold = cold.calc.last_result.n_iter
    print("cold: niter=", n_cold, "gamma_real=", cold.calc.last_result.gamma_real)

    warm = _atoms(scale=1.0); warm.calc = _calc()
    warm.get_potential_energy(); n_small = warm.calc.last_result.n_iter
    big = _atoms(scale=BIG)
    warm.set_cell(big.cell, scale_atoms=False)
    warm.set_positions(big.get_positions())
    e_warm = warm.get_potential_energy(); n_warm = warm.calc.last_result.n_iter
    print("small: niter=", n_small)
    print("warm: niter=", n_warm, "remaps=", warm.calc._warm_start_remaps,
          "gamma_real=", warm.calc.last_result.gamma_real)
    print(f"n_warm={n_warm} n_cold={n_cold} warm<cold={n_warm < n_cold} "
          f"de={abs(e_warm - e_cold):.2e}")

    def res_traj(hist):
        key = "res" if hist and "res" in hist[0] else None
        if key is None and hist:
            key = next((k for k in hist[0] if "res" in str(k) or "rho" in str(k)), None)
        return [f"{float(h.get(key, float('nan'))):.2e}" for h in hist] if key else hist
    print("cold res:", res_traj(cold.calc.last_result.history))
    print("warm res:", res_traj(warm.calc.last_result.history))


if __name__ == "__main__":
    main()
