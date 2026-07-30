"""Attack 2: analytic-vs-finite-difference force cross-check ceiling.

D(x) = | F_analytic(x) - F_fd(x) |_max  over a single-atom displacement x, where

  F_analytic = postscf.forces.forces()          (Hellmann-Feynman, frozen density)
  F_fd       = -(E(tau+h) - E(tau-h)) / (2 h)    (central difference of the
                                                  converged free energy per atom
                                                  displacement component)

F_fd re-converges the SCF at each displaced geometry, so it is a genuinely
independent estimate of -dE/dtau "through the converged calculation" (the same
oracle benchmarks/derivatives validates the analytic force against, quoted there
at the 1e-4 eV/Ang FD floor). Both paths are already validated, so the expected
outcome is a NULL result: the disagreement stays at the FD floor everywhere. We
report the CEILING of |F_analytic - F_fd| found over the scan.

Why a scan, not gradient ascent: F_fd is itself a finite difference and is not
cheaply differentiable, so there is no smooth D(x) to ascend here (unlike Attack
1's eggbox). The probe brief anticipates this: if the two force paths track to
numerical precision, that null result is the finding, and the honest deliverable
is the measured ceiling of disagreement.

Convergence guard (added after the first run): at d = 0.30 Ang the displaced
cell goes metallic (negative indirect gap) and the smearing="none" SCF fails to
converge (charge sloshing, converged=False at max_iter). The resulting 4 eV/Ang
"disagreement" measures the SCF failure, not the force implementation, because
HF stationarity only holds at convergence. Every SCF's converged flag is now
recorded per scan point and non-converged points are excluded from the ceiling.
This is itself a probe finding: any adversarial maximizer of D over inputs will
preferentially drive into non-converged regions and report false positives, so
a real harness must treat SCF convergence as a domain constraint on D.

Usage:
    uv run python experiments/adversarial_probe/attack2_force_crosscheck.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

torch.set_num_threads(2)
RNG = np.random.default_rng(1)

RY = 13.605693122994
FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "qe" / "pseudos"
UPF = parse_upf(FIX / "Si_ONCV_PBE-1.2.upf")

A = 5.43
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
CELL = A / 2 * FCC
POS0 = np.array([[0.0, 0.0, 0.0], [A / 4, A / 4, A / 4]])

# production-ish cutoff (the discretization test-suite treats ~20-30 Ry as
# converged for Si NC); symmetry OFF so the displaced geometry is not folded.
ECUT = 24.0 * RY
KMESH = (2, 2, 2)
SCF_KW = dict(smearing="none", etol=1e-10, rhotol=1e-9, verbose=False, max_iter=120)

_SCF_COUNT = 0
_CONV_LOG: list[bool] = []  # per-SCF convergence flags; a non-converged SCF
# invalidates the HF/FD comparison at that point (stationarity does not hold)


def _energy(pos: np.ndarray) -> float:
    global _SCF_COUNT
    _SCF_COUNT += 1
    system = setup_system(CELL, pos, [0, 0], [UPF], ecut=ECUT, kmesh=KMESH,
                          use_symmetry=False)
    res = scf(system, PBE(), **SCF_KW)
    _CONV_LOG.append(bool(res.converged))
    return float(res.energies.free_energy)


def _analytic_force(pos: np.ndarray) -> np.ndarray:
    global _SCF_COUNT
    _SCF_COUNT += 1
    system = setup_system(CELL, pos, [0, 0], [UPF], ecut=ECUT, kmesh=KMESH,
                          use_symmetry=False)
    res = scf(system, PBE(), **SCF_KW)
    _CONV_LOG.append(bool(res.converged))
    return forces(res, remove_net=False).detach().numpy()  # (na, 3)


def _fd_force_on_atom(pos: np.ndarray, atom: int, h: float) -> np.ndarray:
    """Central-difference F = -dE/dtau for one atom (3 components, 6 SCFs)."""
    f = np.zeros(3)
    for c in range(3):
        pp = pos.copy()
        pp[atom, c] += h
        pm = pos.copy()
        pm[atom, c] -= h
        f[c] = -(_energy(pp) - _energy(pm)) / (2 * h)
    return f


def scan(n_points: int, h: float):
    direction = RNG.normal(size=3)
    direction /= np.linalg.norm(direction)
    atom = 1  # displace the second Si atom
    rows = []
    ceiling = -1.0
    ceiling_row = None
    for i in range(n_points):
        d = 0.3 * (i + 1) / n_points  # displacement magnitude in [0, 0.3] Ang
        pos = POS0.copy()
        pos[atom] = POS0[atom] + d * direction
        n0 = len(_CONV_LOG)
        f_an = _analytic_force(pos)[atom]
        f_fd = _fd_force_on_atom(pos, atom, h)
        all_converged = all(_CONV_LOG[n0:])
        disagree = float(np.abs(f_an - f_fd).max())
        rows.append({
            "disp_ang": d, "f_analytic": f_an.tolist(),
            "f_fd": f_fd.tolist(), "abs_diff_max": disagree,
            "rel_diff": disagree / (float(np.abs(f_fd).max()) + 1e-12),
            "all_scf_converged": all_converged,
        })
        print(f"  d={d:.3f} Ang  |F_an|={np.abs(f_an).max():.4f}  "
              f"|F_an-F_fd|_max={disagree:.2e}  converged={all_converged}")
        # the ceiling is only meaningful over VALID points: a non-converged SCF
        # breaks HF stationarity, so its "disagreement" measures the SCF failure,
        # not the force implementation (observed: 4 eV/Ang at d=0.30 where the
        # displaced cell goes metallic and the smearing="none" SCF oscillates)
        if all_converged and disagree > ceiling:
            ceiling = disagree
            ceiling_row = rows[-1]
    return direction, atom, rows, ceiling, ceiling_row


def main():
    t0 = time.time()
    h = 5e-3  # FD step (Ang); central difference floors ~ h^2 * E''' + eps/h
    n_points = 6
    print(f"[attack2] scan {n_points} displacements, h={h} Ang, ecut={ECUT/RY:.0f} Ry")
    direction, atom, rows, ceiling, ceiling_row = scan(n_points, h)

    out = {
        "setup": {
            "cell_a_ang": A, "ecut_Ry": ECUT / RY, "kmesh": list(KMESH),
            "fd_step_ang": h, "displaced_atom": atom,
            "direction": direction.tolist(),
            "note": "F_analytic=postscf.forces remove_net=False; "
                    "F_fd=central diff of free energy; symmetry OFF",
        },
        "scan": rows,
        "disagreement_ceiling_eV_ang": ceiling,
        "ceiling_point": ceiling_row,
        "total_scf_count": _SCF_COUNT,
        "wall_s": time.time() - t0,
    }
    dst = Path(__file__).resolve().parent / "attack2_results.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {_SCF_COUNT} SCFs, {out['wall_s']:.0f}s -> {dst}")
    print(f"  disagreement ceiling |F_analytic - F_fd|_max = "
          f"{ceiling:.3e} eV/Ang  (FD step {h} Ang)")


if __name__ == "__main__":
    main()
