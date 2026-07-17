"""Spin-spiral dispersion of bcc Fe via the magnitude-robust moment constraint.

A spin spiral needs each atomic moment held at full magnitude while its direction
is rotated. The direction-only Ma-Dudarev penalty (|M^perp|^2) cannot do this: a
strong ferromagnet forced away from collinear demagnetizes to satisfy the penalty
for free. The magnitude-robust "vector" penalty, E_p = lambda|M - m0 e|^2, pins
the full moment vector, so the moments are held at m0 at any angle.

Two-atom cubic (bcc) Fe cell: corner and body-center sublattices. Rotating the
body-center moment by an angle theta relative to the corner traces a commensurate
frozen spin spiral along (111) -- no generalized-Bloch machinery needed. The
resulting E(theta) is the spin-spiral dispersion; for ferromagnetic Fe it rises
monotonically from theta = 0 (FM ground state) to theta = 180 deg (antiparallel),
and the moment stays ~2.2 muB throughout. A "perp" spot-check at large theta shows
the moment collapsing instead -- the magnitude problem the "vector" mode solves.

Committed results (examples/fe_spin_spiral.json, .png) were run on CPU; each point
is a full non-collinear SCF at kmesh (3,3,3), ~8-13 min on 8 threads, so the whole
sweep is ~1.5 h. The points are independent — run them as parallel processes to cut
wall-clock. Run:
    PYTHONPATH=src python examples/fe_spin_spiral.py
"""
import json
import time

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.moment_config import atomic_weights, constrained_moment_scf
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

torch.set_num_threads(8)
RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"

a = 2.87                                        # bcc Fe lattice constant [Å]
cell = a * np.eye(3)                            # cubic 2-atom cell
pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) * a
LAM = 6.0                                        # penalty strength [eV/muB^2]
THETAS = (0, 30, 60, 90, 120, 150, 180)         # relative sublattice angle [deg]
SCF = dict(smearing="gaussian", width=0.1, etol=1e-7, rhotol=1e-6, max_iter=200,
           mixing_alpha=0.4, verbose=False)


def build():
    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0, 0], [fe, fe], ecut=60 * RY,
                          kmesh=(3, 3, 3), nbands=24, time_reversal=False)
    return system, NoncollinearXC(LSDA_PW92())


def targets(theta_deg):
    th = np.deg2rad(theta_deg)
    return [[0.0, 0, 1.0], [float(np.sin(th)), 0, float(np.cos(th))]]


def point(system, xc, w, theta, mode, tm):
    t0 = time.time()
    _, info = constrained_moment_scf(system, xc, targets(theta), lam=LAM, weights=w,
                                     mode=mode, target_mag=tm, mag_init_scale=1.2, **SCF)
    M = info["M"]
    mag = [round(float(x), 3) for x in torch.linalg.norm(M, dim=-1)]
    ca = float((M[0] * M[1]).sum()
               / (torch.linalg.norm(M[0]) * torch.linalg.norm(M[1])).clamp_min(1e-9))
    return dict(theta=theta, mode=mode, converged=info["converged"],
                energy_eV=info["energy_eV"], moment_muB=mag,
                relative_angle_deg=round(float(np.rad2deg(np.arccos(
                    np.clip(ca, -1, 1)))), 1), seconds=round(time.time() - t0))


def main():
    system, xc = build()
    w = atomic_weights(system)

    # ferromagnetic reference magnitude (unconstrained, seeded high-spin)
    _, fm = constrained_moment_scf(system, xc, targets(0), lam=0.0, weights=w,
                                   mag_init_scale=1.2, **SCF)
    m0 = torch.linalg.norm(fm["M"], dim=-1)
    print(f"FM reference |M| = {[round(float(x), 3) for x in m0]} muB", flush=True)

    results = []
    print("\ntheta  E-E0 [meV]   |M| [muB]     rel[deg]  conv")
    for theta in THETAS:
        r = point(system, xc, w, theta, "vector", m0)
        results.append(r)
        de = (r["energy_eV"] - results[0]["energy_eV"]) * 1000
        print(f"{theta:4d}  {de:9.2f}   {r['moment_muB']}  {r['relative_angle_deg']:6.1f}"
              f"   {r['converged']}", flush=True)

    print("\nperp spot-check (demagnetizes at large theta):")
    for theta in (120, 180):
        r = point(system, xc, w, theta, "perp", None)
        results.append(r)
        print(f"{theta:4d}  |M|={r['moment_muB']}  rel={r['relative_angle_deg']:.1f}",
              flush=True)

    with open("examples/fe_spin_spiral.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote examples/fe_spin_spiral.json")


if __name__ == "__main__":
    main()
