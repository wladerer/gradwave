"""L1_0 FePt MAE by the magnetic force theorem: one SOC SCF, cheap directions.

The self-consistent route (fept_mae.py) costs a full SOC SCF per direction.
Here one reference SCF along [001] freezes (rho, m); every other direction is
a rigid rotation of the magnetization plus ONE frozen-potential
diagonalization (postscf/mae.py), seeded with the SU(2)-rotated reference
spinors. The force theorem needs the FULL mesh (a k-fold by the reference
magnetic group is not a valid quadrature for a rotated moment), so the
reference SCF here costs more than a magnetic-IBZ one. The saving grows with
the number of directions, which is why this is the route to MAE maps
E(theta, phi) rather than a two-point difference. fept_mae_map.py adds the
next lever: magmoms= folds each one-shot solve into its own direction's
magnetic IBZ, so the tilted solves themselves run on 30-56 of the 144 points.

Self-consistent yardstick at the same mesh/ecut (fept_mae.py, asus CPU):
    kmesh (6,6,4) = 144 k, 70 Ry: MAE = E[100]-E[001] = +2.552 meV/cell,
    easy axis [001].
The force-theorem number should land in the same band. Agreement within
roughly 10-30% is a reasonable expectation for FePt-class anisotropy.

Measured (this script, 8-core CPU, 2026-07-19):
    [001] reference SCF: 26 iterations, 5048 s, |M| = 3.223 muB.
    4 force-theorem directions: 2612 s total (~11 min each, ~7.7x cheaper
    than a full SCF per direction).
    FT MAE [100]-[001] = +2.6727 meV/cell vs self-consistent +2.552 (4.7%).
    [110] = +2.7131 (in-plane anisotropy only 0.04 meV, as tetragonal
    symmetry wants); [101] = +1.3398 ~ half of [100], the uniaxial
    K1 sin^2(theta) form evaluated at theta = 45 deg.
"""
import time

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.mae import force_theorem_mae
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear

PSE = "tests/fixtures/qe/pseudos"
RY = 13.605693122994
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)

fe = parse_upf(f"{PSE}/Fe_ONCV_PBE_FR-1.0.upf")
pt = parse_upf(f"{PSE}/Pt_ONCV_PBE_FR-1.0.upf")
a, c = 2.723, 3.712                         # L1_0 FePt tetragonal [Å]
cell = np.diag([a, a, c])
pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
KMESH = (6, 6, 4)
ECUT = 70 * RY

init = [[0, 0, 3.0], [0, 0, 0.4]]           # Fe ~3, Pt induced ~0.4, along c
system = setup_system(cell, pos, [0, 1], [fe, pt], ecut=ECUT, kmesh=KMESH,
                      nbands=30, use_symmetry=False, time_reversal=False)
if dev != "cpu":
    system = system.to(dev)
print(f"device={dev}  L1_0 FePt  kmesh={KMESH} (full, {len(system.spheres)} k)"
      f"  ecut=70Ry", flush=True)

xc = NoncollinearXC(LSDA_PW92())
t0 = time.time()
res = scf_noncollinear(system, xc, mag_vec_init=init, smearing="gaussian",
                       width=0.1, etol=1e-9, rhotol=1e-7, max_iter=300,
                       mixing_alpha=0.3, mixing_history=12, verbose=True)
print(f"[001 reference] conv={res.converged} n_it={res.n_iter} "
      f"{time.time() - t0:.0f}s  F = {float(res.energies.free_energy):+.8f} eV  "
      f"|M| = {np.linalg.norm(np.array(res.mag_vec)):.4f}", flush=True)

SQ2 = 1.0 / np.sqrt(2.0)
dirs = [[0, 0, 1.0], [1.0, 0, 0], [SQ2, SQ2, 0], [SQ2, 0, SQ2]]
t0 = time.time()
ft = force_theorem_mae(res, xc, dirs, verbose=True)
print(f"force-theorem solves: {time.time() - t0:.0f}s for {len(dirs)} directions",
      flush=True)

mae_100 = float(ft.mae[1]) * 1000
print(f"\nFT MAE = F[100]-F[001] = {mae_100:+.4f} meV/cell "
      f"(self-consistent yardstick +2.552)", flush=True)
for d, dm in zip(dirs, ft.mae, strict=True):
    print(f"  n=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})  "
          f"dF = {float(dm) * 1000:+.4f} meV", flush=True)
print(f"easy axis: {'c-axis [001] (correct for FePt)' if mae_100 > 0 else '[100] (!?)'}",
      flush=True)
print("FEPT_FT_MAE_DONE", flush=True)
