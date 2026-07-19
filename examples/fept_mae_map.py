"""L1_0 FePt anisotropy map E(theta): one SOC SCF, folded one-shot solves.

The force-theorem route (examples/fept_mae_force_theorem.py) already makes
each magnetization direction one frozen-potential diagonalization instead of
a full SCF. Here ``magmoms=`` folds each of those solves into the
direction's own magnetic (Shubnikov) IBZ over the same underlying mesh:
on the (6, 6, 4) mesh [001] keeps 30 of 144 k-points, [100] keeps 48 and a
generic tilt in the (010) plane keeps 56. The two savings compound: the
reference SCF is paid once on the full mesh, and every additional direction
costs a fraction of a single full-mesh solve. That is what makes dense
E(theta, phi) maps cheap enough to run.

The scan tilts m from [001] (the easy axis) to [100] in the (010) plane and
fits the uniaxial form

    F(theta) - F(0) = K1 sin^2(theta) + K2 sin^4(theta).

The validation run in fept_mae_force_theorem.py already saw the 45-degree
direction land at half the [100] value, which is what a dominant K1 sin^2
term predicts. This script measures the whole curve.
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
THETAS = np.deg2rad([0, 15, 30, 45, 60, 75, 90])

init = [[0, 0, 3.0], [0, 0, 0.4]]           # Fe ~3, Pt induced ~0.4, along c
system = setup_system(cell, pos, [0, 1], [fe, pt], ecut=ECUT, kmesh=KMESH,
                      nbands=30, use_symmetry=False, time_reversal=False)
if dev != "cpu":
    system = system.to(dev)
print(f"device={dev}  L1_0 FePt  kmesh={KMESH} (full, {len(system.spheres)} k)"
      f"  ecut=70Ry  {len(THETAS)} thetas [001]->[100]", flush=True)

xc = NoncollinearXC(LSDA_PW92())
t0 = time.time()
res = scf_noncollinear(system, xc, mag_vec_init=init, smearing="gaussian",
                       width=0.1, etol=1e-9, rhotol=1e-7, max_iter=300,
                       mixing_alpha=0.3, mixing_history=12, verbose=True)
print(f"[001 reference] conv={res.converged} n_it={res.n_iter} "
      f"{time.time() - t0:.0f}s  F = {float(res.energies.free_energy):+.8f} eV  "
      f"|M| = {np.linalg.norm(np.array(res.mag_vec)):.4f}", flush=True)

dirs = [[np.sin(t), 0.0, np.cos(t)] for t in THETAS]
t0 = time.time()
ft = force_theorem_mae(res, xc, dirs, magmoms=init, verbose=True)
print(f"folded force-theorem solves: {time.time() - t0:.0f}s "
      f"for {len(dirs)} directions, folds {ft.nk}", flush=True)

# K1 sin^2 + K2 sin^4 least-squares fit through the origin
s2 = np.sin(THETAS) ** 2
dF = ft.mae.numpy() * 1000.0                 # meV/cell
k1, k2 = np.linalg.lstsq(np.stack([s2, s2**2], axis=1), dF, rcond=None)[0]
fit = k1 * s2 + k2 * s2**2

print("\nE(theta) - E(0) [meV/cell], theta from [001] to [100] in (010):",
      flush=True)
for t, d, f, nk in zip(np.rad2deg(THETAS), dF, fit, ft.nk, strict=True):
    print(f"  theta={t:5.1f}  dF = {d:+.4f}   fit = {f:+.4f}   nk={nk}",
          flush=True)
print(f"K1 = {k1:+.4f} meV/cell   K2 = {k2:+.4f} meV/cell   "
      f"max fit residual = {np.abs(dF - fit).max():.4f} meV", flush=True)
print("FEPT_MAE_MAP_DONE", flush=True)
