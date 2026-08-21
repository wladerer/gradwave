"""Amdahl profile of the analytic shielding route sigma_shielding_dq.

Splits one Si 12 Ry 4^3 evaluation into:
  - PER-K SETUP  : ShieldingDq.__init__ (eigh + velocity matrices at each mesh k)
  - PER-K SOLVE  : branch_fields_axis (resolvent applies, per axis)  [REDUCIBLE]
  - FIXED        : Biot-Savart G-sum + lstsq                          [NOT reducible]
IBZ wedge reduction removes only the per-k work; the fixed part is untouched.
"""
import time
import numpy as np
import torch

from gradwave.constants import RY_EV
from gradwave.scf.loop import setup_system, scf
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.grids import reciprocal_cell
from gradwave.postscf.kgeometry_nmr import (
    ShieldingDq, _transverse_frame, _biot_savart_sigma_cols_dq,
)

torch.manual_seed(0)
RY = RY_EV
PSEUDO = "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf"

a = 5.43
cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos = np.array([[0.0, 0, 0], [a / 4] * 3])
KMESH = (4, 4, 4)

t0 = time.perf_counter()
system = setup_system(cell, pos, [0, 0], [parse_upf(PSEUDO)], ecut=12 * RY,
                      kmesh=KMESH, nbands=8, use_symmetry=False,
                      fft_shape=(20, 20, 20))
res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=120)
t_scf = time.perf_counter() - t0
nk = len(system.spheres)
print(f"# Si {KMESH} 12Ry  nk(full mesh)={nk}  SCF={t_scf:.2f}s", flush=True)

def timeit(fn, n=1):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

# ---- PER-K SETUP: ShieldingDq.__init__ (eigh at every mesh k) ----
t_setup = timeit(lambda: ShieldingDq(res))
print(f"# setup done {t_setup:.2f}s", flush=True)
eng = ShieldingDq(res)

b = reciprocal_cell(system.grid.cell)
k_frac = np.stack([sph.k_frac for sph in system.spheres])
mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
axes = [i for i in range(3) if mesh_n[i] > 1]
sites = system.positions.detach().cpu().to(torch.float64)

# ---- PER-K SOLVE: branch_fields_axis, per axis (resolvent applies) ----
axis_fields = {}
def solve_axis(i):
    q_hat = np.asarray(b[i], float) / np.linalg.norm(b[i])
    pols = list(_transverse_frame(q_hat))
    axis_fields[i] = (q_hat, pols, eng.branch_fields_axis(q_hat, pols))
t_solve_total = 0.0
for i in axes:
    t_solve_total += timeit(lambda i=i: solve_axis(i))
    print(f"# axis {i} solved (cum {t_solve_total:.2f}s)", flush=True)

# ---- FIXED: Biot-Savart + lstsq ----
def fixed_part():
    b_rows, m_rows = [], []
    for i in axes:
        q_hat, pols, fields = axis_fields[i]
        for pol, (s0f, dsf) in zip(pols, fields):
            col, _ = _biot_savart_sigma_cols_dq(
                s0f, dsf, torch.as_tensor(q_hat, dtype=torch.float64),
                system.grid.g_cart, sites)
            m_rows.append(col); b_rows.append(np.cross(q_hat, pol))
    bmat = torch.as_tensor(np.stack(b_rows), dtype=torch.float64)
    mmat = torch.stack(m_rows)
    for s in range(mmat.shape[1]):
        torch.linalg.lstsq(bmat, mmat[:, s, :]).solution.mT
t_fixed = timeit(fixed_part)

reducible = t_setup + t_solve_total
total = reducible + t_fixed
print(f"# PER-K SETUP (eigh/vel, __init__)     = {t_setup:.3f}s")
print(f"# PER-K SOLVE (resolvent, {len(axes)} axes)     = {t_solve_total:.3f}s")
print(f"# FIXED (Biot-Savart + lstsq)          = {t_fixed:.3f}s")
print(f"# --- shielding total (excl SCF)       = {total:.3f}s")
print(f"# REDUCIBLE fraction (per-k)/(total)   = {reducible/total:.4f}")
print(f"# FIXED     fraction                   = {t_fixed/total:.4f}")

# Amdahl ceiling for a few discrete-mesh reduction factors r
frac = reducible / total
for r in (2.0, 3.2, 3.9, 8.0):
    speedup = 1.0 / ((1 - frac) + frac / r)
    print(f"# Amdahl: reducible={frac:.3f}, r={r:>4}x  ->  net speedup {speedup:.2f}x")
print("# EXIT=0")
