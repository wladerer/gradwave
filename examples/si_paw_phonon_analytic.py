"""ANALYTIC Si Gamma phonon — no finite differences anywhere.

The 6x6 Hessian comes from postscf/uspp_position.hessian_column: each
column is one self-consistent position response (Sternheimer + Anderson,
~10 s at 45 Ry) contracted through the force graph. Result (psl kjpaw,
PBE, 45/180 Ry, 2x2x2, 32^3): optical triple 585.91/585.99/586.32 cm-1
(mean 586.07) vs ph.x DFPT 586.093 and FD-of-analytic-forces 586.11;
acoustic modes exactly zero under the ASR. The FD route needs 12 SCF
re-runs; this needs one SCF plus six response solves.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)
REPO = Path(__file__).parents[1]

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.postscf.uspp_position import hessian_column  # noqa: E402
from gradwave.pseudo.upf_paw import parse_upf_paw  # noqa: E402
from gradwave.scf.uspp import scf_uspp, setup_uspp  # noqa: E402

RY = 13.605693122994
FIX = REPO / "tests/fixtures/qe"
paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
cell = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos0 = np.array([[0.0, 0, 0], [5.43 / 4] * 3])

t0 = time.time()
system = setup_uspp(cell, pos0, [0, 0], [paw], ecut=45 * RY, kmesh=(2, 2, 2),
                    ecutrho=180 * RY, fft_shape=(32, 32, 32))
res = scf_uspp(system, PBE(), smearing="none", etol=1e-12, rhotol=1e-10,
               verbose=False, max_iter=80)
assert res["converged"]
print(f"scf {res['n_iter']} it ({time.time()-t0:.0f}s)")

H = np.zeros((6, 6))
for a in range(2):
    for alpha in range(3):
        t1 = time.time()
        col = hessian_column(res, PBE(), a, alpha)
        H[:, 3 * a + alpha] = col.reshape(-1).numpy()
        print(f"col ({a},{alpha}) ({time.time()-t1:.0f}s)")
H = 0.5 * (H + H.T)
# acoustic sum rule: D[a,a] -= sum_b D[a,b] (rigid translations exact)
Hblk = H.reshape(2, 3, 2, 3)
for a in range(2):
    corr = Hblk[a].sum(axis=1)  # (3, 3) summed over b
    Hblk[a, :, a, :] -= corr
H = Hblk.reshape(6, 6)
H = 0.5 * (H + H.T)

EV_A2 = 1.602176634e-19 / 1e-20
w2, vecs = np.linalg.eigh(H / 28.0855)
# frequencies: w^2 [eV/A^2/amu] -> SI
w2_SI = w2 * EV_A2 / 1.66053906660e-27
freqs = np.sqrt(np.abs(w2_SI)) * np.sign(w2) / (2 * np.pi * 2.99792458e10)
print("Gamma frequencies (cm^-1):", np.round(freqs, 2))
print("QE ph.x reference optical: 586.093 cm^-1; FD-of-forces: 586.11")
print("PHONON DONE")
