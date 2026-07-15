"""ANALYTIC Si Gamma phonon — no finite differences anywhere.

The Hessian comes from postscf/phonons.gamma_hessian: HessianSymmetry
finds the irreducible displacements (diamond Si: ONE column of six —
the site group rotates x onto y and z, and the glide maps sublattice 0
onto 1) and each column is one self-consistent position response
(Sternheimer + Anderson, ~10 s at 45 Ry) contracted through the force
graph. Result (psl kjpaw, PBE, 45/180 Ry, 2x2x2, 32^3): optical triple
586 cm-1 vs ph.x DFPT 586.093 and FD-of-analytic-forces 586.11;
acoustic modes exactly zero under the ASR. The FD route needs 12 SCF
re-runs; this needs one SCF plus one response solve.
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
from gradwave.postscf.phonons import (  # noqa: E402
    gamma_frequencies,
    gamma_hessian,
)
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

t1 = time.time()
H = gamma_hessian(res, PBE(), verbose=True)
print(f"hessian ({time.time()-t1:.0f}s)")

freqs = gamma_frequencies(H, [28.0855, 28.0855])
print("Gamma frequencies (cm^-1):", np.round(freqs, 2))
print("QE ph.x reference optical: 586.093 cm^-1; FD-of-forces: 586.11")
print("PHONON DONE")
