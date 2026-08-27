"""Report cond(S(k)) on a real MgO PAW context.

Historically (PR #399) this verified that the ecut-instability *guard* fired on
the ill-conditioned MgO overlap. The guard is gone: :class:`ShieldingDq` now
solves the ∂/∂q resolvents matrix-free by the S-metric CG Sternheimer, which is
ecut-stable regardless of cond(S), so cond(S) is no longer a divergence
predictor — only a basis-health diagnostic. This script now just prints it.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gradwave.constants import RY_EV as RY
from gradwave.core.xc.pbe import PBE
from gradwave.postscf import kgeometry_nmr as kg
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

MG = sys.argv[1]
O = "tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF"
A = 4.21
CELL = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL

torch.set_num_threads(5)
system = setup_uspp(CELL, POS, [0, 1], [parse_upf_paw(MG), parse_upf_paw(O)],
                    ecut=40 * RY, kmesh=(2, 2, 2), ecutrho=160 * RY, nbands=12)
res = scf_uspp(system, PBE(), etol=1e-8, rhotol=1e-7, diago_tol=1e-9,
               verbose=False, max_iter=80)
assert res["converged"]
ctx = kg.build_uspp_response_ctx(res, PBE())

cond = kg.uspp_overlap_conditioning(ctx)
print(f"cond(S) over mesh: {cond}", flush=True)
print("VERIFY_DONE", flush=True)
