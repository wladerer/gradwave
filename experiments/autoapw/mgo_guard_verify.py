"""End-to-end verify: the conditioning guard fires on a real MgO PAW context."""
from __future__ import annotations

import sys
import warnings

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

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    kg._warn_if_ill_conditioned_uspp(ctx)
fired = [x for x in w if issubclass(x.category, kg.USPPShieldingConditioningWarning)]
print(f"guard fired: {len(fired) == 1}", flush=True)
if fired:
    print(f"message: {fired[0].message}", flush=True)
print("VERIFY_DONE", flush=True)
