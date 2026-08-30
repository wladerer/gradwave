"""Focused k-mesh / smearing-width convergence probe for the PBE moment.

Finds, per system, a (mesh, width) where the plain-PBE moment is stable to
<=0.02 muB against a mesh refinement — so the fit corrects the functional, not
a Brillouin-zone-sampling artifact.
"""
from __future__ import annotations

import time

import experiments.spin_adapted_pbe.train as T
from gradwave.core.xc.spin import SpinPBE
from gradwave.scf.loop import scf


def run(spec, km, width):
    sysm = T.build(spec, km)
    xc = SpinPBE()
    t = time.time()
    res = scf(sysm, xc, nspin=2, smearing="mp1", width=width,
              start_mag=[spec.start_mag], max_iter=T.MAX_ITER, verbose=False)
    return abs(float(res.mag_total)), res.converged, res.n_iter, time.time() - t


PLAN = {
    "Fe": [((10, 10, 10), 0.1), ((12, 12, 12), 0.1), ((14, 14, 14), 0.1),
           ((16, 16, 16), 0.1),
           ((12, 12, 12), 0.2), ((14, 14, 14), 0.2), ((16, 16, 16), 0.2)],
    "Ni": [((12, 12, 12), 0.1), ((14, 14, 14), 0.1), ((16, 16, 16), 0.1),
           ((12, 12, 12), 0.2), ((14, 14, 14), 0.2)],
    "Co": [((12, 12, 12), 0.1), ((14, 14, 14), 0.1), ((16, 16, 16), 0.1),
           ((12, 12, 12), 0.2), ((14, 14, 14), 0.2)],
}

for key, plan in PLAN.items():
    spec = T.SYSTEMS[key]
    T._log(f"=== {key} (exp {spec.exp_moment}) ===")
    for km, width in plan:
        m, conv, nit, dt = run(spec, km, width)
        T._log(f"  k={km} w={width}: |m|={m:.4f}  conv={conv} nit={nit} ({dt:.0f}s)")
T._log("PROBE DONE")
