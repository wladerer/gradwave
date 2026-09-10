"""Capture a real mid-SCF Davidson round-boundary state for the native-loop probe.

Runs the production NC SCF on a small Si cell, intercepts the solver-registry
call at a chosen SCF iteration (warm regime), and snapshots everything the
batched Davidson solve consumes:

  x0 (warm block), t_solve, bk.{mask, t, flat_idx}, idx_scatter, gather_idx,
  v_eff_r box, projectors p, dij, grid shape, tol_eff.

The snapshot is exact input state: both the eager baseline (the real
``davidson_batched``) and the native C kernel are then run from the SAME state,
so agreement is checked at fp64 round-off, not "same physics eventually".

GRADWAVE_TOEPLITZ=off is forced so the eager local term is the FFT path — the
native kernel implements the FFT path, and the probe compares like with like.

Usage: uv run python experiments/native_davidson_probe/capture_state.py [small|medium]
Writes: experiments/native_davidson_probe/state_{small,medium}.npz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["GRADWAVE_TOEPLITZ"] = "off"  # must precede gradwave import (read at import)

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
A = 5.43
ROOT = Path(__file__).parents[2]
HERE = Path(__file__).parent

# small: 2-atom fcc Si primitive cell (deep latency-bound regime)
# medium: 8-atom conventional diamond cell
CONFIGS = {
    "small": dict(
        cell=A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        pos=np.array([[0.0, 0, 0], [A / 4] * 3]),
        kmesh=(4, 4, 4),
    ),
    "medium": dict(
        cell=A * np.eye(3),
        pos=A * np.array(
            [[0.0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
             [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75],
             [0.75, 0.75, 0.25]]),
        kmesh=(2, 2, 2),
    ),
}

CAPTURE_AT_CALL = 4  # solver invocations (== SCF iterations at nspin=1): warm regime


class _Stop(Exception):
    pass


def capture(tag: str) -> None:
    cfg = CONFIGS[tag]
    si = parse_upf(ROOT / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    system = setup_system(
        cfg["cell"], cfg["pos"], [0] * len(cfg["pos"]), [si],
        ecut=30 * RY, kmesh=cfg["kmesh"])

    import gradwave.solvers.registry as registry

    orig_get = registry.get
    snap: dict = {"n": 0}

    def hooked_get(name):
        solver = orig_get(name)

        def wrapped(h_apply, x0, t, mask, tol=1e-9, nbands=None):
            snap["n"] += 1
            if snap["n"] == CAPTURE_AT_CALL:
                # plain NC/PBE path: h_apply is BatchedHamiltonian.apply (bound)
                h = h_apply.__self__
                snap.update(
                    x0=x0.clone(), t_solve=t.clone(), mask=mask.clone(),
                    tol=float(tol), v_eff=h.v_eff_r.clone(), p=h.p.clone(),
                    dij=h.bk.dij_full.clone(), idx_scatter=h.idx_scatter.clone(),
                    gather_idx=h.gather_idx.clone(), bk_t=h.bk.t.clone(),
                    shape=h.shape)
                raise _Stop
            return solver(h_apply, x0, t, mask, tol=tol, nbands=nbands)

        return wrapped

    registry.get = hooked_get
    try:
        scf(system, PBE(), max_iter=CAPTURE_AT_CALL + 2, verbose=False)
    except _Stop:
        pass
    finally:
        registry.get = orig_get
    assert "x0" in snap, f"never reached solver call {CAPTURE_AT_CALL}"

    nk, nb, m = snap["x0"].shape
    out = HERE / f"state_{tag}.npz"
    np.savez_compressed(
        out,
        x0=snap["x0"].numpy(), t_solve=snap["t_solve"].numpy(),
        mask=snap["mask"].numpy(), v_eff=snap["v_eff"].numpy(),
        p=snap["p"].numpy(), dij=snap["dij"].numpy(),
        idx_scatter=snap["idx_scatter"].numpy(),
        gather_idx=snap["gather_idx"].numpy(), bk_t=snap["bk_t"].numpy(),
        shape=np.array(snap["shape"]), tol=np.array(snap["tol"]))
    print(f"{tag}: nk={nk} nb={nb} npw_max={m} box={snap['shape']} "
          f"tol_eff={snap['tol']:.2e} -> {out}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["small", "medium"]
    torch.set_num_threads(8)
    for tag in tags:
        capture(tag)
