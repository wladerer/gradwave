"""NMR shielding-tensor wall-clock benchmark: Si PBE, finite-q ±q assembly
(and optionally the analytic dq route) on an unreduced time_reversal=False
mesh — the sigma-dense-prepass A/B rig.

Usage:
    uv run python benchmarks/bench_sigma_shielding.py MESH [threads] [dense]
                                                      [workers] [route] [ecut_ry]

    MESH     k per axis (2 -> (2,2,2) unreduced)
    threads  torch intra-op threads          [default 8]
    dense    on|off|auto  (GRADWAVE_DENSE_STERNHEIMER)  [default auto]
    workers  spawn-pool width for the (axis, ±q) pipelines [default 1 = serial]
    route    finite|dq|both                  [default finite]
    ecut_ry  cutoff in Ry                    [default 12]

The executable body lives under ``if __name__ == "__main__"``: with ``workers
> 1`` the pool uses the spawn start method, so each worker RE-IMPORTS this
module. Without the guard the child would re-run the whole SCF at import
(recursively spawning), which is exactly the ``BrokenProcessPool`` /
``freeze_support`` failure the guard prevents. Under spawn the child's
``sys.argv`` is also the bootstrap argv, not ours, so the arg parsing must not
run at import either — hence everything is inside :func:`main`.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    mesh = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dense = sys.argv[3] if len(sys.argv) > 3 else "auto"
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    route = sys.argv[5] if len(sys.argv) > 5 else "finite"
    ecut_ry = float(sys.argv[6]) if len(sys.argv) > 6 else 12.0

    os.environ["GRADWAVE_DENSE_STERNHEIMER"] = dense
    torch.set_num_threads(threads)

    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    ry = 13.605693122994
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    root = Path(__file__).parents[1]
    si = parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")

    print(f"# mesh=({mesh},{mesh},{mesh}) threads={threads} dense={dense} "
          f"workers={workers} route={route} ecut={ecut_ry}Ry", flush=True)

    t0 = time.time()
    system = setup_system(cell, pos, [0, 0], [si], ecut=ecut_ry * ry,
                          kmesh=(mesh, mesh, mesh), nbands=8,
                          use_symmetry=False, time_reversal=False)
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    npws = [s.npw for s in system.spheres]
    print(f"# scf {time.time() - t0:.1f}s  nk={len(npws)} "
          f"npw=[{min(npws)},{max(npws)}]", flush=True)

    if route in ("finite", "both"):
        from gradwave.postscf.kgeometry_nmr import sigma_shielding

        t0 = time.time()
        sig = sigma_shielding(res, cg_tol=1e-9,
                              n_workers=(workers if workers > 1 else None))
        wall = time.time() - t0
        iso = torch.diagonal(sig, dim1=1, dim2=2).mean(dim=1)
        print(f"RESULT finite mesh={mesh} dense={dense} workers={workers} "
              f"wall={wall:.1f}s iso={[f'{v:.3f}' for v in iso.tolist()]}",
              flush=True)

    if route in ("dq", "both"):
        from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq

        t0 = time.time()
        sig = sigma_shielding_dq(res)
        wall = time.time() - t0
        iso = torch.diagonal(sig, dim1=1, dim2=2).mean(dim=1)
        print(f"RESULT dq mesh={mesh} wall={wall:.1f}s "
              f"iso={[f'{v:.3f}' for v in iso.tolist()]}", flush=True)


if __name__ == "__main__":
    main()
