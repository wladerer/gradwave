"""Bit-reproducibility harness for the shielding k-streaming refactor.

Computes the bare NMR shielding σ (nsite,3,3) for a small Si cell via sigma_shielding_dq,
caching the norm-conserving SCF so the dev loop only re-runs the (refactored) shielding solve.
The REFERENCE run stores σ; every streaming increment must reproduce it to solver tolerance.

Env: STREAM (0=reference all-k path, else chunk size), ECUT (default 250), K (default 2),
     REF (path to store/compare the reference .npy; default ~/si_sigma_ref.npy),
     RES (path to cache the SCF result pickle; default ~/si_res.pkl).
"""
import os
import pickle
import resource
import time

import numpy as np


def _peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB (linux ru_maxrss=KB)


def main():
    from pathlib import Path

    from ase.build import bulk

    from gradwave.api.scf import run_scf
    from gradwave.inputs import Input, KPointsParams, NmrParams
    from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq

    ecut = float(os.environ.get("ECUT", "250"))
    k = int(os.environ.get("K", "2"))
    stream = int(os.environ.get("STREAM", "0"))
    ref = os.path.expanduser(os.environ.get("REF", "~/si_sigma_ref.npy"))
    respath = os.path.expanduser(os.environ.get("RES", "~/si_res.pkl"))
    pdir = Path(os.path.expanduser(os.environ.get(
        "PDIR", "~/github/gradwave/tests/fixtures/qe/pseudos")))

    if os.path.exists(respath):
        with open(respath, "rb") as f:
            res = pickle.load(f)
        print(f"# loaded cached SCF res from {respath}", flush=True)
    else:
        atoms = bulk("Si", "diamond", a=5.431)
        inp = Input(atoms=atoms, pseudo_dir=pdir, pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"},
                    ecut=ecut, task="nmr", symmetry=False,
                    kpoints=KPointsParams(mesh=[k, k, k]), nmr=NmrParams(task="shielding"))
        t0 = time.time()
        res = run_scf(inp, verbose=False)
        print(f"# SCF {time.time()-t0:.0f}s", flush=True)
        try:
            with open(respath, "wb") as f:
                pickle.dump(res, f)
            print(f"# cached res -> {respath}", flush=True)
        except Exception as e:
            print(f"# res not picklable ({type(e).__name__}); will recompute each run", flush=True)

    kw = {} if stream == 0 else {"chunk_k": stream}   # chunk_k added by the refactor
    t0 = time.time()
    sig = sigma_shielding_dq(res, **kw).detach().cpu().numpy()
    dt = time.time() - t0
    iso = [float(np.trace(0.5 * (s + s.T)) / 3.0) for s in sig]
    print(f"# STREAM={stream} shielding {dt:.0f}s peak_rss={_peak_gb():.1f}GB", flush=True)
    print(f"# sigma_iso per site (ppm): {[round(x,4) for x in iso]}", flush=True)

    if stream == 0:
        np.save(ref, sig)
        print(f"# REFERENCE saved -> {ref}", flush=True)
    else:
        ref_sig = np.load(ref)
        max_abs = float(np.max(np.abs(sig - ref_sig)))
        print(f"# vs reference: max|Δσ| = {max_abs:.3e} ppm  "
              f"{'PASS' if max_abs < 1e-6 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
