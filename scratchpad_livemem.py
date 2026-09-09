"""Deterministic peak-field-memory harness for the Γ-real vs complex SCF path.

Samples peak LIVE torch storage bytes (unique data_ptr, so views don't double
count) during scf() — allocator/fragmentation independent, unlike RSS HWM.
Usage: python scratchpad_livemem.py <natoms:16|32> <mode:0|1> [ecut_ry]
"""
import gc, os, sys, threading, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests.helpers import RY, si_upf  # noqa: E402
import gradwave.scf.loop as loop  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402
from gradwave.core.xc.lda_pw92 import LDA_PW92  # noqa: E402

A0 = 5.43
CONV = np.array([
    [0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
    [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])


def si_supercell(natoms):
    reps = {16: (2, 1, 1), 32: (2, 2, 1)}[natoms]
    nx, ny, nz = reps
    cell = np.diag([A0 * nx, A0 * ny, A0 * nz])
    frac = [(CONV + [i, j, k]) / [nx, ny, nz]
            for i in range(nx) for j in range(ny) for k in range(nz)]
    return cell, np.concatenate(frac, axis=0) @ cell


def live_bytes():
    seen = {}
    for o in gc.get_objects():
        if isinstance(o, torch.Tensor):
            try:
                st = o.untyped_storage()
                seen[st.data_ptr()] = st.nbytes()
            except Exception:
                pass
    return sum(seen.values())


def main():
    natoms = int(sys.argv[1]); mode = sys.argv[2]
    ecut = (float(sys.argv[3]) if len(sys.argv) > 3 else 16.0) * RY
    torch.set_num_threads(8)
    upf = si_upf(); cell, pos = si_supercell(natoms)
    nb = natoms * 3
    system = setup_system(cell, pos, [0] * natoms, [upf], ecut=ecut,
                          kmesh=(1, 1, 1), nbands=nb, use_symmetry=False)
    loop._GAMMA_REAL_ENV = mode
    base = live_bytes()
    stop = threading.Event(); pk = [base]
    def samp():
        while not stop.is_set():
            pk[0] = max(pk[0], live_bytes()); time.sleep(0.003)
    th = threading.Thread(target=samp); th.start()
    res = scf(system, LDA_PW92(), smearing="gaussian", width=0.1,
              etol=1e-8, rhotol=1e-7, verbose=False, max_iter=100)
    stop.set(); th.join()
    print(f"natoms={natoms} mode={mode} npw={system.spheres[0].npw} "
          f"grid={tuple(system.grid.shape)} nb={nb} "
          f"base_mb={base/1e6:.1f} peaklive_mb={pk[0]/1e6:.1f} "
          f"deltalive_mb={(pk[0]-base)/1e6:.1f} e={float(res.energies.total):.8f} "
          f"niter={res.n_iter} gamma_real={res.gamma_real}")


if __name__ == "__main__":
    main()
