"""Peak-RSS harness for the Γ-real vs complex SCF path.

Usage: python scratchpad_rss.py <natoms:16|32> <mode:0|1> [ecut_ry]
Prints one line:  natoms=.. mode=.. peak_mb=.. rss0_mb=.. delta_mb=.. e=.. niter=.. gamma_real=..
Run one config per process so resource.getrusage(RUSAGE_SELF).ru_maxrss
(lifetime peak RSS) reflects that config's peak.
"""
import os, sys, threading, time, resource
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
    if natoms == 16:
        reps = (2, 1, 1)
    elif natoms == 32:
        reps = (2, 2, 1)
    else:
        raise SystemExit(f"natoms {natoms} unsupported")
    nx, ny, nz = reps
    cell = np.diag([A0 * nx, A0 * ny, A0 * nz])
    frac = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                frac.append((CONV + [i, j, k]) / [nx, ny, nz])
    frac = np.concatenate(frac, axis=0)
    pos = frac @ cell
    return cell, pos


def sample_peak(stop, out):
    pg = os.getpid()
    mx = 0
    while not stop.is_set():
        try:
            with open(f"/proc/{pg}/statm") as f:
                rss_pages = int(f.read().split()[1])
            mx = max(mx, rss_pages * resource.getpagesize())
        except Exception:
            pass
        time.sleep(0.01)
    out[0] = mx


def rss_now():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize()


def main():
    natoms = int(sys.argv[1])
    mode = sys.argv[2]
    ecut = (float(sys.argv[3]) if len(sys.argv) > 3 else 16.0) * RY
    torch.set_num_threads(8)
    upf = si_upf()
    cell, pos = si_supercell(natoms)
    nb = natoms * 3  # 4 val e- per Si -> 2 occ bands/atom, pad to 3
    system = setup_system(cell, pos, [0] * natoms, [upf], ecut=ecut,
                          kmesh=(1, 1, 1), nbands=nb, use_symmetry=False)
    loop._GAMMA_REAL_ENV = mode
    rss0 = rss_now()
    stop = threading.Event()
    out = [0]
    th = threading.Thread(target=sample_peak, args=(stop, out))
    th.start()
    res = scf(system, LDA_PW92(), smearing="gaussian", width=0.1,
              etol=1e-8, rhotol=1e-7, verbose=False, max_iter=100)
    stop.set()
    th.join()
    peak = max(out[0], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    print(f"natoms={natoms} mode={mode} npw={system.spheres[0].npw} "
          f"grid={tuple(system.grid.shape)} nb={nb} "
          f"peak_mb={peak/1e6:.1f} rss0_mb={rss0/1e6:.1f} "
          f"delta_mb={(peak-rss0)/1e6:.1f} e={float(res.energies.total):.8f} "
          f"niter={res.n_iter} gamma_real={res.gamma_real} conv={res.converged}")


if __name__ == "__main__":
    main()
