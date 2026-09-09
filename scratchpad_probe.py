"""Localize peak-RSS contributors in the Γ-real SCF path.

Wraps density_b and the per-spin eigensolve to record the peak RSS *increase*
during each, so we know which field build dominates. One config per process.
Usage: python scratchpad_probe.py <natoms> <mode>
"""
import os, sys, threading, time, resource
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests.helpers import RY, si_upf  # noqa: E402
import gradwave.scf.loop as loop  # noqa: E402
import gradwave.core.batch as batch  # noqa: E402
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
    frac = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                frac.append((CONV + [i, j, k]) / [nx, ny, nz])
    return cell, np.concatenate(frac, axis=0) @ cell


def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize()


class PeakBox:
    def __init__(self):
        self.peak = 0
        self.on = False
    def sample(self):
        if self.on:
            self.peak = max(self.peak, rss())

STATS = {"density": PeakBox(), "eig": PeakBox()}


def make_wrapper(orig, box):
    def wrapped(*a, **k):
        base = rss()
        box.on = True
        box.peak = base
        r = orig(*a, **k)
        box.on = False
        STATS.setdefault(box, 0)
        delta = box.peak - base
        wrapped.max_delta = max(getattr(wrapped, "max_delta", 0), delta)
        return r
    wrapped.max_delta = 0
    return wrapped


def main():
    natoms = int(sys.argv[1]); mode = sys.argv[2]
    torch.set_num_threads(8)
    upf = si_upf(); cell, pos = si_supercell(natoms)
    nb = natoms * 3
    system = setup_system(cell, pos, [0] * natoms, [upf], ecut=16 * RY,
                          kmesh=(1, 1, 1), nbands=nb, use_symmetry=False)
    loop._GAMMA_REAL_ENV = mode

    dbox = STATS["density"]; ebox = STATS["eig"]
    orig_density = batch.density_b
    dwrap = make_wrapper(orig_density, dbox)
    batch.density_b = dwrap
    # loop imports density_b lazily inside _output_density, so patch the module attr

    orig_solve_c = loop._solve_bands
    orig_solve_g = loop._solve_bands_gamma
    ewrap_c = make_wrapper(orig_solve_c, ebox); ewrap_c.__name__ = "c"
    ewrap_g = make_wrapper(orig_solve_g, ebox); ewrap_g.__name__ = "g"
    loop._solve_bands = ewrap_c
    loop._solve_bands_gamma = ewrap_g

    # global sampler for whole run peak
    stop = threading.Event(); allpk = [0]
    def samp():
        while not stop.is_set():
            r = rss()
            allpk[0] = max(allpk[0], r)
            for b in (dbox, ebox):
                if b.on:
                    b.peak = max(b.peak, r)
            time.sleep(0.005)
    th = threading.Thread(target=samp); th.start()
    res = scf(system, LDA_PW92(), smearing="gaussian", width=0.1,
              etol=1e-8, rhotol=1e-7, verbose=False, max_iter=100)
    stop.set(); th.join()
    print(f"natoms={natoms} mode={mode} nb={nb} n={np.prod(system.grid.shape)} "
          f"whole_peak_mb={allpk[0]/1e6:.1f} "
          f"density_maxdelta_mb={dwrap.max_delta/1e6:.1f} "
          f"eig_maxdelta_mb={max(ewrap_c.max_delta, ewrap_g.max_delta)/1e6:.1f} "
          f"niter={res.n_iter} e={float(res.energies.total):.8f}")


if __name__ == "__main__":
    main()
