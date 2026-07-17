"""Local Thomas–Fermi vs bare Kerker preconditioner: SCF iteration counts.

A single constant-q0 Kerker filter is the right density preconditioner for a
bulk metal but the wrong one for an inhomogeneous cell, where it over-screens
the vacuum. The local-TF preconditioner (`precond='local_tf'`) lets the
screening wavevector track the local density, so it matches bare Kerker on the
bulk and beats it on slabs, with the margin growing as the cell gets more
inhomogeneous.

Usage: uv run python benchmarks/bench_precond.py [nc|paw]

Measured on an 8-core laptop (fcc Al, PBE, gaussian 0.1 eV), NC path:

    bulk fcc Al (8x8x8)        kerker  9   local_tf  9   (neutral)
    Al(100) slab, 4 layers     kerker 21   local_tf 17   (1.24x)
    Al(100) slab, 6 layers     kerker 27   local_tf 21   (1.29x)

Energies are bit-identical between the two preconditioners (same fixed point,
different path). Iteration count, not wall time, is the trustworthy metric for a
solver-logic question (see docs/manual/performance.md).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from ase.build import fcc100

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradwave.core.xc.pbe import PBE  # noqa: E402

RY = 13.605693122994
FIX = ROOT / "tests/fixtures/qe/pseudos"
mode = sys.argv[1] if len(sys.argv) > 1 else "nc"
torch.set_num_threads(8)


def cases_nc():
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    cases = [("bulk fcc Al (8x8x8)",
              setup_system(cell, np.array([[0.0, 0, 0]]), [0], [al],
                           ecut=30 * RY, kmesh=(8, 8, 8), nbands=12,
                           use_symmetry=True))]
    for nlay in (4, 6):
        slab = fcc100("Al", size=(1, 1, nlay), a=4.05, vacuum=8.0)
        cases.append((f"Al(100) slab, {nlay} layers",
                      setup_system(np.array(slab.cell), slab.get_positions(),
                                   [0] * len(slab), [al], ecut=25 * RY,
                                   kmesh=(4, 4, 1), nbands=6 * len(slab),
                                   use_symmetry=True)))

    def run(system, precond):
        t0 = time.time()
        r = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-8,
                rhotol=1e-7, max_iter=80, verbose=False, precond=precond)
        return r.n_iter, r.converged, float(r.energies.total), time.time() - t0

    return cases, run


def cases_paw():
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    paw = parse_upf_paw(FIX / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    slab = fcc100("Al", size=(1, 1, 4), a=4.05, vacuum=5.0)
    cases = [
        ("bulk fcc Al PAW (8x8x8)",
         setup_uspp(cell, [[0.0, 0, 0]], [0], [paw], ecut=25 * RY,
                    ecutrho=120 * RY, kmesh=(8, 8, 8), nbands=12,
                    use_symmetry=True)),
        ("Al(100) PAW slab, 4 layers",
         setup_uspp(np.array(slab.cell), slab.get_positions(), [0] * len(slab),
                    [paw], ecut=25 * RY, ecutrho=120 * RY, kmesh=(4, 4, 1),
                    nbands=6 * len(slab), use_symmetry=True)),
    ]

    def run(system, precond):
        t0 = time.time()
        r = scf_uspp(system, PBE(), smearing="gaussian", width=0.1, etol=1e-8,
                     rhotol=1e-7, max_iter=80, mixing_scheme="johnson",
                     precond=precond, verbose=False)
        return r["n_iter"], r["converged"], float(r["energies"].total), \
            time.time() - t0

    return cases, run


cases, run_fn = cases_paw() if mode == "paw" else cases_nc()
print(f"=== local-TF vs Kerker ({mode.upper()}) ===")
for name, system in cases:
    print(name)
    base = None
    for p in ("kerker", "local_tf"):
        it, cv, E, t = run_fn(system, p)
        speed = "" if base is None else f"  ({base / it:.2f}x)"
        if p == "kerker":
            base = it
        print(f"  {p:10s}: {it:3d} iters  conv={cv}  E={E:.6f}  {t:6.1f}s{speed}")
